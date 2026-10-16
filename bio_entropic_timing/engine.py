"""
Bio-entropic timing engine.

Builds a 168-hour (one week) activity profile from three biological rhythms and
turns it into
  - a macro state (ACTIVE / REST) for any moment of the week, and
  - log-normally distributed delays whose spread depends on the current activity.

The profile is either synthesised parametrically (the default, see
`_generate_synthetic_profile`) or loaded from an external CSV/Parquet series via
`load_telemetry`. The synthetic profile is a model, not measured data.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("BioEntropyEngine")

ULTRADIAN_PERIOD_HOURS = 1.5   # basic rest-activity cycle (BRAC), ~90 min
JITTER_CLIP = (0.35, 3.8)      # design bounds for the delay multiplier


class BioEntropyEngine:
    """Maps a 168-hour activity profile to macro states and micro-jitter delays."""

    def __init__(
        self,
        telemetry_file: Optional[Path] = None,
        macro_cycle_hours: int = 168,
        active_threshold: float = 0.45,
        micro_jitter_std: float = 0.25,
        noise_std: float = 0.08,
        reference_start_time: Optional[datetime] = None,
        seed: Optional[int] = None,
    ):
        self.macro_cycle_hours = macro_cycle_hours
        self.active_threshold = active_threshold
        self.micro_jitter_std = micro_jitter_std
        self.noise_std = noise_std
        self.reference_start_time = reference_start_time or datetime.now()
        self.rng = np.random.default_rng(seed)

        self.telemetry_df: pd.DataFrame = pd.DataFrame()
        self.activity_profile: np.ndarray = np.array([])
        self.resolution_minutes: int = 5

        if telemetry_file and Path(telemetry_file).exists():
            self.load_telemetry(Path(telemetry_file))
        else:
            self._generate_synthetic_profile(telemetry_file)

    def _generate_synthetic_profile(self, target_path: Optional[Path] = None):
        """
        Synthesises a one-week profile at 5-minute resolution:
          - 24 h circadian wave, anchored to wall-clock time (trough ~04:00, peak ~16:00),
          - smooth weekly modulation (±15 %), anchored to Monday 00:00,
          - 90-minute ultradian wave (amplitude 0.15),
          - Gaussian white noise (`noise_std`).
        The sum is clipped to [0, 1].
        """
        logger.info("Synthesising 168-hour parametric activity profile (model, not measured data)")
        n_points = self.macro_cycle_hours * 60 // self.resolution_minutes
        time_index = pd.date_range(
            start=self.reference_start_time,
            periods=n_points,
            freq=f"{self.resolution_minutes}min",
        )
        t_hours = np.arange(n_points) * self.resolution_minutes / 60.0

        start = self.reference_start_time
        start_hod = start.hour + start.minute / 60.0 + start.second / 3600.0
        hour_of_day = (start_hod + t_hours) % 24.0
        hour_of_week = (start.weekday() * 24.0 + start_hod + t_hours) % 168.0

        circadian_wave = 0.5 * (1.0 + np.sin(2 * np.pi * (hour_of_day - 10.0) / 24.0))
        weekly_wave = 0.85 + 0.15 * np.sin(2 * np.pi * hour_of_week / 168.0)
        ultradian_wave = 0.15 * np.sin(2 * np.pi * t_hours / ULTRADIAN_PERIOD_HOURS)
        noise = self.rng.normal(0.0, self.noise_std, n_points)

        activity_index = np.clip(circadian_wave * weekly_wave + ultradian_wave + noise, 0.0, 1.0)

        self.telemetry_df = pd.DataFrame({
            "timestamp": time_index,
            "hour_of_week": hour_of_week,
            "activity_index": activity_index,
            "resting_phase": (activity_index < self.active_threshold).astype(int),
            "activity_variance": np.abs(np.gradient(activity_index)),
        })
        self.activity_profile = activity_index

        if target_path:
            target_path = Path(target_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            self.telemetry_df.to_csv(target_path, index=False)
            logger.info(f"Saved synthetic profile to: {target_path}")

    def load_telemetry(self, file_path: Path):
        """Loads an activity series from CSV or Parquet (Parquet needs pyarrow)."""
        logger.info(f"Loading activity series from {file_path}")
        if file_path.suffix == ".parquet":
            self.telemetry_df = pd.read_parquet(file_path)
        else:
            self.telemetry_df = pd.read_csv(file_path)

        # Anchor the phase to the series' own start time; otherwise the rest window
        # would move with every process start.
        if "timestamp" in self.telemetry_df.columns:
            try:
                first_ts = pd.to_datetime(self.telemetry_df["timestamp"].iloc[0])
                self.reference_start_time = first_ts.to_pydatetime()
            except (ValueError, TypeError) as exc:
                logger.warning(f"Unparseable timestamp column, keeping process-start phase: {exc}")

        if "activity_index" not in self.telemetry_df.columns:
            # Min-max normalise the first numeric column
            numeric_cols = self.telemetry_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("Telemetry file has no numeric activity column.")
            col = self.telemetry_df[numeric_cols[0]]
            self.telemetry_df["activity_index"] = (col - col.min()) / (col.max() - col.min() + 1e-6)

        self.activity_profile = self.telemetry_df["activity_index"].to_numpy()

    def get_current_bio_state(self, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Returns the macro state (ACTIVE / REST) for the given moment of the cycle."""
        now = current_time or datetime.now()
        delta_hours = (now - self.reference_start_time).total_seconds() / 3600.0
        cycle_hour = delta_hours % self.macro_cycle_hours

        total_steps = len(self.activity_profile)
        idx = int((cycle_hour / self.macro_cycle_hours) * total_steps) % total_steps

        activity_val = float(self.activity_profile[idx])
        is_active = activity_val >= self.active_threshold
        variance = (float(self.telemetry_df["activity_variance"].iloc[idx])
                    if "activity_variance" in self.telemetry_df else 0.05)

        return {
            "timestamp": now.isoformat(),
            "cycle_hour": cycle_hour,
            "activity_index": activity_val,
            "macro_state": "ACTIVE" if is_active else "REST",
            "is_active": is_active,
            "activity_variance": variance,
        }

    def generate_micro_jitter(self, base_duration_ms: float,
                              current_time: Optional[datetime] = None) -> float:
        """
        Scales `base_duration_ms` by a log-normal multiplier. Higher activity gives a
        narrower spread (sigma shrinks), lower activity a wider one with a longer tail.
        The multiplier is clipped to JITTER_CLIP.
        """
        activity = self.get_current_bio_state(current_time)["activity_index"]
        sigma = self.micro_jitter_std * (1.5 - activity * 0.7)
        multiplier = np.clip(self.rng.lognormal(mean=0.0, sigma=sigma), *JITTER_CLIP)
        return round(float(base_duration_ms * multiplier), 2)

    def calculate_inter_action_delay(self, base_seconds: float = 2.0,
                                     current_time: Optional[datetime] = None) -> float:
        """Delay in seconds between two actions, based on the current activity."""
        return self.generate_micro_jitter(base_seconds * 1000.0, current_time) / 1000.0
