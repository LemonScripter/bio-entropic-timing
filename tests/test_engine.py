from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from bio_entropic_timing import BioEntropyEngine, JITTER_CLIP, ULTRADIAN_PERIOD_HOURS

MONDAY = datetime(2026, 1, 5, 0, 0)  # a Monday


def test_profile_shape_and_range():
    eng = BioEntropyEngine(reference_start_time=MONDAY, seed=1)
    assert len(eng.activity_profile) == 168 * 60 // 5
    assert eng.activity_profile.min() >= 0.0 and eng.activity_profile.max() <= 1.0


def test_seed_is_reproducible():
    a = BioEntropyEngine(reference_start_time=MONDAY, seed=7).activity_profile
    b = BioEntropyEngine(reference_start_time=MONDAY, seed=7).activity_profile
    assert np.array_equal(a, b)


def test_circadian_anchored_to_wall_clock():
    # Start at an odd hour: the trough must still be near 04:00, the peak near 16:00.
    eng = BioEntropyEngine(reference_start_time=datetime(2026, 1, 7, 13, 25), seed=3)
    hod = pd.to_datetime(eng.telemetry_df["timestamp"]).dt.hour
    at_4 = eng.telemetry_df.loc[hod == 4, "activity_index"].mean()
    at_16 = eng.telemetry_df.loc[hod == 16, "activity_index"].mean()
    assert at_4 < 0.2 < 0.6 < at_16


def test_ultradian_period_is_90_minutes():
    # Without noise, the strongest component faster than 12 h must be the 1.5 h wave.
    eng = BioEntropyEngine(reference_start_time=MONDAY, noise_std=0.0, seed=0)
    x = eng.activity_profile - eng.activity_profile.mean()
    freqs = np.fft.rfftfreq(len(x), d=5 / 60)  # cycles per hour
    power = np.abs(np.fft.rfft(x))
    fast = freqs > 1 / 12
    peak_period = 1 / freqs[fast][np.argmax(power[fast])]
    assert abs(peak_period - ULTRADIAN_PERIOD_HOURS) < 0.05


def test_jitter_stays_within_bounds():
    eng = BioEntropyEngine(reference_start_time=MONDAY, seed=2)
    t = MONDAY + timedelta(hours=40)
    vals = np.array([eng.generate_micro_jitter(1000.0, t) for _ in range(5000)])
    assert vals.min() >= 1000 * JITTER_CLIP[0] and vals.max() <= 1000 * JITTER_CLIP[1]


def test_lower_activity_gives_wider_spread():
    eng = BioEntropyEngine(reference_start_time=MONDAY, seed=4)
    df = eng.telemetry_df
    t_low = pd.Timestamp(df.loc[df["activity_index"].idxmin(), "timestamp"]).to_pydatetime()
    t_high = pd.Timestamp(df.loc[df["activity_index"].idxmax(), "timestamp"]).to_pydatetime()
    low = np.log([eng.generate_micro_jitter(1000.0, t_low) for _ in range(4000)])
    high = np.log([eng.generate_micro_jitter(1000.0, t_high) for _ in range(4000)])
    assert low.std() > high.std()


def test_loaded_series_is_phase_anchored(tmp_path):
    src = BioEntropyEngine(reference_start_time=MONDAY, seed=5)
    path = tmp_path / "profile.csv"
    src.telemetry_df.to_csv(path, index=False)
    eng = BioEntropyEngine(telemetry_file=path)
    assert eng.reference_start_time == MONDAY
    state = eng.get_current_bio_state(MONDAY + timedelta(hours=16))
    assert state["macro_state"] in ("ACTIVE", "REST")
