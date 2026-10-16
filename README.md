# bio-entropic-timing

A small Python engine that builds a one-week (168 h) activity profile from biological
rhythms and turns it into
- a **macro state** (`ACTIVE` / `REST`) for any moment of the week, and
- **log-normal delays** whose spread depends on the current activity level.

The aim is a timing model with daily and weekly structure instead of a flat
pseudo-random delay.

## The model

| Component | Form | Anchoring |
|---|---|---|
| Circadian | 24 h sine, trough ≈ 04:00, peak ≈ 16:00 | wall-clock time |
| Weekly | ±15 % smooth modulation | Monday 00:00 |
| Ultradian | 90 min sine (basic rest–activity cycle), amplitude 0.15 | profile start |
| Noise | Gaussian white noise, σ = 0.08 (configurable) | — |

The sum is clipped to [0, 1] at 5-minute resolution (2016 points). Below
`active_threshold` (default 0.45) the state is `REST`.

Delays: `base × LogNormal(0, σ)`, where `σ = micro_jitter_std × (1.5 − 0.7 × activity)`.
Lower activity → wider spread and a longer tail. The multiplier is clipped to
[0.35, 3.8]; these are design bounds, not measured human values.

**The synthetic profile is a parametric model, not measured data.** A real series can
be loaded with `load_telemetry()` (CSV or Parquet with a `timestamp` column and an
`activity_index` column, or any numeric column, which is then min-max normalised).

## Usage

```bash
pip install -r requirements.txt
python -m pytest -q
PYTHONPATH=. python examples/demo.py
```

```python
from datetime import datetime
from bio_entropic_timing import BioEntropyEngine

eng = BioEntropyEngine(seed=42)
eng.get_current_bio_state()            # {'macro_state': 'ACTIVE', 'activity_index': 0.77, ...}
eng.calculate_inter_action_delay(2.0)  # e.g. 2.29 (seconds)
```

## Tests

`tests/test_engine.py` checks the profile size and range, seed reproducibility,
wall-clock anchoring of the circadian trough and peak, that the ultradian component
really has a 90-minute period (FFT), the jitter bounds, that lower activity gives a
wider spread, and phase anchoring of a loaded series.

## Origin

Extracted from an earlier, non-public project of the author; that project's browser
automation is not part of this repository. Fixes compared with the original code:
the ultradian term had a 3.75-minute period instead of 90 minutes, the weekly wave was
not anchored to the day of the week, and the noise was described as pink (1/f)
although it was white.

---

**Magyarul röviden:** biológiai ritmusokból (napi, heti és 90 perces ciklusból) épített
heti aktivitásprofil, amely megadja, hogy egy adott időpont aktív vagy pihenő fázis-e,
és ehhez igazodó, log-normális eloszlású késleltetést ad. A szintetikus profil modell,
nem mért adat.

Szőke László-Ferenc (LemonScripter) · laszlo.szoke@metaspace.bio
