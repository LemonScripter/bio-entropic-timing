"""Prints a one-week activity summary and a few sample delays."""
from datetime import datetime, timedelta

import pandas as pd

from bio_entropic_timing import BioEntropyEngine

start = datetime(2026, 1, 5)  # Monday 00:00
eng = BioEntropyEngine(reference_start_time=start, seed=42)

df = eng.telemetry_df
hourly = df.groupby(pd.to_datetime(df["timestamp"]).dt.hour)["activity_index"].mean()
print("Mean activity by hour of day:")
for hour, val in hourly.items():
    print(f"  {hour:02d}:00  {val:4.2f}  {'#' * int(val * 40)}")

print(f"\nShare of the week in REST state: {df['resting_phase'].mean():.0%}")

for h in (4, 10, 16):
    t = start + timedelta(hours=h)
    state = eng.get_current_bio_state(t)
    delays = [eng.calculate_inter_action_delay(2.0, t) for _ in range(5)]
    print(f"\n{t:%a %H:%M}  {state['macro_state']:<6} activity={state['activity_index']:.2f}")
    print("  sample delays (s):", ", ".join(f"{d:.2f}" for d in delays))
