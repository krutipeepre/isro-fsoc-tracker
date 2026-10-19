
import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path("results/tracking_20261019_115003.csv")
df = pd.read_csv(csv_path)

print("\n--- DATA OVERVIEW ---")
print("Rows:", len(df))
print("Columns:", list(df.columns))

tracking = df[
    (df["auto_track"] == True) &
    (df["detected"] == True)
].copy()

tracking["pixel_error"] = np.hypot(
    tracking["pixel_error_x"],
    tracking["pixel_error_y"]
)

print("\n--- WORST 10 TRACKING ERRORS ---")
print(
    tracking.nlargest(10, "pixel_error")[
        [
            "elapsed_s",
            "pixel_error_x",
            "pixel_error_y",
            "pixel_error",
            "pan_deg",
            "tilt_deg",
            "processing_ms",
        ]
    ].to_string(index=False)
)

print("\n--- ERROR DISTRIBUTION ---")
print(tracking["pixel_error"].describe(
    percentiles=[0.50, 0.90, 0.95, 0.99]
))

print("\n--- FRAMES ABOVE 10 PIXELS ---")
bad = tracking[tracking["pixel_error"] > 10]
print("Count:", len(bad))
print("Percentage:", round(100 * len(bad) / max(1, len(tracking)), 3), "%")
