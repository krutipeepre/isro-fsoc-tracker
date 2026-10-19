
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

csv_path = Path("results/tracking_20261019_115003.csv")
df = pd.read_csv(csv_path)

tracking = df[
    (df["auto_track"] == True) &
    (df["detected"] == True)
].copy()

tracking["pixel_error"] = np.hypot(
    tracking["pixel_error_x"],
    tracking["pixel_error_y"]
)

plt.figure(figsize=(12, 5))
plt.plot(
    tracking["elapsed_s"],
    tracking["pixel_error"],
    linewidth=0.8
)
plt.axhline(10, linestyle="--", label="10 px threshold")
plt.xlabel("Elapsed time (seconds)")
plt.ylabel("Pixel-centering error (pixels)")
plt.title("FSOC Virtual Camera — Tracking Error Over Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/tracking_error.png", dpi=150)
plt.show()

print("Plot saved to results/tracking_error.png")
