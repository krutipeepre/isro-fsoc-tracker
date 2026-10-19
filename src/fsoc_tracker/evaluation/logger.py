
import csv
import json
import math
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


class PerformanceLogger:
    """Record tracking telemetry and generate run-level reports."""

    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"tracking_{stamp}.csv"
        self.json_path = self.output_dir / f"summary_{stamp}.json"

        self.start_time = time.perf_counter()
        self.rows = []

        self.fields = [
            "elapsed_s",
            "detected",
            "auto_track",
            "pixel_error_x",
            "pixel_error_y",
            "pan_deg",
            "tilt_deg",
            "processing_ms",
            "fps",
        ]

    def record(
        self,
        elapsed_s,
        detected,
        auto_track,
        error_x,
        error_y,
        pan_deg,
        tilt_deg,
        processing_ms,
        fps,
    ):
        row = {
            "elapsed_s": round(float(elapsed_s), 5),
            "detected": bool(detected),
            "auto_track": bool(auto_track),
            "pixel_error_x": (
                round(float(error_x), 4)
                if error_x is not None else ""
            ),
            "pixel_error_y": (
                round(float(error_y), 4)
                if error_y is not None else ""
            ),
            "pan_deg": round(float(pan_deg), 5),
            "tilt_deg": round(float(tilt_deg), 5),
            "processing_ms": round(float(processing_ms), 4),
            "fps": round(float(fps), 3),
        }
        self.rows.append(row)

    def save(self):
        """Save per-frame CSV and a summary JSON."""
        with self.csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(self.rows)

        tracking_rows = [
            row for row in self.rows
            if row["auto_track"] and row["detected"]
        ]

        errors = [
            math.hypot(
                row["pixel_error_x"],
                row["pixel_error_y"],
            )
            for row in tracking_rows
        ]

        detected_frames = sum(row["detected"] for row in self.rows)
        auto_frames = sum(row["auto_track"] for row in self.rows)
        eligible_frames = sum(
            row["auto_track"] for row in self.rows
        )

        processing_times = [
            row["processing_ms"] for row in self.rows
        ]

        summary = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(
                time.perf_counter() - self.start_time, 3
            ),
            "frames_logged": len(self.rows),
            "detection_count": int(detected_frames),
            "auto_track_frames": int(auto_frames),
            "tracking_error_sample_count": len(errors),
            "mean_pixel_error_px": (
                round(float(np.mean(errors)), 4)
                if errors else None
            ),
            "p95_pixel_error_px": (
                round(float(np.percentile(errors, 95)), 4)
                if errors else None
            ),
            "max_pixel_error_px": (
                round(float(np.max(errors)), 4)
                if errors else None
            ),
            "detection_rate_all_frames": (
                round(detected_frames / len(self.rows), 4)
                if self.rows else None
            ),
            "mean_processing_ms": (
                round(float(np.mean(processing_times)), 4)
                if processing_times else None
            ),
            "p95_processing_ms": (
                round(float(np.percentile(processing_times, 95)), 4)
                if processing_times else None
            ),
            "metric_note": (
                "Pixel error is distance from detected beacon to "
                "image center, measured only on frames with auto-track "
                "enabled and a detection. It is not ground-truth "
                "angular tracking error or lock retention."
            ),
            "csv_file": str(self.csv_path),
        }

        self.json_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print("\nPerformance report saved:")
        print(f"CSV:  {self.csv_path.resolve()}")
        print(f"JSON: {self.json_path.resolve()}")
        print(json.dumps(summary, indent=2))

        return summary
