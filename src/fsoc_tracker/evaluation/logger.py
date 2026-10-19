
import csv
import json
import math
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


class PerformanceLogger:
    """Record per-frame tracking and controller diagnostics."""

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
            "requested_pan_deg",
            "requested_tilt_deg",
            "applied_pan_deg",
            "applied_tilt_deg",
            "pan_saturated",
            "tilt_saturated",
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
        control_info=None,
    ):
        info = control_info or {}

        def number(value):
            return (
                round(float(value), 5)
                if value is not None else ""
            )

        row = {
            "elapsed_s": number(elapsed_s),
            "detected": bool(detected),
            "auto_track": bool(auto_track),
            "pixel_error_x": number(error_x),
            "pixel_error_y": number(error_y),
            "pan_deg": number(pan_deg),
            "tilt_deg": number(tilt_deg),
            "processing_ms": number(processing_ms),
            "fps": number(fps),
            "requested_pan_deg": number(
                info.get("requested_pan_deg")
            ),
            "requested_tilt_deg": number(
                info.get("requested_tilt_deg")
            ),
            "applied_pan_deg": number(
                info.get("applied_pan_deg")
            ),
            "applied_tilt_deg": number(
                info.get("applied_tilt_deg")
            ),
            "pan_saturated": bool(
                info.get("pan_saturated", False)
            ),
            "tilt_saturated": bool(
                info.get("tilt_saturated", False)
            ),
        }

        self.rows.append(row)

    def save(self):
        with self.csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=self.fields
            )
            writer.writeheader()
            writer.writerows(self.rows)

        tracking = [
            row for row in self.rows
            if row["auto_track"] and row["detected"]
        ]

        errors = [
            math.hypot(
                float(row["pixel_error_x"]),
                float(row["pixel_error_y"]),
            )
            for row in tracking
        ]

        processing = [
            float(row["processing_ms"])
            for row in self.rows
        ]

        duration = (
            float(self.rows[-1]["elapsed_s"])
            if self.rows else 0.0
        )

        summary = {
            "generated_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "duration_s": round(duration, 3),
            "frames_logged": len(self.rows),
            "detection_count": sum(
                row["detected"] for row in self.rows
            ),
            "auto_track_frames": sum(
                row["auto_track"] for row in self.rows
            ),
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
            "frames_over_10px": sum(
                error > 10 for error in errors
            ),
            "detection_rate_all_frames": (
                round(
                    summary_count / len(self.rows), 4
                )
                if self.rows
                else None
            ) if False else None,
            "mean_processing_ms": (
                round(float(np.mean(processing)), 4)
                if processing else None
            ),
            "p95_processing_ms": (
                round(float(np.percentile(processing, 95)), 4)
                if processing else None
            ),
            "pan_saturation_frames": sum(
                row["pan_saturated"] for row in self.rows
            ),
            "tilt_saturation_frames": sum(
                row["tilt_saturated"] for row in self.rows
            ),
            "csv_file": str(self.csv_path),
            "metric_note": (
                "Pixel error is measured relative to image center. "
                "Saturation counts are based on controller requests. "
                "Ground-truth tracking accuracy is not established."
            ),
        }

        detections = summary["detection_count"]
        summary["detection_rate_all_frames"] = (
            round(detections / len(self.rows), 4)
            if self.rows else None
        )

        self.json_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print("\nPerformance reports saved:")
        print(f"CSV:  {self.csv_path.resolve()}")
        print(f"JSON: {self.json_path.resolve()}")
        print(json.dumps(summary, indent=2))

        return summary
