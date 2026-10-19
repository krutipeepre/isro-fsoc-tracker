
import csv
import json
import math
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


class PerformanceLogger:
    """Record tracking performance against synthetic ground truth."""

    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"tracking_{stamp}.csv"
        self.json_path = self.output_dir / f"summary_{stamp}.json"

        self.start_time = time.perf_counter()
        self.rows = []
        self._saved = False

        self.fields = [
            "elapsed_s", "phase", "target_visible", "target_in_frame",
            "detected", "auto_track",
            "detected_x", "detected_y",
            "ground_truth_x", "ground_truth_y",
            "pixel_error_x", "pixel_error_y", "pixel_error_px",
            "localization_error_px",
            "ground_truth_pan_error_deg", "ground_truth_tilt_error_deg",
            "pan_deg", "tilt_deg",
            "processing_ms", "fps",
            "requested_pan_deg", "requested_tilt_deg",
            "applied_pan_deg", "applied_tilt_deg",
            "pan_saturated", "tilt_saturated",
        ]

    @staticmethod
    def _number(value):
        if value is None:
            return ""
        value = float(value)
        return round(value, 6) if math.isfinite(value) else ""

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
        *,
        phase="UNKNOWN",
        target_visible=True,
        target_in_frame=None,
        detected_x=None,
        detected_y=None,
        ground_truth_x=None,
        ground_truth_y=None,
        ground_truth_pan_error_deg=None,
        ground_truth_tilt_error_deg=None,
    ):
        info = control_info or {}

        pixel_error = (
            math.hypot(float(error_x), float(error_y))
            if error_x is not None and error_y is not None
            else None
        )

        localization_error = None
        if (
            detected
            and detected_x is not None
            and detected_y is not None
            and ground_truth_x is not None
            and ground_truth_y is not None
        ):
            localization_error = math.hypot(
                float(detected_x) - float(ground_truth_x),
                float(detected_y) - float(ground_truth_y),
            )

        row = {
            "elapsed_s": self._number(elapsed_s),
            "phase": str(phase),
            "target_visible": bool(target_visible),
            "target_in_frame": (
                bool(target_in_frame)
                if target_in_frame is not None else ""
            ),
            "detected": bool(detected),
            "auto_track": bool(auto_track),
            "detected_x": self._number(detected_x),
            "detected_y": self._number(detected_y),
            "ground_truth_x": self._number(ground_truth_x),
            "ground_truth_y": self._number(ground_truth_y),
            "pixel_error_x": self._number(error_x),
            "pixel_error_y": self._number(error_y),
            "pixel_error_px": self._number(pixel_error),
            "localization_error_px": self._number(localization_error),
            "ground_truth_pan_error_deg": self._number(
                ground_truth_pan_error_deg
            ),
            "ground_truth_tilt_error_deg": self._number(
                ground_truth_tilt_error_deg
            ),
            "pan_deg": self._number(pan_deg),
            "tilt_deg": self._number(tilt_deg),
            "processing_ms": self._number(processing_ms),
            "fps": self._number(fps),
            "requested_pan_deg": self._number(
                info.get("requested_pan_deg")
            ),
            "requested_tilt_deg": self._number(
                info.get("requested_tilt_deg")
            ),
            "applied_pan_deg": self._number(
                info.get("applied_pan_deg")
            ),
            "applied_tilt_deg": self._number(
                info.get("applied_tilt_deg")
            ),
            "pan_saturated": bool(info.get("pan_saturated", False)),
            "tilt_saturated": bool(info.get("tilt_saturated", False)),
        }

        self.rows.append(row)

    @staticmethod
    def _stats(values):
        values = [float(v) for v in values if v is not None]
        if not values:
            return {
                "sample_count": 0,
                "mean": None,
                "median": None,
                "p95": None,
                "max": None,
            }
        return {
            "sample_count": len(values),
            "mean": round(float(np.mean(values)), 4),
            "median": round(float(np.median(values)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "max": round(float(np.max(values)), 4),
        }

    def save(self):
        if self._saved:
            return self.last_summary

        with self.csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(self.rows)

        total = len(self.rows)
        detections = sum(bool(r["detected"]) for r in self.rows)
        visible_in_frame = [
            r for r in self.rows
            if r["target_visible"] and r["target_in_frame"] is True
        ]
        expected_count = len(visible_in_frame)
        true_detections = sum(r["detected"] for r in visible_in_frame)

        false_detections = sum(
            r["detected"] and (
                not r["target_visible"] or r["target_in_frame"] is False
            )
            for r in self.rows
        )

        tracking = [
            r for r in self.rows
            if r["auto_track"] and r["detected"]
            and r["pixel_error_px"] != ""
        ]
        tracking_errors = [r["pixel_error_px"] for r in tracking]

        localization_errors = [
            r["localization_error_px"] for r in self.rows
            if r["localization_error_px"] != ""
        ]

        angular_pan_errors = [
            abs(r["ground_truth_pan_error_deg"]) for r in tracking
            if r["ground_truth_pan_error_deg"] != ""
        ]
        angular_tilt_errors = [
            abs(r["ground_truth_tilt_error_deg"]) for r in tracking
            if r["ground_truth_tilt_error_deg"] != ""
        ]

        processing = [
            r["processing_ms"] for r in self.rows
            if r["processing_ms"] != ""
        ]

        duration = (
            float(self.rows[-1]["elapsed_s"]) if self.rows else 0.0
        )

        summary = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(duration, 3),
            "frames_logged": total,
            "detection_count_all_frames": detections,
            "detection_rate_all_frames": (
                round(detections / total, 4) if total else None
            ),
            "visible_in_frame_expected_detection_frames": expected_count,
            "detections_when_target_visible_and_in_frame": true_detections,
            "detection_recall_when_target_visible_and_in_frame": (
                round(true_detections / expected_count, 4)
                if expected_count else None
            ),
            "false_detections_outside_expected_visibility": false_detections,
            "auto_track_frames": sum(r["auto_track"] for r in self.rows),
            "tracking_error_px": self._stats(tracking_errors),
            "ground_truth_localization_error_px": self._stats(
                localization_errors
            ),
            "absolute_ground_truth_pan_error_deg": self._stats(
                angular_pan_errors
            ),
            "absolute_ground_truth_tilt_error_deg": self._stats(
                angular_tilt_errors
            ),
            "tracking_frames_over_5px": sum(
                e > 5 for e in tracking_errors
            ),
            "tracking_frames_over_10px": sum(
                e > 10 for e in tracking_errors
            ),
            "processing_ms": self._stats(processing),
            "pan_saturation_frames": sum(
                r["pan_saturated"] for r in self.rows
            ),
            "tilt_saturation_frames": sum(
                r["tilt_saturated"] for r in self.rows
            ),
            "csv_file": str(self.csv_path),
            "metric_notes": [
                "Pixel tracking error is distance from detected beacon to image center.",
                "Localization error compares detected beacon coordinates with synthetic ground-truth coordinates.",
                "Ground-truth angular errors are computed from target angles and camera orientation.",
                "Detection recall only uses frames where the target is visible and inside the camera field of view.",
                "Intentional beacon-loss frames are not counted as expected detections.",
                "Processing time measures the timed frame-processing section, not full display-loop latency.",
                "Results are from a synthetic simulator and do not establish real FSOC hardware performance."
            ],
        }

        self.json_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        self.last_summary = summary
        self._saved = True

        print("\nPerformance reports saved:")
        print(f"CSV:  {self.csv_path.resolve()}")
        print(f"JSON: {self.json_path.resolve()}")
        print(json.dumps(summary, indent=2))

        return summary
