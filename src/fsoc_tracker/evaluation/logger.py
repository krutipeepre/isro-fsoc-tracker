
"""
FSOC performance logger.

Responsibilities:
- Record frame-by-frame tracking and camera telemetry.
- Compare detected beacon coordinates with supplied ground-truth coordinates.
- Calculate detection, tracking, localization and processing statistics.
- Calculate phase-specific statistics for WARMUP, BEACON_LOSS
  and REACQUISITION.
- Save detailed CSV data and a JSON summary.

Important:
This logger does not generate disturbances or control the camera.
Ground-truth accuracy depends on the coordinates supplied by main.py.
"""

import csv
import json
import math
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


class PerformanceLogger:
    """Record tracking performance against supplied synthetic ground truth."""

    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"tracking_{stamp}.csv"
        self.json_path = self.output_dir / f"summary_{stamp}.json"

        self.start_time = time.perf_counter()
        self.rows = []
        self._saved = False
        self.last_summary = None

        # Keep the existing CSV schema compatible with main.py.
        self.fields = [
            "elapsed_s",
            "phase",
            "target_visible",
            "target_in_frame",
            "detected",
            "auto_track",
            "detected_x",
            "detected_y",
            "ground_truth_x",
            "ground_truth_y",
            "pixel_error_x",
            "pixel_error_y",
            "pixel_error_px",
            "localization_error_px",
            "ground_truth_pan_error_deg",
            "ground_truth_tilt_error_deg",
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

    @staticmethod
    def _number(value):
        """Convert finite numeric values for CSV/JSON-safe storage."""
        if value is None:
            return ""

        value = float(value)

        if not math.isfinite(value):
            return ""

        return round(value, 6)

    @staticmethod
    def _stats(values):
        """Calculate descriptive statistics for a numeric sequence."""
        clean_values = []

        for value in values:
            if value is None:
                continue

            try:
                number = float(value)
            except (TypeError, ValueError):
                continue

            if math.isfinite(number):
                clean_values.append(number)

        if not clean_values:
            return {
                "sample_count": 0,
                "mean": None,
                "median": None,
                "p95": None,
                "max": None,
            }

        return {
            "sample_count": len(clean_values),
            "mean": round(float(np.mean(clean_values)), 4),
            "median": round(float(np.median(clean_values)), 4),
            "p95": round(float(np.percentile(clean_values, 95)), 4),
            "max": round(float(np.max(clean_values)), 4),
        }

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
        """
        Record one processed frame.

        pixel_error_px:
            Distance between the detected beacon and image centre.

        localization_error_px:
            Distance between detected coordinates and the ground-truth
            coordinates supplied by main.py.

        These are different metrics and should not be interpreted as equal.
        """
        info = control_info or {}

        # Tracking error: distance from image centre.
        pixel_error = None

        if error_x is not None and error_y is not None:
            pixel_error = math.hypot(
                float(error_x),
                float(error_y),
            )

        # Localization error: distance from supplied ground truth.
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
                if target_in_frame is not None
                else ""
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
            "pan_saturated": bool(
                info.get("pan_saturated", False)
            ),
            "tilt_saturated": bool(
                info.get("tilt_saturated", False)
            ),
        }

        self.rows.append(row)

    @staticmethod
    def _valid_values(rows, field):
        """Return finite values from a field, ignoring empty CSV-style values."""
        values = []

        for row in rows:
            value = row.get(field)

            if value == "" or value is None:
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            if math.isfinite(value):
                values.append(value)

        return values

    def _phase_error_stats(self, rows):
        """Summarize image-centre tracking error separately by test phase."""
        result = {}

        phases = sorted({row["phase"] for row in rows})

        for phase in phases:
            phase_rows = [
                row for row in rows
                if row["phase"] == phase
                and row["auto_track"]
                and row["detected"]
            ]

            result[phase] = self._stats(
                self._valid_values(phase_rows, "pixel_error_px")
            )

        return result

    def save(self):
        """Save CSV and JSON reports and return the summary dictionary."""
        if self._saved:
            return self.last_summary

        # Save detailed per-frame records.
        with self.csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.fields,
            )
            writer.writeheader()
            writer.writerows(self.rows)

        total = len(self.rows)

        detection_count = sum(
            bool(row["detected"])
            for row in self.rows
        )

        # Only frames where the beacon is intended to be visible and
        # geometrically inside the field of view are eligible for recall.
        eligible_rows = [
            row for row in self.rows
            if row["target_visible"]
            and row["target_in_frame"] is True
        ]

        expected_count = len(eligible_rows)

        true_detections = sum(
            bool(row["detected"])
            for row in eligible_rows
        )

        # Detections are considered false here only when the beacon is
        # intentionally hidden or known to be outside the field of view.
        false_detections = sum(
            bool(row["detected"])
            and (
                not row["target_visible"]
                or row["target_in_frame"] is False
            )
            for row in self.rows
        )

        tracking_rows = [
            row for row in self.rows
            if row["auto_track"]
            and row["detected"]
            and row["pixel_error_px"] != ""
        ]

        tracking_errors = self._valid_values(
            tracking_rows,
            "pixel_error_px",
        )

        localization_errors = self._valid_values(
            self.rows,
            "localization_error_px",
        )

        angular_pan_errors = [
            abs(value)
            for value in self._valid_values(
                tracking_rows,
                "ground_truth_pan_error_deg",
            )
        ]

        angular_tilt_errors = [
            abs(value)
            for value in self._valid_values(
                tracking_rows,
                "ground_truth_tilt_error_deg",
            )
        ]

        processing_values = self._valid_values(
            self.rows,
            "processing_ms",
        )

        fps_values = self._valid_values(
            self.rows,
            "fps",
        )

        duration = (
            float(self.rows[-1]["elapsed_s"])
            if self.rows
            else 0.0
        )

        # Count tracking errors above the selected reporting thresholds.
        errors_over_5px = sum(
            error > 5.0 for error in tracking_errors
        )
        errors_over_10px = sum(
            error > 10.0 for error in tracking_errors
        )

        # Keep phase-specific tracking metrics separate.
        phase_tracking_stats = self._phase_error_stats(self.rows)

        summary = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(duration, 3),
            "frames_logged": total,

            "detection_count_all_frames": detection_count,
            "detection_rate_all_frames": (
                round(detection_count / total, 4)
                if total else None
            ),

            "visible_in_frame_expected_detection_frames": expected_count,
            "detections_when_target_visible_and_in_frame": true_detections,

            "detection_recall_when_target_visible_and_in_frame": (
                round(true_detections / expected_count, 4)
                if expected_count else None
            ),

            # This is the same eligible-frame ratio as recall in the
            # current single-beacon simulation, not an independent metric.
            "lock_retention_rate": (
                round(true_detections / expected_count, 4)
                if expected_count else None
            ),

            "false_detections_outside_expected_visibility": false_detections,

            "auto_track_frames": sum(
                bool(row["auto_track"])
                for row in self.rows
            ),

            # Image-centre error, not detector localization accuracy.
            "tracking_error_px": self._stats(tracking_errors),

            # Phase-specific metrics help distinguish warm-up from recovery.
            "tracking_error_px_by_phase": phase_tracking_stats,

            # Error against coordinates provided by main.py.
            # Its correctness depends on main.py providing the appropriate
            # ground truth for the rendered/disturbed beacon.
            "ground_truth_localization_error_px": self._stats(
                localization_errors
            ),

            "absolute_ground_truth_pan_error_deg": self._stats(
                angular_pan_errors
            ),

            "absolute_ground_truth_tilt_error_deg": self._stats(
                angular_tilt_errors
            ),

            "tracking_frames_over_5px": errors_over_5px,
            "tracking_frames_over_10px": errors_over_10px,

            "processing_ms": self._stats(processing_values),
            "fps": self._stats(fps_values),

            "pan_saturation_frames": sum(
                bool(row["pan_saturated"])
                for row in self.rows
            ),

            "tilt_saturation_frames": sum(
                bool(row["tilt_saturated"])
                for row in self.rows
            ),

            "csv_file": str(self.csv_path),

            "metric_notes": [
                "Pixel tracking error is the distance from the detected beacon to the image centre.",
                "Localization error compares detected coordinates with ground-truth coordinates supplied by main.py.",
                "Ground-truth localization accuracy depends on main.py supplying coordinates consistent with the rendered scene.",
                "Detection recall uses frames where the target is visible and inside the field of view.",
                "Lock retention rate currently uses the same eligible frames as detection recall.",
                "Phase-specific tracking statistics include detected frames with auto-tracking enabled.",
                "Intentional beacon-loss frames are excluded from expected detections.",
                "Processing time measures the timed frame-processing section, not full display-loop latency.",
                "These results come from a synthetic simulator and do not establish real FSOC hardware performance.",
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
