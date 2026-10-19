
"""
ISRO FSOC Virtual Camera Tracker
Milestone 3: Tracking, Performance Logging & Diagnostics

Run:
    python -m fsoc_tracker.main

Controls:
    A       : Toggle automatic tracking
    Arrows  : Manual camera movement
    R       : Reset camera
    Space   : Pause/resume target
    Q / Esc : Quit and save reports
"""

import time

import cv2
import numpy as np

from .geometry import CameraIntrinsics, VirtualCamera
from .simulation import MovingBeacon
from .perception.detector import BrightBeaconDetector
from .control.controller import ProportionalTracker
from .evaluation.logger import PerformanceLogger


def draw_crosshair(frame, x, y):
    """Draw the optical-axis center."""
    cv2.line(
        frame, (x - 12, y), (x + 12, y),
        (90, 90, 90), 1, cv2.LINE_AA
    )
    cv2.line(
        frame, (x, y - 12), (x, y + 12),
        (90, 90, 90), 1, cv2.LINE_AA
    )


def draw_telemetry(frame, lines):
    """Draw telemetry after beacon detection."""
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (14, 22 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (225, 230, 235),
            1,
            cv2.LINE_AA,
        )


def main():
    # 1. Simulation setup.
    intrinsics = CameraIntrinsics(
        width=640,
        height=480,
        hfov_deg=60.0,
    )

    camera = VirtualCamera(intrinsics=intrinsics)
    beacon = MovingBeacon()

    # 2. Perception, control and performance logger.
    detector = BrightBeaconDetector()
    controller = ProportionalTracker()
    logger = PerformanceLogger()

    auto_track = False
    paused = False

    run_start = time.perf_counter()
    previous_time = run_start
    previous_frame_time = run_start

    fps = 0.0
    frame_count = 0

    print("=" * 58)
    print("ISRO FSOC - Tracking + Controller Diagnostics")
    print("=" * 58)
    print("A       : Toggle automatic tracking")
    print("Arrows  : Manual pan/tilt")
    print("R       : Reset camera")
    print("Space   : Pause/resume target")
    print("Q / Esc : Quit and save reports")
    print("=" * 58)

    try:
        while True:
            frame_start = time.perf_counter()
            now = frame_start

            dt = min(max(now - previous_time, 0.0), 0.1)
            previous_time = now

            if not paused:
                beacon.advance(dt)

            # 3. Render synthetic scene.
            frame = np.zeros(
                (intrinsics.height, intrinsics.width, 3),
                dtype=np.uint8,
            )
            frame[:] = (18, 22, 28)

            for x in range(0, intrinsics.width, 80):
                cv2.line(
                    frame, (x, 0), (x, intrinsics.height),
                    (31, 36, 43), 1
                )

            for y in range(0, intrinsics.height, 80):
                cv2.line(
                    frame, (0, y), (intrinsics.width, y),
                    (31, 36, 43), 1
                )

            center_x = int(round(intrinsics.cx))
            center_y = int(round(intrinsics.cy))
            draw_crosshair(frame, center_x, center_y)

            # 4. Ground-truth target projection for simulation.
            target_point = camera.project_angles(
                beacon.azimuth_deg,
                beacon.elevation_deg,
            )

            if target_point is not None:
                target_x = int(round(target_point[0]))
                target_y = int(round(target_point[1]))

                cv2.circle(
                    frame, (target_x, target_y),
                    15, (35, 80, 130), -1, cv2.LINE_AA
                )
                cv2.circle(
                    frame, (target_x, target_y),
                    8, (80, 190, 255), -1, cv2.LINE_AA
                )
                cv2.circle(
                    frame, (target_x, target_y),
                    3, (230, 250, 255), -1, cv2.LINE_AA
                )

            # 5. Detect beacon before drawing text.
            detection = detector.detect(frame)

            pixel_error_x = None
            pixel_error_y = None
            angular_error_pan = None
            angular_error_tilt = None

            control_info = {
                "requested_pan_deg": 0.0,
                "requested_tilt_deg": 0.0,
                "applied_pan_deg": 0.0,
                "applied_tilt_deg": 0.0,
                "pan_saturated": False,
                "tilt_saturated": False,
            }

            if detection is not None:
                detected_x, detected_y = detection

                pixel_error_x = detected_x - intrinsics.cx
                pixel_error_y = intrinsics.cy - detected_y

                # Green marker indicates detected beacon center.
                cv2.circle(
                    frame,
                    (int(round(detected_x)), int(round(detected_y))),
                    11,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                if auto_track:
                    # UPDATED CONTROLLER INTEGRATION:
                    # command() now returns a diagnostic dictionary.
                    control_info = controller.command(
                        camera,
                        detected_x,
                        detected_y,
                    )

                    angular_error_pan = control_info["pan_error_deg"]
                    angular_error_tilt = control_info["tilt_error_deg"]

            # 6. Frame processing time and FPS.
            frame_end = time.perf_counter()
            processing_ms = (frame_end - frame_start) * 1000.0

            frame_elapsed = frame_end - previous_frame_time
            previous_frame_time = frame_end

            if frame_elapsed > 0:
                instant_fps = 1.0 / frame_elapsed
                fps = (
                    0.9 * fps + 0.1 * instant_fps
                    if fps > 0
                    else instant_fps
                )

            frame_count += 1

            # 7. Status.
            if detection is None:
                status = "NO DETECTION"
            elif target_point is None:
                status = "TARGET OUTSIDE FOV"
            elif auto_track:
                status = "AUTO-TRACK ACTIVE"
            else:
                status = "BEACON DETECTED"

            mode = "ON" if auto_track else "OFF"

            if pixel_error_x is not None:
                pixel_line = (
                    f"Pixel error dx/dy: "
                    f"{pixel_error_x:+.1f} / "
                    f"{pixel_error_y:+.1f}"
                )
            else:
                pixel_line = "Pixel error: unavailable"

            if angular_error_pan is not None:
                angular_line = (
                    f"Angular error pan/tilt: "
                    f"{angular_error_pan:+.2f} / "
                    f"{angular_error_tilt:+.2f} deg"
                )
            else:
                angular_line = "Angular error: unavailable"

            saturation = (
                f"Pan saturated: {control_info['pan_saturated']} | "
                f"Tilt saturated: {control_info['tilt_saturated']}"
            )

            telemetry = [
                "ISRO FSOC - Tracking + Diagnostics",
                (
                    f"Target az/el: "
                    f"{beacon.azimuth_deg:+.2f} / "
                    f"{beacon.elevation_deg:+.2f} deg"
                ),
                (
                    f"Camera pan/tilt: "
                    f"{camera.pan_deg:+.2f} / "
                    f"{camera.tilt_deg:+.2f} deg"
                ),
                f"Status: {status} | Auto-track: {mode}",
                pixel_line,
                angular_line,
                saturation,
                (
                    f"FOV: {intrinsics.hfov_deg:.1f} / "
                    f"{intrinsics.vfov_deg:.1f} deg | FPS: {fps:.1f}"
                ),
                f"Processing: {processing_ms:.2f} ms | Paused: {paused}",
                "A: auto-track | Arrows: manual | R: reset",
                "Space: pause | Q/Esc: quit and save reports",
            ]

            draw_telemetry(frame, telemetry)

            # 8. Log the frame.
            # Existing logger schema is preserved for compatibility.
            logger.record(
                elapsed_s=time.perf_counter() - run_start,
                detected=detection is not None,
                auto_track=auto_track,
                error_x=pixel_error_x,
                error_y=pixel_error_y,
                pan_deg=camera.pan_deg,
                tilt_deg=camera.tilt_deg,
                processing_ms=processing_ms,
                fps=fps,
            )

            cv2.imshow("ISRO FSOC - Milestone 3", frame)

            # 9. Keyboard controls.
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key in (ord("a"), ord("A")):
                auto_track = not auto_track
                print(
                    "Automatic tracking:",
                    "ON" if auto_track else "OFF",
                )

            elif key in (ord("r"), ord("R")):
                camera.reset()
                print("Camera orientation reset.")

            elif key == ord(" "):
                paused = not paused
                print(
                    "Simulation paused."
                    if paused
                    else "Simulation resumed."
                )

            elif key == 81:  # Left
                camera.move(delta_pan_deg=-1.0)

            elif key == 83:  # Right
                camera.move(delta_pan_deg=1.0)

            elif key == 82:  # Up
                camera.move(delta_tilt_deg=1.0)

            elif key == 84:  # Down
                camera.move(delta_tilt_deg=-1.0)

    finally:
        cv2.destroyAllWindows()
        logger.save()
        print(f"Frames processed: {frame_count}")
        print("FSOC simulator closed. Reports saved.")


if __name__ == "__main__":
    main()
