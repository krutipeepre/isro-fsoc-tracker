
"""
ISRO FSOC Virtual Camera Tracking System
Milestone 3.3: Automatic Beacon Loss and Recovery Test

Run:
    python -m fsoc_tracker.main

Test sequence:
    0-3 s : Beacon visible, auto-tracking ON
    3-5 s : Beacon hidden automatically
    5+ s  : Beacon visible again; measure recovery and stability

Controls:
    Arrows  : Manual pan/tilt
    R       : Reset camera
    Space   : Pause/resume target motion
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


WARMUP_SECONDS = 3.0
LOSS_SECONDS = 2.0
STABILITY_WINDOW_SECONDS = 1.0
RECOVERY_THRESHOLD_PX = 5.0


def draw_crosshair(frame, x, y):
    cv2.line(
        frame, (x - 12, y), (x + 12, y),
        (90, 90, 90), 1, cv2.LINE_AA
    )
    cv2.line(
        frame, (x, y - 12), (x, y + 12),
        (90, 90, 90), 1, cv2.LINE_AA
    )


def draw_telemetry(frame, lines):
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
    intrinsics = CameraIntrinsics(
        width=640,
        height=480,
        hfov_deg=60.0,
    )

    camera = VirtualCamera(intrinsics=intrinsics)
    beacon = MovingBeacon()
    detector = BrightBeaconDetector()
    controller = ProportionalTracker()
    logger = PerformanceLogger()

    # Automatic test: no A/B key presses required.
    auto_track = True
    paused = False

    run_start = time.perf_counter()
    previous_time = run_start
    previous_frame_time = run_start

    fps = 0.0
    frame_count = 0

    # Test state and timestamps.
    test_phase = "WARMUP"
    loss_start_time = None
    beacon_show_time = None
    reacquisition_time = None
    recovery_time = None

    # Post-recovery stability measurements.
    stability_start_time = None
    stability_samples = 0
    stable_samples = 0
    stability_complete = False

    print("=" * 58)
    print("ISRO FSOC - Milestone 3.3")
    print("Automatic Beacon Loss and Recovery Test")
    print("=" * 58)
    print(f"Warm-up: {WARMUP_SECONDS:.1f} s")
    print(f"Beacon loss: {LOSS_SECONDS:.1f} s")
    print(f"Recovery threshold: {RECOVERY_THRESHOLD_PX:.1f} px")
    print(f"Stability window: {STABILITY_WINDOW_SECONDS:.1f} s")
    print("Auto-tracking starts ON.")
    print("Arrows: manual | R: reset | Space: pause target")
    print("Q / Esc: quit and save reports")
    print("=" * 58)

    try:
        while True:
            frame_start = time.perf_counter()
            elapsed = frame_start - run_start

            dt = min(max(frame_start - previous_time, 0.0), 0.1)
            previous_time = frame_start

            if not paused:
                beacon.advance(dt)

            # Automatic test phases.
            if test_phase == "WARMUP" and elapsed >= WARMUP_SECONDS:
                test_phase = "BEACON_LOSS"
                loss_start_time = frame_start
                print(
                    f"[{elapsed:.3f} s] Beacon hidden. "
                    "Loss test started."
                )

            elif (
                test_phase == "BEACON_LOSS"
                and loss_start_time is not None
                and frame_start - loss_start_time >= LOSS_SECONDS
            ):
                test_phase = "REACQUISITION"
                beacon_show_time = frame_start
                print(
                    f"[{elapsed:.3f} s] Beacon visible again. "
                    "Reacquisition started."
                )

            beacon_visible = test_phase != "BEACON_LOSS"

            # Render synthetic background.
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

            # Ground-truth projection is used only for rendering.
            target_point = camera.project_angles(
                beacon.azimuth_deg,
                beacon.elevation_deg,
            )

            if beacon_visible and target_point is not None:
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

            # Detect before drawing telemetry.
            detection = detector.detect(frame)

            # Force loss during the scheduled hidden interval.
            if not beacon_visible:
                detection = None

            pixel_error_x = None
            pixel_error_y = None
            angular_error_pan = None
            angular_error_tilt = None
            error_px = None

            control_info = {
                "requested_pan_deg": 0.0,
                "requested_tilt_deg": 0.0,
                "applied_pan_deg": 0.0,
                "applied_tilt_deg": 0.0,
                "pan_saturated": False,
                "tilt_saturated": False,
            }

            # Detect reacquisition.
            if (
                test_phase == "REACQUISITION"
                and detection is not None
                and reacquisition_time is None
            ):
                reacquisition_time = time.perf_counter()
                delay = reacquisition_time - beacon_show_time
                print(
                    f"Beacon reacquired after {delay:.4f} s."
                )

            if detection is not None:
                detected_x, detected_y = detection

                pixel_error_x = detected_x - intrinsics.cx
                pixel_error_y = intrinsics.cy - detected_y
                error_px = float(
                    np.hypot(pixel_error_x, pixel_error_y)
                )

                cv2.circle(
                    frame,
                    (
                        int(round(detected_x)),
                        int(round(detected_y)),
                    ),
                    11,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                # Measure error before applying this frame's correction.
                if (
                    test_phase == "REACQUISITION"
                    and recovery_time is None
                    and error_px <= RECOVERY_THRESHOLD_PX
                ):
                    recovery_time = time.perf_counter()
                    delay = recovery_time - beacon_show_time
                    stability_start_time = recovery_time

                    print(
                        f"Error reached <= {RECOVERY_THRESHOLD_PX:.1f} px "
                        f"after {delay:.4f} s from beacon return."
                    )

                if auto_track:
                    control_info = controller.command(
                        camera,
                        detected_x,
                        detected_y,
                    )

                    angular_error_pan = control_info["pan_error_deg"]
                    angular_error_tilt = control_info["tilt_error_deg"]

            # Measure the one-second window after first reaching threshold.
            if (
                stability_start_time is not None
                and not stability_complete
                and test_phase == "REACQUISITION"
                and time.perf_counter() - stability_start_time
                <= STABILITY_WINDOW_SECONDS
                and detection is not None
            ):
                stability_samples += 1
                if error_px <= RECOVERY_THRESHOLD_PX:
                    stable_samples += 1

            if (
                stability_start_time is not None
                and not stability_complete
                and time.perf_counter() - stability_start_time
                > STABILITY_WINDOW_SECONDS
            ):
                stability_complete = True
                if stability_samples:
                    ratio = stable_samples / stability_samples
                    print(
                        "Post-recovery stability: "
                        f"{stable_samples}/{stability_samples} "
                        "valid detection samples were evaluated."
                    )
                    print(
                        f"Frames within threshold: "
                        f"{100.0 * ratio:.1f}%."
                    )
                else:
                    print("Post-recovery stability: no valid samples.")

            # FPS and processing time.
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

            if not beacon_visible:
                status = "BEACON HIDDEN"
            elif detection is None:
                status = "NO DETECTION"
            elif test_phase == "REACQUISITION":
                status = "REACQUISITION"
            else:
                status = "TRACKING"

            pixel_line = (
                f"Pixel error: {error_px:.2f} px"
                if error_px is not None
                else "Pixel error: unavailable"
            )

            angular_line = (
                f"Angular error: "
                f"{angular_error_pan:+.2f} / "
                f"{angular_error_tilt:+.2f} deg"
                if angular_error_pan is not None
                else "Angular error: unavailable"
            )

            phase_elapsed = (
                time.perf_counter() - beacon_show_time
                if beacon_show_time is not None
                else 0.0
            )

            telemetry = [
                "ISRO FSOC - Automatic Recovery Test",
                f"Phase: {test_phase} | Status: {status}",
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
                f"Auto-track: {auto_track} | {pixel_line}",
                angular_line,
                (
                    f"FOV: {intrinsics.hfov_deg:.1f} / "
                    f"{intrinsics.vfov_deg:.1f} deg | FPS: {fps:.1f}"
                ),
                f"Processing: {processing_ms:.2f} ms",
                f"Elapsed since beacon return: {phase_elapsed:.2f} s",
                "Arrows: manual | R: reset | Space: pause target",
                "Q/Esc: quit and save",
            ]

            draw_telemetry(frame, telemetry)

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
                control_info=control_info,
            )

            cv2.imshow("ISRO FSOC - Automatic Recovery Test", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

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

            elif key == 81:  # Left arrow
                camera.move(delta_pan_deg=-1.0)

            elif key == 83:  # Right arrow
                camera.move(delta_pan_deg=1.0)

            elif key == 82:  # Up arrow
                camera.move(delta_tilt_deg=1.0)

            elif key == 84:  # Down arrow
                camera.move(delta_tilt_deg=-1.0)

    finally:
        cv2.destroyAllWindows()
        logger.save()

        print("\n" + "=" * 58)
        print("AUTOMATIC TEST SUMMARY")
        print("=" * 58)
        print(f"Frames processed: {frame_count}")

        if beacon_show_time is not None:
            if reacquisition_time is not None:
                print(
                    "Reacquisition delay: "
                    f"{reacquisition_time - beacon_show_time:.4f} s"
                )
            else:
                print("Reacquisition delay: not measured.")

            if recovery_time is not None:
                print(
                    "Recovery delay (<=5 px): "
                    f"{recovery_time - beacon_show_time:.4f} s"
                )
            else:
                print("Recovery threshold was not reached.")

        if stability_samples:
            print(
                "Post-recovery samples within 5 px: "
                f"{stable_samples}/{stability_samples} "
                f"({100.0 * stable_samples / stability_samples:.1f}%)"
            )
        else:
            print("Post-recovery stability samples: none.")

        print("Reports saved. FSOC simulator closed.")


if __name__ == "__main__":
    main()
