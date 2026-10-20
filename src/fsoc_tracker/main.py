
"""
ISRO FSOC Virtual Camera Tracking System
Milestone 3.5: Configurable Disturbance Simulation

Run from project root:
    python -m fsoc_tracker.main

Automatic test sequence:
    Warm-up       : Beacon visible, auto-tracking ON
    Beacon loss   : Beacon hidden automatically
    Reacquisition : Beacon becomes visible again
    Stability     : Evaluate every frame for 1 second after recovery

Controls:
    Arrows  : Manual pan/tilt
    R       : Reset camera orientation
    Space   : Pause/resume target motion
    D       : Toggle simulated disturbances
    Q / Esc : Quit and save reports

Note:
    Disturbances are software-generated approximations. They are not
    a validated physical atmospheric or hardware model.
"""

import time

import cv2
import numpy as np

from .geometry import CameraIntrinsics, VirtualCamera
from .simulation import (
    MovingBeacon,
    DisturbanceConfig,
    DisturbanceModel,
)
from .perception.detector import BrightBeaconDetector
from .control.controller import ProportionalTracker
from .evaluation.logger import PerformanceLogger


WARMUP_SECONDS = 3.0
LOSS_SECONDS = 2.0
STABILITY_WINDOW_SECONDS = 1.0
RECOVERY_THRESHOLD_PX = 5.0

# Adjustable software-simulation disturbance settings.
# Set all strengths to 0.0 for a clean baseline.
DISTURBANCE_CONFIG = DisturbanceConfig(
    atmospheric_jitter_deg=0.03,
    vibration_amplitude_deg=0.05,
    vibration_frequency_hz=8.0,
    camera_drift_rate_deg_s=0.005,
    random_seed=42,
)


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
    disturbance = DisturbanceModel(DISTURBANCE_CONFIG)
    detector = BrightBeaconDetector()
    controller = ProportionalTracker()
    logger = PerformanceLogger()

    auto_track = True
    paused = False
    disturbances_enabled = True

    run_start = time.perf_counter()
    previous_time = run_start
    previous_frame_time = run_start

    fps = 0.0
    frame_count = 0

    test_phase = "WARMUP"
    loss_start_time = None
    beacon_show_time = None
    reacquisition_time = None
    recovery_time = None

    stability_start_time = None
    stability_samples = 0
    stable_samples = 0
    stability_complete = False

    # Internal camera offsets model vibration and accumulated drift.
    # The camera object's pan/tilt remains the controller's commanded pose.
    vibration_pan = 0.0
    vibration_tilt = 0.0
    drift_pan = 0.0
    drift_tilt = 0.0

    print("=" * 60)
    print("ISRO FSOC - Milestone 3.5")
    print("Configurable Disturbance Simulation")
    print("=" * 60)
    print(f"Warm-up: {WARMUP_SECONDS:.1f} s")
    print(f"Beacon loss: {LOSS_SECONDS:.1f} s")
    print(f"Recovery threshold: {RECOVERY_THRESHOLD_PX:.1f} px")
    print(f"Stability window: {STABILITY_WINDOW_SECONDS:.1f} s")
    print(f"Atmospheric jitter sigma: {DISTURBANCE_CONFIG.atmospheric_jitter_deg:.3f} deg")
    print(f"Vibration amplitude: {DISTURBANCE_CONFIG.vibration_amplitude_deg:.3f} deg")
    print(f"Vibration frequency: {DISTURBANCE_CONFIG.vibration_frequency_hz:.2f} Hz")
    print(f"Camera drift rate: {DISTURBANCE_CONFIG.camera_drift_rate_deg_s:.4f} deg/s")
    print("Disturbances start ON. Press D to toggle.")
    print("Arrows: manual | R: reset | Space: pause target")
    print("Q / Esc: quit and save reports")
    print("=" * 60)

    try:
        while True:
            frame_start = time.perf_counter()
            elapsed = frame_start - run_start

            dt = min(max(frame_start - previous_time, 0.0), 0.1)
            previous_time = frame_start

            if not paused:
                beacon.advance(dt)

            # Keep disturbance time running even when target motion is paused.
            disturbance.advance(dt)

            if test_phase == "WARMUP" and elapsed >= WARMUP_SECONDS:
                test_phase = "BEACON_LOSS"
                loss_start_time = frame_start
                print(f"[{elapsed:.3f} s] Beacon hidden. Loss test started.")

            elif (
                test_phase == "BEACON_LOSS"
                and loss_start_time is not None
                and frame_start - loss_start_time >= LOSS_SECONDS
            ):
                test_phase = "REACQUISITION"
                beacon_show_time = frame_start
                print(f"[{elapsed:.3f} s] Beacon visible again. Reacquisition started.")

            beacon_visible = test_phase != "BEACON_LOSS"

            # Update disturbance components.
            if disturbances_enabled:
                vibration_pan, vibration_tilt = disturbance.camera_vibration()
                drift_pan, drift_tilt = disturbance.camera_drift()
                apparent_azimuth, apparent_elevation = (
                    disturbance.apparent_target_angles(
                        beacon.azimuth_deg,
                        beacon.elevation_deg,
                    )
                )
            else:
                vibration_pan = vibration_tilt = 0.0
                drift_pan = drift_tilt = 0.0
                apparent_azimuth = beacon.azimuth_deg
                apparent_elevation = beacon.elevation_deg

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

            # Effective pointing includes vibration and drift.
            effective_pan = camera.pan_deg + vibration_pan + drift_pan
            effective_tilt = camera.tilt_deg + vibration_tilt + drift_tilt

            # Project using a temporary camera pose so the commanded camera
            # pose remains separate from simulated disturbances.
            effective_camera = VirtualCamera(
                intrinsics=intrinsics,
                pan_deg=effective_pan,
                tilt_deg=effective_tilt,
                max_pan_deg=camera.max_pan_deg,
                max_tilt_deg=camera.max_tilt_deg,
            )

            camera_pan_before = effective_pan
            camera_tilt_before = effective_tilt

            # Ground truth is based on the undisturbed world trajectory.
            true_target_point = camera.project_angles(
                beacon.azimuth_deg,
                beacon.elevation_deg,
            )

            # The rendered target includes apparent atmospheric jitter and
            # the effective disturbed camera pointing.
            target_point = effective_camera.project_angles(
                apparent_azimuth,
                apparent_elevation,
            )

            target_in_frame = true_target_point is not None
            ground_truth_x = None
            ground_truth_y = None

            if true_target_point is not None:
                ground_truth_x, ground_truth_y = true_target_point

            ground_truth_pan_error_deg = (
                beacon.azimuth_deg - camera_pan_before
            )
            ground_truth_tilt_error_deg = (
                beacon.elevation_deg - camera_tilt_before
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

            detection = detector.detect(frame)

            # The hidden beacon is intentionally unavailable to the detector.
            if not beacon_visible:
                detection = None

            detected_x = None
            detected_y = None
            pixel_error_x = None
            pixel_error_y = None
            error_px = None
            angular_error_pan = None
            angular_error_tilt = None

            control_info = {
                "pan_error_deg": 0.0,
                "tilt_error_deg": 0.0,
                "requested_pan_deg": 0.0,
                "requested_tilt_deg": 0.0,
                "applied_pan_deg": 0.0,
                "applied_tilt_deg": 0.0,
                "pan_saturated": False,
                "tilt_saturated": False,
            }

            if (
                test_phase == "REACQUISITION"
                and detection is not None
                and reacquisition_time is None
            ):
                reacquisition_time = time.perf_counter()
                print(
                    "Beacon reacquired after "
                    f"{reacquisition_time - beacon_show_time:.4f} s."
                )

            if detection is not None:
                detected_x, detected_y = detection

                pixel_error_x = detected_x - intrinsics.cx
                pixel_error_y = intrinsics.cy - detected_y
                error_px = float(np.hypot(pixel_error_x, pixel_error_y))

                cv2.circle(
                    frame,
                    (int(round(detected_x)), int(round(detected_y))),
                    11,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                if (
                    test_phase == "REACQUISITION"
                    and recovery_time is None
                    and error_px <= RECOVERY_THRESHOLD_PX
                ):
                    recovery_time = time.perf_counter()
                    stability_start_time = recovery_time
                    print(
                        f"Error reached <= {RECOVERY_THRESHOLD_PX:.1f} px after "
                        f"{recovery_time - beacon_show_time:.4f} s."
                    )

                if auto_track:
                    # Controller changes the commanded pose. Disturbances
                    # remain independent offsets applied during projection.
                    control_info = controller.command(
                        camera,
                        detected_x,
                        detected_y,
                    )
                    angular_error_pan = control_info["pan_error_deg"]
                    angular_error_tilt = control_info["tilt_error_deg"]

            # Stability denominator includes every processed frame in window.
            if (
                stability_start_time is not None
                and not stability_complete
                and test_phase == "REACQUISITION"
            ):
                now = time.perf_counter()
                window_elapsed = now - stability_start_time

                if window_elapsed < STABILITY_WINDOW_SECONDS:
                    stability_samples += 1
                    if (
                        detection is not None
                        and error_px is not None
                        and error_px <= RECOVERY_THRESHOLD_PX
                    ):
                        stable_samples += 1
                else:
                    stability_complete = True
                    stability_ratio = (
                        stable_samples / stability_samples
                        if stability_samples > 0 else 0.0
                    )
                    print(
                        "Post-recovery stability: "
                        f"{stable_samples}/{stability_samples} frames "
                        f"within threshold ({100.0 * stability_ratio:.1f}%)."
                    )
                    if stability_samples == 0:
                        print("Warning: no frames were sampled.")

            frame_end = time.perf_counter()
            processing_ms = (frame_end - frame_start) * 1000.0

            frame_elapsed = frame_end - previous_frame_time
            previous_frame_time = frame_end
            if frame_elapsed > 0:
                instant_fps = 1.0 / frame_elapsed
                fps = 0.9 * fps + 0.1 * instant_fps if fps > 0 else instant_fps

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
                if error_px is not None else "Pixel error: unavailable"
            )
            angular_line = (
                f"Controller error: {angular_error_pan:+.2f} / "
                f"{angular_error_tilt:+.2f} deg"
                if angular_error_pan is not None
                else "Controller error: unavailable"
            )

            phase_elapsed = (
                time.perf_counter() - beacon_show_time
                if beacon_show_time is not None else 0.0
            )

            telemetry = [
                "ISRO FSOC - Disturbance Simulation",
                f"Phase: {test_phase} | Status: {status}",
                (
                    f"Target az/el: {beacon.azimuth_deg:+.2f} / "
                    f"{beacon.elevation_deg:+.2f} deg"
                ),
                (
                    f"Camera pan/tilt: {camera.pan_deg:+.2f} / "
                    f"{camera.tilt_deg:+.2f} deg"
                ),
                f"Disturbances: {'ON' if disturbances_enabled else 'OFF'} | Auto-track: {auto_track}",
                pixel_line,
                angular_line,
                (
                    f"FOV H/V: {intrinsics.hfov_deg:.1f} / "
                    f"{intrinsics.vfov_deg:.1f} deg | FPS: {fps:.1f}"
                ),
                f"Processing: {processing_ms:.2f} ms",
                f"Since beacon return: {phase_elapsed:.2f} s",
                "D: toggle disturbances | Arrows: manual | R: reset",
                "Space: pause target | Q/Esc: quit and save",
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
                phase=test_phase,
                target_visible=beacon_visible,
                target_in_frame=target_in_frame,
                detected_x=detected_x,
                detected_y=detected_y,
                ground_truth_x=ground_truth_x,
                ground_truth_y=ground_truth_y,
                ground_truth_pan_error_deg=ground_truth_pan_error_deg,
                ground_truth_tilt_error_deg=ground_truth_tilt_error_deg,
            )

            cv2.imshow("ISRO FSOC - Disturbance Simulation", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key in (ord("r"), ord("R")):
                camera.reset()
                print("Camera commanded orientation reset.")
            elif key == ord(" "):
                paused = not paused
                print("Simulation paused." if paused else "Simulation resumed.")
            elif key in (ord("d"), ord("D")):
                disturbances_enabled = not disturbances_enabled
                print(
                    "Disturbances enabled."
                    if disturbances_enabled else "Disturbances disabled."
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

        print("\n" + "=" * 60)
        print("MILESTONE 3.5 TEST SUMMARY")
        print("=" * 60)
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

        if stability_complete and stability_samples > 0:
            final_stability_ratio = stable_samples / stability_samples
            print(
                "Post-recovery stability: "
                f"{stable_samples}/{stability_samples} frames "
                f"({100.0 * final_stability_ratio:.1f}%)."
            )
        elif stability_samples > 0:
            partial_ratio = stable_samples / stability_samples
            print(
                "Stability window incomplete: "
                f"{stable_samples}/{stability_samples} frames passed "
                f"({100.0 * partial_ratio:.1f}%) before exit."
            )
        else:
            print("Post-recovery stability samples: none.")

        print("Reports saved. FSOC simulator closed.")


if __name__ == "__main__":
    main()
