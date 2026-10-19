
"""
ISRO FSOC Virtual Camera Tracker
Milestone 2: Beacon Detection + Automatic Camera Tracking

Run:
    python -m fsoc_tracker.main

Controls:
    A       : Toggle automatic tracking
    Arrows  : Manual camera movement
    R       : Reset camera
    Space   : Pause/resume target
    Q / Esc : Quit
"""

import time

import cv2
import numpy as np

from .geometry import CameraIntrinsics, VirtualCamera
from .simulation import MovingBeacon
from .perception.detector import BrightBeaconDetector
from .control.controller import ProportionalTracker


def draw_crosshair(frame, x, y, color=(90, 90, 90)):
    """Draw the camera's optical-axis center."""
    cv2.line(
        frame, (x - 12, y), (x + 12, y),
        color, 1, cv2.LINE_AA
    )
    cv2.line(
        frame, (x, y - 12), (x, y + 12),
        color, 1, cv2.LINE_AA
    )


def draw_text(frame, lines):
    """Render application telemetry."""
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
    # 1. Initialize the virtual camera and target.
    intrinsics = CameraIntrinsics(
        width=640,
        height=480,
        hfov_deg=60.0,
    )

    camera = VirtualCamera(intrinsics=intrinsics)
    beacon = MovingBeacon()

    # 2. Initialize perception and control.
    detector = BrightBeaconDetector()
    controller = ProportionalTracker()

    auto_track = False
    paused = False

    previous_time = time.perf_counter()
    previous_frame_time = previous_time
    fps = 0.0

    print("=" * 55)
    print("ISRO FSOC - Virtual Camera Tracking System")
    print("Milestone 2: Automatic Beacon Tracking")
    print("=" * 55)
    print("A       : Toggle automatic tracking")
    print("Arrows  : Manual pan/tilt")
    print("R       : Reset camera")
    print("Space   : Pause/resume target")
    print("Q / Esc : Quit")
    print("=" * 55)

    try:
        while True:
            now = time.perf_counter()

            # Bound dt so a window stall does not cause a large simulation jump.
            dt = min(max(now - previous_time, 0.0), 0.1)
            previous_time = now

            if not paused:
                beacon.advance(dt)

            # 3. Render a synthetic camera frame.
            frame = np.zeros(
                (intrinsics.height, intrinsics.width, 3),
                dtype=np.uint8,
            )
            frame[:] = (18, 22, 28)

            # Background grid.
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

            # 4. Project the beacon into the camera image.
            target_point = camera.project_angles(
                beacon.azimuth_deg,
                beacon.elevation_deg,
            )

            if target_point is not None:
                target_x = int(round(target_point[0]))
                target_y = int(round(target_point[1]))

                # Synthetic optical beacon with a bright central core.
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

            # 5. Detect the beacon BEFORE drawing telemetry text.
            # Otherwise bright text could become a false detection.
            detection = detector.detect(frame)

            pixel_error_x = None
            pixel_error_y = None
            angular_error_pan = None
            angular_error_tilt = None

            if detection is not None:
                detected_x, detected_y = detection

                pixel_error_x = detected_x - intrinsics.cx
                pixel_error_y = intrinsics.cy - detected_y

                # Show the detector's estimated position in green.
                cv2.circle(
                    frame,
                    (int(round(detected_x)), int(round(detected_y))),
                    11,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.circle(
                    frame,
                    (int(round(detected_x)), int(round(detected_y))),
                    2,
                    (0, 255, 0),
                    -1,
                )

                # 6. Closed-loop camera control.
                if auto_track:
                    angular_error_pan, angular_error_tilt = (
                        controller.command(
                            camera,
                            detected_x,
                            detected_y,
                        )
                    )

            # 7. Calculate frame rate for display.
            frame_elapsed = now - previous_frame_time
            previous_frame_time = now

            if frame_elapsed > 0:
                instantaneous_fps = 1.0 / frame_elapsed
                fps = (
                    0.9 * fps + 0.1 * instantaneous_fps
                    if fps > 0
                    else instantaneous_fps
                )

            # 8. Status and telemetry.
            if detection is None:
                tracking_status = "NO DETECTION"
            elif target_point is None:
                tracking_status = "TARGET OUTSIDE FOV"
            elif auto_track:
                tracking_status = "AUTO-TRACK ACTIVE"
            else:
                tracking_status = "BEACON DETECTED"

            mode = "ON" if auto_track else "OFF"

            if pixel_error_x is not None:
                error_line = (
                    f"Pixel error dx/dy: "
                    f"{pixel_error_x:+.1f} / "
                    f"{pixel_error_y:+.1f}"
                )
            else:
                error_line = "Pixel error: unavailable"

            if angular_error_pan is not None:
                angle_line = (
                    f"Angular error pan/tilt: "
                    f"{angular_error_pan:+.2f} / "
                    f"{angular_error_tilt:+.2f} deg"
                )
            else:
                angle_line = "Angular error: unavailable"

            telemetry = [
                "ISRO FSOC - Automatic Virtual Camera Tracker",
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
                f"Status: {tracking_status} | Auto-track: {mode}",
                error_line,
                angle_line,
                (
                    f"HFOV/VFOV: {intrinsics.hfov_deg:.1f} / "
                    f"{intrinsics.vfov_deg:.1f} deg | "
                    f"FPS: {fps:.1f}"
                ),
                "A: auto-track | Arrows: manual | R: reset",
                "Space: pause | Q/Esc: quit",
            ]

            draw_text(frame, telemetry)

            cv2.imshow(
                "ISRO FSOC - Milestone 2",
                frame,
            )

            # 9. Keyboard input.
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
                print("Simulation paused." if paused else "Simulation resumed.")

            elif key == 81:  # Left arrow on common OpenCV Windows builds
                camera.move(delta_pan_deg=-1.0)

            elif key == 83:  # Right arrow
                camera.move(delta_pan_deg=1.0)

            elif key == 82:  # Up arrow
                camera.move(delta_tilt_deg=1.0)

            elif key == 84:  # Down arrow
                camera.move(delta_tilt_deg=-1.0)

    finally:
        cv2.destroyAllWindows()
        print("FSOC simulator closed.")


if __name__ == "__main__":
    main()
