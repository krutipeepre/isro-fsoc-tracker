"""Interactive Milestone 1 demo. Run with: python -m fsoc_tracker.main"""
import time
import cv2
import numpy as np

from .geometry import CameraIntrinsics, VirtualCamera
from .simulation import MovingBeacon


def draw_crosshair(frame, x, y, color=(90, 90, 90)):
    cv2.line(frame, (x - 12, y), (x + 12, y), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x, y - 12), (x, y + 12), color, 1, cv2.LINE_AA)


def main():
    intrinsics = CameraIntrinsics(width=640, height=480, hfov_deg=60.0)
    camera = VirtualCamera(intrinsics=intrinsics)
    beacon = MovingBeacon()
    paused = False
    dt = 1.0 / 30.0
    last_time = time.perf_counter()

    print("FSOC Milestone 1 simulator started.")
    print("Arrow keys: pan/tilt | R: reset | Space: pause | Q/Esc: quit")

    while True:
        now = time.perf_counter()
        elapsed = now - last_time
        last_time = now

        if not paused:
            # Use measured elapsed time, bounded to avoid jumps after window stalls.
            beacon.advance(min(max(elapsed, 0.0), 0.1))

        frame = np.zeros((intrinsics.height, intrinsics.width, 3), dtype=np.uint8)
        frame[:] = (18, 22, 28)

        # Subtle grid for visual reference.
        for x in range(0, intrinsics.width, 80):
            cv2.line(frame, (x, 0), (x, intrinsics.height), (31, 36, 43), 1)
        for y in range(0, intrinsics.height, 80):
            cv2.line(frame, (0, y), (intrinsics.width, y), (31, 36, 43), 1)

        cx, cy = int(round(intrinsics.cx)), int(round(intrinsics.cy))
        draw_crosshair(frame, cx, cy)

        point = camera.project_angles(beacon.azimuth_deg, beacon.elevation_deg)
        if point is not None:
            px, py = int(round(point[0])), int(round(point[1]))
            # Glow + core represents a synthetic optical beacon.
            cv2.circle(frame, (px, py), 15, (35, 80, 130), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 8, (80, 190, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 3, (230, 250, 255), -1, cv2.LINE_AA)
            status = "VISIBLE"
        else:
            status = "OUTSIDE FOV"

        lines = [
            "ISRO FSOC - Virtual Camera Geometry (Milestone 1)",
            f"Target az/el: {beacon.azimuth_deg:+.2f} / {beacon.elevation_deg:+.2f} deg",
            f"Camera pan/tilt: {camera.pan_deg:+.1f} / {camera.tilt_deg:+.1f} deg",
            f"Target status: {status}",
            f"HFOV/VFOV: {intrinsics.hfov_deg:.1f} / {intrinsics.vfov_deg:.1f} deg",
            "Arrows: steer | R: reset | Space: pause | Q/Esc: quit",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                frame, line, (14, 24 + i * 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (225, 230, 235), 1, cv2.LINE_AA
            )

        cv2.imshow("FSOC Virtual Camera - Milestone 1", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            camera.reset()
        elif key == ord(" "):
            paused = not paused
        elif key == 81:  # left arrow on many OpenCV Windows builds
            camera.move(delta_pan_deg=-1.0)
        elif key == 83:  # right arrow
            camera.move(delta_pan_deg=1.0)
        elif key == 82:  # up arrow
            camera.move(delta_tilt_deg=1.0)
        elif key == 84:  # down arrow
            camera.move(delta_tilt_deg=-1.0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
