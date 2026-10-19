import math
from fsoc_tracker.geometry import CameraIntrinsics, VirtualCamera


def test_optical_axis_projects_to_image_center():
    intr = CameraIntrinsics(width=640, height=480, hfov_deg=60)
    cam = VirtualCamera(intrinsics=intr)
    point = cam.project_angles(0.0, 0.0)
    assert point is not None
    assert abs(point[0] - intr.cx) < 1e-9
    assert abs(point[1] - intr.cy) < 1e-9


def test_positive_azimuth_projects_right():
    intr = CameraIntrinsics()
    cam = VirtualCamera(intrinsics=intr)
    point = cam.project_angles(5.0, 0.0)
    assert point is not None
    assert point[0] > intr.cx


def test_camera_pan_shifts_stationary_target_left():
    intr = CameraIntrinsics()
    cam = VirtualCamera(intrinsics=intr)
    before = cam.project_angles(5.0, 0.0)
    cam.move(delta_pan_deg=2.0)
    after = cam.project_angles(5.0, 0.0)
    assert before is not None and after is not None
    assert after[0] < before[0]


def test_target_outside_fov_returns_none():
    intr = CameraIntrinsics(width=640, height=480, hfov_deg=60)
    cam = VirtualCamera(intrinsics=intr)
    assert cam.project_angles(40.0, 0.0) is None


def test_camera_motion_is_clamped():
    cam = VirtualCamera(intrinsics=CameraIntrinsics())
    cam.move(delta_pan_deg=1000, delta_tilt_deg=-1000)
    assert cam.pan_deg == cam.max_pan_deg
    assert cam.tilt_deg == -cam.max_tilt_deg


def test_vertical_fov_matches_aspect_ratio():
    intr = CameraIntrinsics(width=640, height=480, hfov_deg=60)
    assert math.isclose(intr.vfov_deg, 46.826, abs_tol=0.02)
