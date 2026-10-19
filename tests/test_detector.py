
import cv2
import numpy as np

from fsoc_tracker.perception.detector import BrightBeaconDetector


def test_detector_finds_bright_beacon():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.circle(frame, (80, 60), 5, (255, 255, 255), -1)

    result = BrightBeaconDetector().detect(frame)

    assert result is not None
    assert abs(result[0] - 80) <= 1
    assert abs(result[1] - 60) <= 1


def test_detector_returns_none_without_beacon():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    assert BrightBeaconDetector().detect(frame) is None
