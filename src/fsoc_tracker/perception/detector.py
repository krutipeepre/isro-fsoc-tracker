
import cv2
import numpy as np


class BrightBeaconDetector:
    def __init__(self, threshold=220, min_area=5, max_area=1500):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(
            gray, self.threshold, 255, cv2.THRESH_BINARY
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((2, 2), dtype=np.uint8)
        )

        count, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(mask, connectivity=8)
        )

        candidates = []

        for i in range(1, count):
            area = int(stats[i, cv2.CC_STAT_AREA])
            width = int(stats[i, cv2.CC_STAT_WIDTH])
            height = int(stats[i, cv2.CC_STAT_HEIGHT])

            if not self.min_area <= area <= self.max_area:
                continue

            aspect = max(width, height) / max(1, min(width, height))

            if aspect <= 4:
                candidates.append((area, i))

        if not candidates:
            return None

        area, idx = max(candidates)

        x, y = centroids[idx]

        return float(x), float(y)