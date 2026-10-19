
from dataclasses import dataclass
import math


@dataclass
class ProportionalTracker:
    gain: float = 0.65
    max_step_deg: float = 2.0
    deadband_px: float = 4.0

    def command(self, camera, u: float, v: float):
        intr = camera.intrinsics

        dx = u - intr.cx
        dy = intr.cy - v

        if abs(dx) <= self.deadband_px:
            dx = 0.0
        if abs(dy) <= self.deadband_px:
            dy = 0.0

        pan_error_deg = math.degrees(
            math.atan(dx / intr.fx)
        )

        fy = (intr.height / 2) / math.tan(
            math.radians(intr.vfov_deg) / 2
        )

        tilt_error_deg = math.degrees(
            math.atan(dy / fy)
        )

        requested_pan = self.gain * pan_error_deg
        requested_tilt = self.gain * tilt_error_deg

        applied_pan = max(
            -self.max_step_deg,
            min(self.max_step_deg, requested_pan)
        )

        applied_tilt = max(
            -self.max_step_deg,
            min(self.max_step_deg, requested_tilt)
        )

        old_pan = camera.pan_deg
        old_tilt = camera.tilt_deg

        camera.move(
            delta_pan_deg=applied_pan,
            delta_tilt_deg=applied_tilt,
        )

        return {
            "pan_error_deg": pan_error_deg,
            "tilt_error_deg": tilt_error_deg,
            "requested_pan_deg": requested_pan,
            "requested_tilt_deg": requested_tilt,
            "applied_pan_deg": camera.pan_deg - old_pan,
            "applied_tilt_deg": camera.tilt_deg - old_tilt,
            "pan_saturated": abs(requested_pan) > self.max_step_deg,
            "tilt_saturated": abs(requested_tilt) > self.max_step_deg,
        }
