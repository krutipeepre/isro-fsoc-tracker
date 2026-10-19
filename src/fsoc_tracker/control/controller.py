
import math


class ProportionalTracker:
    def __init__(self, gain=0.65, max_step_deg=2.0):
        self.gain = gain
        self.max_step_deg = max_step_deg

    def command(self, camera, u, v):
        intr = camera.intrinsics

        dx = u - intr.cx
        dy = intr.cy - v

        pan_error = math.degrees(
            math.atan(dx / intr.fx)
        )

        fy = (intr.height / 2) / math.tan(
            math.radians(intr.vfov_deg) / 2
        )

        tilt_error = math.degrees(
            math.atan(dy / fy)
        )

        pan_step = max(
            -self.max_step_deg,
            min(self.max_step_deg, self.gain * pan_error)
        )

        tilt_step = max(
            -self.max_step_deg,
            min(self.max_step_deg, self.gain * tilt_error)
        )

        camera.move(
            delta_pan_deg=pan_step,
            delta_tilt_deg=tilt_step
        )

        return pan_error, tilt_error
