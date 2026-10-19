"""Camera geometry for the Milestone 1 angular simulator."""
from dataclasses import dataclass
import math
from typing import Optional, Tuple


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int = 640
    height: int = 480
    hfov_deg: float = 60.0

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive.")
        if not 0.0 < self.hfov_deg < 179.0:
            raise ValueError("hfov_deg must be between 0 and 179 degrees.")

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def vfov_deg(self) -> float:
        return math.degrees(
            2.0 * math.atan((self.height / 2.0) / self.fx)
        )

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2.0


@dataclass
class VirtualCamera:
    """Pan and tilt are in degrees; positive pan points toward +world azimuth."""
    intrinsics: CameraIntrinsics
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    max_pan_deg: float = 90.0
    max_tilt_deg: float = 45.0

    def move(self, delta_pan_deg: float = 0.0, delta_tilt_deg: float = 0.0) -> None:
        self.pan_deg = max(
            -self.max_pan_deg,
            min(self.max_pan_deg, self.pan_deg + delta_pan_deg),
        )
        self.tilt_deg = max(
            -self.max_tilt_deg,
            min(self.max_tilt_deg, self.tilt_deg + delta_tilt_deg),
        )

    def reset(self) -> None:
        self.pan_deg = 0.0
        self.tilt_deg = 0.0

    def project_angles(
        self, azimuth_deg: float, elevation_deg: float
    ) -> Optional[Tuple[float, float]]:
        """
        Project a target specified by world azimuth/elevation.

        Returns (u, v) pixels when inside the camera FOV; otherwise None.
        The angular camera model is appropriate for this first milestone.
        """
        az = math.radians(azimuth_deg - self.pan_deg)
        el = math.radians(elevation_deg - self.tilt_deg)

        # Pinhole projection in the angular plane. Reject rays too far from
        # the optical axis before tangent projection becomes ill-conditioned.
        half_h = math.radians(self.intrinsics.hfov_deg / 2.0)
        half_v = math.radians(self.intrinsics.vfov_deg / 2.0)
        if abs(az) >= half_h or abs(el) >= half_v:
            return None

        u = self.intrinsics.cx + self.intrinsics.fx * math.tan(az)
        fy = (self.intrinsics.height / 2.0) / math.tan(
            math.radians(self.intrinsics.vfov_deg) / 2.0
        )
        v = self.intrinsics.cy - fy * math.tan(el)

        if not (0.0 <= u < self.intrinsics.width and
                0.0 <= v < self.intrinsics.height):
            return None
        return (u, v)
