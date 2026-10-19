"""Deterministic target trajectory for the first simulator milestone."""
from dataclasses import dataclass
import math


@dataclass
class MovingBeacon:
    """A beacon moving in world azimuth/elevation, in degrees."""
    time_s: float = 0.0
    azimuth_center_deg: float = 0.0
    azimuth_amplitude_deg: float = 18.0
    azimuth_frequency_hz: float = 0.12
    elevation_center_deg: float = 2.0
    elevation_amplitude_deg: float = 7.0
    elevation_frequency_hz: float = 0.08

    def advance(self, dt: float) -> None:
        self.time_s += dt

    @property
    def azimuth_deg(self) -> float:
        return self.azimuth_center_deg + self.azimuth_amplitude_deg * math.sin(
            2.0 * math.pi * self.azimuth_frequency_hz * self.time_s
        )

    @property
    def elevation_deg(self) -> float:
        return self.elevation_center_deg + self.elevation_amplitude_deg * math.sin(
            2.0 * math.pi * self.elevation_frequency_hz * self.time_s
        )
