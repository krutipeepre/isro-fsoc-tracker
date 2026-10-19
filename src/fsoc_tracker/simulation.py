
"""FSOC simulation: moving beacon and configurable disturbances.

All angular values are in degrees.
This is a software simulation, not a physical atmospheric model.
"""

from dataclasses import dataclass
import math
import random


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
        """Advance the simulated target trajectory."""
        self.time_s += max(0.0, dt)

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


@dataclass
class DisturbanceConfig:
    """Adjustable strengths for simulated angular disturbances.

    atmospheric_jitter_deg:
        Random angular jitter applied to the apparent beacon direction.

    vibration_amplitude_deg:
        Peak sinusoidal camera vibration amplitude.

    vibration_frequency_hz:
        Frequency of the simulated camera vibration.

    camera_drift_rate_deg_s:
        Optional constant camera drift rate.

    random_seed:
        Seed for repeatable random jitter. Use None for a varying sequence.
    """

    atmospheric_jitter_deg: float = 0.0
    vibration_amplitude_deg: float = 0.0
    vibration_frequency_hz: float = 8.0
    camera_drift_rate_deg_s: float = 0.0
    random_seed: int | None = 42

    def __post_init__(self):
        if self.atmospheric_jitter_deg < 0:
            raise ValueError("Atmospheric jitter cannot be negative.")
        if self.vibration_amplitude_deg < 0:
            raise ValueError("Vibration amplitude cannot be negative.")
        if self.vibration_frequency_hz < 0:
            raise ValueError("Vibration frequency cannot be negative.")


@dataclass
class DisturbanceModel:
    """Generate deterministic, configurable simulated disturbances."""

    config: DisturbanceConfig

    def __post_init__(self):
        self._rng = random.Random(self.config.random_seed)
        self.time_s = 0.0

    def advance(self, dt: float) -> None:
        """Advance disturbance time."""
        self.time_s += max(0.0, dt)

    def apparent_target_angles(
        self,
        azimuth_deg: float,
        elevation_deg: float,
    ) -> tuple[float, float]:
        """Return target angles with simulated atmospheric jitter."""
        jitter = self.config.atmospheric_jitter_deg

        apparent_azimuth = azimuth_deg + self._rng.gauss(0.0, jitter)
        apparent_elevation = elevation_deg + self._rng.gauss(0.0, jitter)

        return apparent_azimuth, apparent_elevation

    def camera_vibration(self) -> tuple[float, float]:
        """Return simulated pan/tilt vibration offsets."""
        phase = 2.0 * math.pi * self.config.vibration_frequency_hz * self.time_s
        amplitude = self.config.vibration_amplitude_deg

        pan_offset = amplitude * math.sin(phase)
        tilt_offset = amplitude * math.cos(phase)

        return pan_offset, tilt_offset

    def camera_drift(self) -> tuple[float, float]:
        """Return accumulated constant drift offsets."""
        drift = self.config.camera_drift_rate_deg_s * self.time_s
        return drift, drift
