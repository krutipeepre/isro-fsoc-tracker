# ISRO FSOC Virtual Camera Tracker — Milestone 1

A CPU-friendly starter project for Problem Statement 26169. This milestone implements:
- A pinhole-style angular camera projection model
- A moving target trajectory
- A virtual camera with pan/tilt controls
- A simple OpenCV-rendered beacon view
- Unit tests for camera geometry and field-of-view behavior

This is a **software simulation**, not a physical FSOC link or a validated atmospheric model.

## Requirements
- Windows 10/11
- Python 3.11 recommended
- CPU is sufficient

## Setup (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If `py -3.11` is unavailable, try `py -m venv .venv` with your installed Python version.

## Run the interactive simulator
```powershell
python -m fsoc_tracker.main
```

Controls:
- Arrow keys: pan/tilt camera
- `R`: reset camera
- `Space`: pause/resume target motion
- `Q` or `Esc`: quit

The target moves on a deterministic trajectory. The camera is initially fixed, so the beacon moves in the image. Use the arrow keys to explore how pan/tilt changes the projection.

## Run acceptance tests
```powershell
python -m pytest -q
```

Expected: all tests pass.

## Milestone 1 acceptance criteria
1. A target directly along the camera's optical axis projects to the image center.
2. Positive target azimuth appears to the right when camera pan is zero.
3. Increasing camera pan shifts a stationary target left in the image.
4. A target outside the configured field of view is reported as not visible.
5. The demo runs without a GPU.

## Known limitations
- The scene is angular/kinematic rather than a full 3D renderer.
- The image is a synthetic visualization, not an optical propagation simulation.
- No automated beacon detector or closed-loop controller is included yet; those are subsequent milestones.
