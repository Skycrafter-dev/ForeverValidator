# Ordinary In-Race Camera Oracle

`ordinary-race.bin` contains observed inputs and outputs from the original
TrackMania United Forever executable, not output from ForeverViewer or a
MediaTracker/forced-render reference. `native_camera_config.h` contains the
corresponding live controller coefficients and an exact resource check.

## Provenance

- Recorded on 2026-09-24 from an offline Stadium B05 race, launched directly
  under Wine. No TMLoader or TMInterface was loaded; MediaTracker was inactive.
- Original `TmForever.exe` SHA-256:
  `4b6a7b31d86766409e94101f1256cd61dffecb23ea497a40b6617506b2ced2d4`.
- A fresh Ghidra import and fresh captures were used. Earlier camera research,
  forced-render traces and reconstructed outputs were not used as the oracle.
- Ordinary dispatch was observed through `0047e020 -> 005e34e0 -> 00693810 ->
  00693040 -> controller vtable + 7c`; the update return site was `0069308b`.
- Native key 1 selects collector index 0, `CGameControlCameraTrackManiaRace2`
  (update `0045e070`, reset `0045d130`). Key 2 selects index 1,
  `CGameControlCameraTrackManiaRace3` (update `0045a660`, reset `00459d00`).
  Key 3 selects index 2, `CGameControlCamera`.
- A passive observer wrapped the original update/reset entries and forwarded
  every call unchanged. It recorded the actual visual and reset poses, controls,
  velocity, speed, wheel contact/surface flags and output matrix/lens. The game
  executable on disk was never modified. Driving, selection and restarts were
  ordinary keyboard inputs; the observer did not request simulation, resets,
  resynchronization or renders.
- A second capture also observed native static-group collision calls at
  `005399c0`, restricted to camera return site `0045b433`. It recorded the query
  and original result without substituting collision behavior.

The raw observations, observer sources, fresh disassembly and screenshots are
retained locally at
`/home/mikael/fv-evidence/in-race-camera-from-binary-20260924/`. No game executable
code or process addresses are included in the portable fixture.

## Coverage

The fixture has 2,844 records: 2,830 updates and 14 resets. Camera 1 covers 1,640
updates, 8 resets, 130 steering samples, 276 airborne samples and 98 reverse
samples. Camera 2 covers 1,190 updates, 6 resets, 103 steering samples, 85 airborne
samples and 156 reverse samples. There are 153 successful native collision
queries, in addition to misses and frames on which no query is issued.

Both raw controllers match every recorded matrix/lens float bit-for-bit,
including signed zero, in both the test and Release builds. The public
API's maximum position error is 0.0000610352 m for camera 1 and 0.0000305176 m
for camera 2; maximum matrix-component error after quaternion conversion is
0.00000980496 and 0.00000686952 respectively. Lens values remain exact. These
figures are reported afresh by the test, not substituted for its comparisons.

The tests replay every retained update, including the distinction between the
simulation pose read during reset and the visual pose read during normal update.
They check the raw matrix controller and public quaternion API separately and
validate collision-query geometry and dispatch. Installed-resource verification
also compares all used coefficients and curve keys against the live instance.

This is evidence for the captured ordinary-race scenarios, not proof of every
vehicle, track, turbo/roll-input branch, or frame schedule. The legacy Race and
internal-camera implementations are checked against the executable and synthetic
tests, but are not covered by this numerical recording. Public quaternion
conversion introduces rounding that the native matrix-only path does not have.

## Reproduction

```sh
python3 tools/make_native_camera_fixture.py /path/to/raw-capture tests/data/camera
cmake --build build-viewer --target forevervalidator-native-race-camera-tests
build-viewer/forevervalidator-native-race-camera-tests
build-viewer/forevervalidator-native-race-camera-tests \
  tests/data/camera/ordinary-race.bin /path/to/TmUnitedForever/Packs
```

The generator uses `native-inputs.bin` (camera 1),
`native-inputs-collision.bin` (both controllers), `native-collision.bin`,
`native-config-2.bin`, `native-config-3.bin`, and `native-runtime-config.json`.
It includes only reset-started sequences up to race-clock 20,000 ms, strips
addresses/raw object storage and excludes incomplete buffered records. Native
outputs are copied directly, not recomputed. Nested target-change resets are
preserved before their enclosing update.

The little-endian fixture header is `(magic=0x46564332, recordSize=296, count)`.
Each record is `<3I63fI7f`: controller kind (2/3), phase (1=update, 3=reset),
clock ms, 63 floats, collision kind (0=none, 1=miss, 2=hit), then start XYZ,
displacement XYZ and hit fraction. The 63 floats contain visual matrix (12),
reset matrix (12), signed speed (1), support up (3), velocity (3), controls (3),
turbo/flight/burning/gear signals (4), wheel contacts (4), wheel surfaces (4),
and native output matrix/lens (17).

For end-to-end cadence checks, `FOREVERVIEWER_CAMERA_TRACE=1` logs the visible
target, simulation tick and camera for each encoded frame. Presentation tests
exercise 30, 60, 120 and 144 fps, respawns and selection resets. Above 100 fps,
frames sharing a physics tick must still advance the retained camera.
