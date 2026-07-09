# BioMotion — webcam hand-motion tracking for biomedical signal extraction

**Goal:** can a webcam + open-source hand-tracking model (MediaPipe) recover
clinically meaningful motor signals:  tremor frequency, tap decrement, joint
ROM, bilateral asymmetry  using plain signal processing, with no medical
dataset and no trained model?

## What we built

- **`python/`** — capture (both hands, all 21 landmarks/hand) + 5 analysis
  scripts: tremor frequency (FFT/Welch), tap rate & rhythm, amplitude/speed
  decrement (bradykinesia-style), per-finger ROM, bilateral L/R asymmetry.
- **`web/`** — live in-browser version: glowing trail, live frequency
  readout, and a digitized Archimedes spiral-tracing test.

## Demo

<p align="center">
  <img src="docs/demo.gif" width="480">
</p>

10s clip, both hands tracked live, per-hand frequency readout on screen.
Recorded with `python record_demo.py --seconds 10`, which also auto-saves
the landmark CSV and generates the PSD plot below from that same take:

<p align="center">
  <img src="docs/demo_tremor_Left.png" width="420">
  <img src="docs/demo_tremor_Right.png" width="420">
</p>

**Result: Left 3.52 Hz, Right 3.02 Hz** — clean sustained in-band peaks, live, not post-hoc.

## What we did to test it

1. **Tremor frequency**: shook hand in time with a metronome at known BPM,
   compared detected frequency to `BPM/60`. Ran a still-hand baseline too.
2. **Tap decrement**: tapped normally, let amplitude/speed fade — checked
   the decrement slope goes negative when it should.
3. **ROM**: bent fingers through full range, checked angle trace looked right.
4. **Bilateral asymmetry**: shook one hand more than the other, checked the
   asymmetry index responded.

## Results

| Target | Detected | Verdict |
|---|---|---|
| still baseline | no peak | correct |
| 2 Hz metronome | 2.04 Hz | matches (~2% error) |
| 3 Hz metronome | 5.10 Hz | mismatch — drifted faster than metronome |
| 5 Hz metronome | no peak | not detected — webcam fps ceiling (~20fps, Nyquist limits reliable detection above ~5-6 Hz) |

<table>
<tr>
<td><img src="docs/images/tremor_baseline.png" width="360"></td>
<td><img src="docs/images/tremor_2hz_validated.png" width="360"></td>
</tr>
</table>

Found and fixed two real bugs during validation: naive peak-search was
reporting the search band's edge as a "peak" for a stationary hand (fixed
with `scipy.signal.find_peaks` + prominence threshold); slow arm drift was
swamping real tremor peaks (fixed with a high-pass filter). Both in
`tremor_math.py`.

Tap / decrement / ROM also run cleanly on real recordings:

<p align="center">
<img src="docs/images/taps.png" width="320">
<img src="docs/images/decrement.png" width="320">
<img src="docs/images/rom.png" width="320">
</p>

10 taps @ 0.87/s, rhythm CV 0.42 · amplitude trend +1.9% (no decrement) · 176-178° ROM across fingers.

## Setup

```
cd python
pip install -r requirements.txt
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Usage

```
python capture.py --session name        # r=record  c=clear trail  q=quit
python analyze_tremor.py ../data/name.csv
python analyze_taps.py ../data/name.csv
python analyze_decrement.py ../data/name.csv
python analyze_rom.py ../data/name.csv
python analyze_asymmetry.py ../data/name.csv   # needs both hands in the clip
python record_demo.py --seconds 10       # hands-free demo recorder
```

All analysis scripts take `--hand Left`/`--hand Right` (default: whichever
hand has more rows).

Web demo: `cd web && python -m http.server 8000`, open `localhost:8000`.

## Data schema

One row per hand per frame: `frame, t_sec, hand, handedness_score, lm0_x, lm0_y, lm0_z, ... lm20_x, lm20_y, lm20_z`
(standard MediaPipe landmark indexing — see `landmarks.py`).

## Extending

- Inter-finger independence during tapping (spasticity/coordination marker)
- Grip aperture over time (reach-to-grasp kinematics)
- Port the web spiral test into Python for offline scoring
