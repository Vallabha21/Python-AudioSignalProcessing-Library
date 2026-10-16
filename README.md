[README.md](https://github.com/user-attachments/files/33525373/README.md)
# Hands-On Python for Audio Signal Processing
## Ten commented DSP laboratories

Each numbered script is self-contained. Comments explain the signal-processing
steps, units, array shapes, design choices, numerical checks, and common mistakes.
The algorithms and acceptance criteria match the previously delivered examples;
only explanatory comments have been added to the ten laboratory scripts.

All inputs are synthetic. No recordings, microphone, sound card, or network access
are required to run the examples after installing dependencies.

## Setup

Use **Python 3.12**. Unzip the package and open a terminal in this folder.

### Windows (PowerShell or Command Prompt)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe 01_fir_decimator.py
.venv\Scripts\python.exe run_all.py
```

### macOS / Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python 01_fir_decimator.py
.venv/bin/python run_all.py
```

These commands use the environment's interpreter directly; activation is optional.
Dependency installation requires internet access or a local package cache.

### VS Code

Open this folder, select the `.venv` Python interpreter, open any numbered file,
and choose **Run Python File in Terminal**. Run complete scripts, not isolated
selections: later steps depend on variables initialized earlier in each script.

## Example guide

| File | What the comments explain |
| --- | --- |
| `01_fir_decimator.py` | FIR specifications, Kaiser design, anti-alias filtering, sample-rate change, amplitude fitting |
| `02_sos_streaming.py` | Biquad states, partial buffers, continuous/reference equivalence, reset-state failure |
| `03_notch_drift.py` | Quality factor, settling, hum drift, least-squares tone measurement |
| `04_crossover.py` | Cascaded Butterworth branches, complex addition, all-pass summed phase |
| `05_multiresolution_stft.py` | Window duration, hop, bin spacing, complex coefficients, exact reconstruction |
| `06_phase_frequency.py` | Frame convention, conjugate phase differences, residual wrapping, off-bin estimation |
| `07_spectral_denoising.py` | Noise calibration, broadcasting, gain masking, smoothing, SNR validation |
| `08_room_convolution.py` | Synthetic source/IR, full output length, tail preservation, common WAV gain |
| `09_overlap_save.py` | History overlap, circular corruption, valid output blocks, padding and tail flushing |
| `10_deconvolution.py` | Weak excitation, regularization, known support, noise-level bias trade-offs |

Read the scripts in numerical order, or choose any one independently. Change one
parameter at a time and compare both the plots and measured values.

## Outputs

Each script creates or refreshes `results/<script_name>/` containing:

- `figure.pdf`: vector plot.

Example 08 additionally creates `dry.wav` and `wet.wav` (16 kHz, float32 WAV).
Both files use one shared gain so their level relationship is preserved.
Precomputed outputs are included to make inspection possible before execution.
Re-running a script replaces that script's output files.

Plots are saved rather than displayed in a popup. Open the image or PDF after
execution. Graphs use separate panels and grayscale-safe colors.

## Checks and reproducibility

`run_all.py` runs all ten examples in separate processes, prints PASS/FAIL, and
writes `test_report.json`. Its exit status is nonzero if any example fails.
Do **not** use Python's `-O` option: it disables the assertions that perform the checks.
The included report records the actual run of this commented package.
`SOURCE_SHA256.json` records checksums of the delivered Python source files.

Pinned environment: Python 3.12, NumPy 2.3.5, SciPy 1.17.0,
Matplotlib 3.10.8. Small floating-point differences across platforms are normal.
Random seeds keep the synthetic fixtures reproducible within the supported setup.

These scripts are educational offline experiments. Plotting, allocation and file
I/O are not audio-callback code. Numerical correctness does not establish real-time
performance. The denoising and deconvolution improvements apply to the documented
fixtures; example 10 explicitly includes a case where regularization worsens error.

## Troubleshooting

- **ModuleNotFoundError:** install requirements with the same interpreter used to run the file.
- **No plot window:** expected; open the saved `results/.../figure.png`.
- **AssertionError:** inspect the traceback and `test_report.json`; check whether you changed a parameter or dependency version.
- **Cannot create output files:** extract the ZIP to a writable folder before running.
- **Several seconds of silence:** some scripts spend time generating figures; `run_all.py` prints a status after each completes.
