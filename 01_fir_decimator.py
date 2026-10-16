# Anti-alias FIR decimation
# Run this file directly with Python 3.12; see README.md for setup.
# Signal generation, processing, checks, and plotting are all contained here.

from pathlib import Path
import json
import numpy as np
import scipy
from scipy import signal
import matplotlib
# Save plots without requiring a GUI or audio hardware. Set the backend before pyplot.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})
# Store outputs beside this script, independent of the current working directory.
out = Path(__file__).resolve().parent / 'results'
out = out / Path(__file__).stem
out.mkdir(parents=True, exist_ok=True)
# Convert amplitude (including complex values) to dB relative to 1.
# Floor at 1e-12 to avoid log(0); this sets a -240 dB display floor.
def db(z):
    return 20*np.log10(np.maximum(np.abs(z), 1e-12))

# 1. Specify the input sample rate in Hz. Decimation by three gives 16 kHz.
# The output Nyquist limit is 8 kHz, so reject higher frequencies BEFORE decimation.
fs = 48000
# Preserve 0-6 kHz; suppress 8-24 kHz before /3.
# Normalize the 2 kHz transition width by the INPUT Nyquist frequency.
# An 80 dB design target provides margin above the 75 dB acceptance threshold.
numtaps, beta = signal.kaiserord(80, 2000/(fs/2))
# Force an odd tap count: bitwise OR with 1 leaves odd values unchanged
# and adds one to even values. This gives an integer input-sample group delay.
numtaps |= 1
# 7 kHz is the middle of the 6-8 kHz transition, not the passband edge.
# The Kaiser beta controls the window sidelobe/transition trade-off.
h = signal.firwin(numtaps, 7000, fs=fs,
                  window=('kaiser', beta))
# 2. Evaluate complex gain on a dense frequency grid (f is in Hz).
# Always measure the realized response; an order estimate is not a guarantee.
f, H = signal.freqz(h, worN=65536, fs=fs)
# The largest stopband magnitude is the worst leakage; negate its dB value
# to report attenuation as a positive number.
stop_db = -np.max(db(H[f >= 8000]))
pass_error = np.max(np.abs(db(H[f <= 6000])))
assert stop_db > 75 and pass_error < 0.01
# Coefficient symmetry verifies the linear-phase FIR construction.
assert np.allclose(h, h[::-1])
# 3. Construct one second of desired 2 kHz tone plus unwanted 11 kHz tone.
t = np.arange(fs)/fs
x = .5*np.sin(2*np.pi*2000*t)
x += .3*np.sin(2*np.pi*11000*t)
# Filter first, then keep every third sample using a polyphase implementation.
# The returned sequence includes FIR startup and tail samples.
y = signal.upfirdn(h, x, down=3)
# Deliberately wrong comparison: skipping filtering folds 11 kHz to 5 kHz.
naive = x[::3]
# Ignore filter startup; fit the two exact output tones.
# Estimate steady-state sinusoidal amplitude without depending on phase.
def amplitude(v, hz):
    # Exclude startup. The 4000-sample output segment spans 0.25 seconds.
    v = v[200:4200]
    tt = np.arange(len(v))/(fs/3)
    # Fit sine and cosine together. Their coefficient norm removes phase dependence
    # (including the phase shift caused by the FIR delay).
    A = np.column_stack((np.sin(2*np.pi*hz*tt),
                         np.cos(2*np.pi*hz*tt)))
    return np.linalg.norm(np.linalg.lstsq(A,v,rcond=None)[0])
# 4. Compare the unwanted 5 kHz component before and after filtering.
# The ratio is an amplitude ratio, so its dB multiplier is 20.
alias_drop = 20*np.log10(amplitude(naive,5000)/
                         amplitude(y,5000))
assert alias_drop > 75
# 5. Separate panels prevent the filtered spectrum from hiding the naive result.
fig, ax = plt.subplots(3,1,figsize=(4.5,6.0),layout='constrained')
ax[0].plot(f/1000, db(H),color='black')
ax[0].set(xlim=(0,12),ylim=(-120,5),
          xlabel='Frequency (kHz)',ylabel='Gain (dB)')
for a,v,label in zip(ax[1:],[naive,y],['Naive /3','FIR then /3']):
    # Welch returns power spectral density, so use 10*log10 rather than 20.
    # Both spectra use the OUTPUT sample rate, 16000 Hz.
    ff,P = signal.welch(v[200:],fs/3,nperseg=2048)
    a.plot(ff/1000,10*np.log10(P+1e-20),color='black')
    a.set(title=label,xlabel='Frequency (kHz)',
          ylabel='PSD (dB/Hz)',ylim=(-150,-10))
# For a symmetric M-tap FIR, delay is (M-1)/2 input samples.
# Divide by the input sample rate and multiply by 1000 for milliseconds.
metrics = dict(taps=numtaps, delay_ms=1000*(numtaps-1)/2/fs,
               stop_attenuation_db=stop_db,
               passband_error_db=pass_error,
               alias_reduction_db=alias_drop)


# Save a high-resolution raster plot, a vector plot, and numerical results.
# Closing the figure releases memory when running many examples in sequence.
for a in fig.axes:
    if a.get_xlabel(): a.grid(alpha=.2)
fig.savefig(out/'figure.png', dpi=320)
fig.savefig(out/'figure.pdf')
plt.close(fig)
# Record dependency versions so numerical results can be traced to an environment.
metrics['numpy_version']=np.__version__
metrics['scipy_version']=scipy.__version__
text=json.dumps(metrics,indent=2)
(out/'metrics.json').write_text(text+'\n')
print(text)
