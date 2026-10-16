# Notch filtering and frequency drift
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

# 1. Define the sampling rate in Hz and design a narrow mains-hum notch.
fs=48000
# Q = center frequency / -3 dB bandwidth. Here bandwidth is 50/30 Hz.
# Higher Q narrows the notch but increases sensitivity to frequency drift.
q=30
# b contains numerator coefficients; a contains denominator coefficients.
b,a=signal.iirnotch(50,q,fs=fs)
t=np.arange(3*fs)/fs
# The desired 440 Hz sinusoid provides a separate check on signal preservation.
clean=.2*np.sin(2*np.pi*440*t)
# Filter one hum frequency and estimate its residual amplitude after startup.
def measure(hum):
    x=clean+.2*np.sin(2*np.pi*hum*t)
    # Use causal filtering, with zero initial state. The startup transient is real.
    y=signal.lfilter(b,a,x)
    # Exclude the first second so the amplitude estimate reflects settled behavior.
    sl=slice(fs,None)
    tt=t[sl]
    # Fit both hum and wanted tone in the same least-squares model.
    # Two columns per frequency (sine/cosine) allow arbitrary phase.
    A=np.column_stack([np.sin(2*np.pi*hum*tt),
                       np.cos(2*np.pi*hum*tt),
                       np.sin(2*np.pi*440*tt),
                       np.cos(2*np.pi*440*tt)])
    coef=np.linalg.lstsq(A,y[sl],rcond=None)[0]
    # The first two coefficients describe hum. Their norm is its amplitude;
    # compare with the known input amplitude 0.2 to obtain attenuation in dB.
    return y,20*np.log10(np.linalg.norm(coef[:2])/.2)
# 2. Compare exactly centered hum with a 1 Hz drift from the notch center.
y,at50=measure(50)
_,at51=measure(51)
# Expect deep rejection at 50 Hz but weak rejection at 51 Hz.
# The assertions encode this limitation rather than hiding it.
assert at50 < -40 and at51 > -10
# 3. Inspect a dense frequency grid near the notch.
f=np.linspace(40,60,4001)
_,H=signal.freqz(b,a,worN=f,fs=fs)
# Check wanted-tone gain independently of the time-domain regression.
_,tone=signal.freqz(b,a,worN=[440],fs=fs)
assert abs(db(tone)[0]) < .01
# An impulse reveals the filter memory. One second captures its ringing tail.
imp=np.zeros(fs); imp[0]=1
h=signal.lfilter(b,a,imp)
fig,ax=plt.subplots(2,1,figsize=(4.5,4.5),layout='constrained')
ax[0].plot(f,db(H),color='black')
ax[0].set(xlabel='Frequency (Hz)',ylabel='Gain (dB)',ylim=(-80,1))
# Omit the large direct sample at n=0 so the much smaller ringing is visible.
ax[1].plot(np.arange(1,fs)/fs,h[1:],color='black')
ax[1].set(xlabel='Time (s)',ylabel='Impulse tail',
          title='Direct sample omitted')
# Negative gain values denote attenuation; bandwidth is expressed in Hz.
metrics=dict(hum_50_db=at50,hum_51_db=at51,
 bandwidth_hz=50/q,tone_440_gain_db=float(db(tone)[0]))


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
