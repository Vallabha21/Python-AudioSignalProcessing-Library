# Frequency estimation from STFT phase
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

# 1. N is window length, hop is frame advance (samples), f0 is the true tone (Hz).
fs=16000; N=1024; hop=128; f0=1003.7
# Use a noiseless isolated sinusoid to test the phase-estimation mathematics.
n=np.arange(fs); x=.5*np.sin(2*np.pi*f0*n/fs)
# A Hann window suppresses leakage from the real sine's negative-frequency image.
w=signal.windows.hann(N,sym=False)
# Explicit left-aligned frames fix the phase convention.
# Include only complete frames. Each FFT uses local indices starting at zero.
starts=np.arange(0,len(x)-N+1,hop)
# rfft keeps nonnegative frequencies. Transpose so rows index bins
# and columns index successive frames.
Z=np.array([np.fft.rfft(x[s:s+N]*w) for s in starts]).T
# 2. Select the nearest FFT bin: fk=1000 Hz for the 1003.7 Hz test tone.
# Knowing a suitable bin is an assumption; this is not a general pitch tracker.
k=int(round(f0*N/fs)); fk=k*fs/N
# Remove expected bin rotation, then wrap to [-pi,pi].
# Multiplying by the previous coefficient's conjugate subtracts phase angles.
# np.angle returns only the principal angle, losing whole 2*pi rotations.
phase_step=np.angle(Z[k,1:]*np.conj(Z[k,:-1]))
# Remove the phase advance expected at the bin center, then wrap again.
# The residual represents the off-bin frequency, not the whole tone frequency.
residual=np.angle(np.exp(1j*(phase_step-2*np.pi*k*hop/N)))
# Convert radians per hop into hertz, then add the bin-center frequency.
# The residual must lie within +/-fs/(2*hop), here +/-62.5 Hz.
fhat=fk+fs*residual/(2*np.pi*hop)
# 3. Compare every estimate with known truth; also show that bin-only estimation errs.
error=np.max(np.abs(fhat-f0))
assert error < .01
assert abs(fk-f0)>1
fig,ax=plt.subplots(2,1,figsize=(4.5,4.5),layout='constrained')
f=np.fft.rfftfreq(N,1/fs)
# Normalize by the window sum. This is the positive-frequency magnitude
# of a real sine without a factor-of-two amplitude correction.
ax[0].plot(f,db(Z[:,10]/w.sum()),color='black')
ax[0].set(xlim=(900,1100),ylim=(-90,0),
          xlabel='Frequency (Hz)',ylabel='Magnitude (dB re 1)')
# Frame-center times label the error. The estimate uses adjacent frame phases.
ax[1].plot((starts[1:]+N/2)/fs,fhat-f0,color='black')
ax[1].set(xlabel='Frame centre (s)',ylabel='Frequency error (Hz)')
# The tiny error applies only to this controlled tone, not arbitrary audio.
metrics=dict(nearest_bin_hz=fk,actual_hz=f0,
             estimated_hz=np.median(fhat),max_error_hz=error,
             unambiguous_residual_hz=fs/(2*hop))


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
