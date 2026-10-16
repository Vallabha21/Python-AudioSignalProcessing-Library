# Fourth-order electrical crossover
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

# 1. Set sample rate and the shared electrical crossover frequency (Hz).
fs=48000; fc=2000
# Design one second-order Butterworth low-pass section.
lo2=signal.butter(2,fc,fs=fs,output='sos')
# Use the same cutoff and sample rate for the high-pass section.
hi2=signal.butter(2,fc,fs=fs,btype='highpass',output='sos')
# Cascading two identical sections squares their transfer function.
# This constructs the two fourth-order Linkwitz-Riley branches.
lo=np.vstack([lo2,lo2]); hi=np.vstack([hi2,hi2])
# A logarithmic grid covers low and high audio frequencies efficiently.
f=np.geomspace(20,20000,4096)
# 2. Keep COMPLEX branch responses: their phase matters when outputs sum.
_,L=signal.sosfreqz(lo,worN=f,fs=fs)
_,H=signal.sosfreqz(hi,worN=f,fs=fs)
# Add complex transfer functions, not absolute magnitudes.
# The electrical sum has unity magnitude but frequency-dependent phase.
S=L+H
# Measure the largest absolute departure from 0 dB over this grid.
flat_error=np.max(np.abs(db(S)))
assert flat_error < 1e-9
# Each branch should be -6.0206 dB at the crossover, not -3 dB.
_,Lc=signal.sosfreqz(lo,worN=[fc],fs=fs)
assert abs(float(db(Lc)[0])+6.020599913) < 1e-6
# A unity magnitude sum is not a zero-delay wire.
# 3. A unit impulse tests whether the summed system equals an identity path.
x=np.zeros(2048); x[0]=1
# Add time-domain branch outputs with the same polarity.
y=signal.sosfilt(lo,x)+signal.sosfilt(hi,x)
# A nonzero impulse difference demonstrates that flat magnitude alone
# does not mean unchanged waveform or zero delay.
assert np.max(np.abs(y-x)) > .1
fig,ax=plt.subplots(3,1,figsize=(4.5,6),layout='constrained')
for a,v,label in zip(ax[:2],[L,H],['Low branch','High branch']):
    a.semilogx(f,db(v),color='black')
    a.set(title=label,xlabel='Frequency (Hz)',
          ylabel='Gain (dB)',ylim=(-60,3))
# Unwrap 2*pi jumps to show the continuous phase of the summed response.
phase=np.unwrap(np.angle(S))
ax[2].semilogx(f,phase,color='black')
ax[2].set(title='Electrical sum: all-pass phase',
           xlabel='Frequency (Hz)',ylabel='Phase (rad)')
# These checks concern an electrical model, not the acoustic sum of real drivers.
metrics=dict(sum_flatness_error_db=flat_error,
 crossover_branch_db=float(db(Lc)[0]),
 impulse_difference=float(np.max(np.abs(y-x))))


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
