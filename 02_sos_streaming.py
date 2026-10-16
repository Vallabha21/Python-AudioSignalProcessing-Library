# Stateful SOS filtering
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

# 1. Generate a reproducible mono test signal at 48 kHz.
fs=48000
# A fixed seed makes the numerical experiment repeatable.
rng=np.random.default_rng(2)
# 12017 samples intentionally leave a partial final buffer for the tested sizes.
x=rng.normal(0,.1,12017)
# A tenth-order elliptic prototype becomes a twentieth-order bandpass.
# Use ten second-order sections (SOS), not one high-order polynomial.
# The parameters specify 0.2 dB passband ripple and 80 dB stopband attenuation.
sos=signal.ellip(10, .2, 80, [300,6000],
                 btype='bandpass',fs=fs,output='sos')
# 2. Process the entire signal once from zero state: the trusted reference.
reference=signal.sosfilt(sos,x)
# Process the same samples in chunks while preserving the filter memory.
def stream(block):
    # Two delay states per biquad for this mono signal. Initialize once per stream,
    # not once per block. Separate channels require separate state arrays.
    state=np.zeros((len(sos),2))
    parts=[]
    # Slicing naturally handles a final block shorter than the nominal block size.
    for start in range(0,len(x),block):
        # Pass the previous final state as zi and retain the returned new final state.
        # This is the essential difference between continuous filtering and repeated restarts.
        y,state=signal.sosfilt(sos,x[start:start+block],zi=state)
        parts.append(y)
    return np.concatenate(parts)
# 3. Check multiple block sizes, including sample-by-sample processing.
errors=[]
for block in [1,64,127,256,1024]:
    y=stream(block)
    errors.append(float(np.max(np.abs(y-reference))))
    # Compare against the one-shot result with float64-appropriate tolerances.
    assert np.allclose(y,reference,atol=1e-12,rtol=1e-12)
block=127
y=stream(block)
# Negative control: omitting zi restarts each block from zero.
# This is intentionally incorrect, so its error should be clearly nonzero.
bad=np.concatenate([signal.sosfilt(sos,x[s:s+block])
                    for s in range(0,len(x),block)])
assert np.sqrt(np.mean((bad-reference)**2)) > .001
# The last three SOS coefficients form each denominator polynomial.
# Poles inside the unit circle indicate stability for the fixed-coefficient filter.
poles=np.concatenate([np.roots(row[3:]) for row in sos])
assert np.max(np.abs(poles)) < 1
# 4. Plot ERROR relative to the reference, rather than nearly identical outputs.
fig,ax=plt.subplots(2,1,figsize=(4.5,4.5),layout='constrained')
n=np.arange(1000)
ax[0].plot(n/fs*1000,(y-reference)[:1000],color='black')
ax[1].plot(n/fs*1000,(bad-reference)[:1000],color='black')
for a,title in zip(ax,['Preserved state error','Reset state error']):
    a.set(title=title,xlabel='Time (ms)',ylabel='Amplitude error')
# Record the worst block error, the broken-path RMS error, and pole radius.
metrics=dict(max_block_error=max(errors),
 reset_rms_error=np.sqrt(np.mean((bad-reference)**2)),
 max_pole_radius=np.max(np.abs(poles)),
 tested_block_sizes=[1,64,127,256,1024])


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
