# Overlap-save convolution
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

# Compute full linear convolution using overlap-save for nonempty real 1-D arrays.
# K is FFT length in samples; h is the finite impulse response.
def overlap_save(x,h,K):
    # 1. Each K-sample FFT block contains M-1 history samples and L new samples.
    M=len(h); L=K-M+1
    # Reject an FFT too short to hold the IR; otherwise there is no valid output block.
    if L<1: raise ValueError('FFT must be at least IR length')
    # The final answer must include the convolution tail, not just len(x) samples.
    total=len(x)+M-1
    # Prefix zeros model silence before the input. Suffix zeros flush the IR tail.
    padded=np.r_[np.zeros(M-1),x,np.zeros(M-1)]
    # Extra zeros complete the last valid-output group. (-total)%L gives
    # zero padding when already aligned, otherwise just enough for one full block.
    padded=np.pad(padded,(0,(-total)%L))
    # 2. Precompute the filter FFT once and reuse it for every input block.
    H=np.fft.rfft(h,K); blocks=[]
    # Advance by L, so consecutive K-sample input blocks overlap by M-1 samples.
    for start in range(0,total,L):
        block=padded[start:start+K]
        # Frequency multiplication implements CIRCULAR convolution within this block.
        # NumPy's inverse FFT includes 1/K normalization; do not divide again.
        z=np.fft.irfft(np.fft.rfft(block,K)*H,K)
        # Discard the first M-1 corrupted samples; save the L valid outputs.
        blocks.append(z[M-1:])
    # Remove only padding beyond the true linear-convolution length.
    return np.concatenate(blocks)[:total]
# 3. Compare with direct convolution using a non-block-aligned random input.
rng=np.random.default_rng(9)
fs=48000; x=rng.normal(0,.1,10003)
h=signal.firwin(257,4500,fs=fs)
y=overlap_save(x,h,1024)
# np.convolve supplies an independent time-domain reference.
ref=np.convolve(x,h)
error=np.max(abs(y-ref))
assert error<1e-12 and len(y)==len(ref)
# Edge cases cover short inputs, exact and partial blocks.
# Test short inputs and block-edge lengths. M=1 checks the zero-overlap case.
for nx in [1,17,768,769,1536]:
    for M in [1,17,257]:
        xx=rng.normal(size=nx); hh=rng.normal(size=M)
        yy=overlap_save(xx,hh,1024)
        assert np.allclose(yy,np.convolve(xx,hh),atol=1e-10)
# 4. Plot output and numerical error separately; tiny errors should be visible.
fig,ax=plt.subplots(2,1,figsize=(4.5,4.5),layout='constrained')
ax[0].plot(np.arange(len(y))/fs,y,color='black',lw=.5)
ax[0].set(xlabel='Time (s)',ylabel='Output amplitude')
ax[1].plot(np.arange(len(y))/fs,y-ref,color='black',lw=.5)
ax[1].set(xlabel='Time (s)',ylabel='Reference error')
# 768 new samples at 48 kHz span 16 ms of acquisition.
# This is not a measured processing time or full-system latency.
metrics=dict(max_error=error,valid_samples_per_fft=768,
 block_acquisition_ms=1000*768/fs,edge_cases=15,
 output_samples=len(y))


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
