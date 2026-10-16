# Synthetic room convolution
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

# 1. Use a one-second synthesized source sampled at 16 kHz.
fs=16000; t=np.arange(fs)/fs
# Constructed plucked-string-like excitation, not a recording.
# Add twelve harmonics with 1/k amplitudes and frequency-dependent decay.
# This resembles a pluck but is not a physically validated instrument model.
x=sum((.2/k)*np.sin(2*np.pi*220*k*t)*
      np.exp(-t*(3+.8*k)) for k in range(1,13))
# 2. Build a reproducible synthetic impulse response (IR).
rng=np.random.default_rng(8)
th=np.arange(int(.7*fs))/fs
# The random tail has a 0.10 s amplitude time constant, not a measured RT60.
h=.008*rng.normal(size=len(th))*np.exp(-th/.10)
# Add direct sound plus reflections at 35 ms and 73 ms.
# Convert delay in seconds to an integer sample index before insertion.
h[0]+=1; h[int(.035*fs)]+=.45; h[int(.073*fs)]+=.25
# 3. Full linear convolution preserves the response after the source ends.
wet=signal.fftconvolve(x,h,mode='full')
# For input length Nx and IR length M, full output length is Nx+M-1.
assert len(wet)==len(x)+len(h)-1
# Independent direct reference on a short input, full IR.
small=x[:300]
# Check FFT convolution against independent direct convolution on a short source.
assert np.allclose(signal.fftconvolve(small,h),
                   np.convolve(small,h),atol=1e-12)
# A one-sample unit impulse must reproduce the input exactly (within roundoff).
assert np.allclose(signal.fftconvolve(x,[1]),x)
# Energy beyond the original source length proves that cropping would lose content.
tail_energy=np.sum(wet[len(x):]**2)
assert tail_energy>0
# Common gain preserves the wet/dry comparison.
# 4. Choose ONE gain for both listening files. Independent normalization
# would conceal the wet/dry level relationship. Peak target 0.95 leaves headroom.
scale=.95/max(np.max(abs(x)),np.max(abs(wet)))
from scipy.io import wavfile
# Write float32 WAV, not 24-bit PCM. File I/O belongs outside an audio callback.
wavfile.write(str(out/'dry.wav'),fs,(scale*x).astype('float32'))
wavfile.write(str(out/'wet.wav'),fs,(scale*wet).astype('float32'))
# 5. Separate source, IR, and output panels show their different durations.
fig,ax=plt.subplots(3,1,figsize=(4.5,6),layout='constrained')
labels=['Dry source','Synthetic room IR','Full wet output']
for a,v,label in zip(ax,[x,h,wet],labels):
    a.plot(np.arange(len(v))/fs,v,color='black',lw=.6)
    a.set(title=label,xlabel='Time (s)',ylabel='Amplitude')
metrics=dict(input_samples=len(x),ir_samples=len(h),
 output_samples=len(wet),tail_energy=tail_energy,
 shared_wav_gain=scale,unscaled_wet_peak=np.max(abs(wet)))


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
