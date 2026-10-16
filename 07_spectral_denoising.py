# Spectral denoising
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

# 1. Construct three seconds of audio; the opening interval contains noise only.
fs=16000; t=np.arange(3*fs)/fs
rng=np.random.default_rng(7)
# A smooth envelope turns the harmonic signal on at 0.6 s and off at 2.8 s.
env=np.zeros_like(t)
on=(t>=.6)&(t<=2.8)
env[on]=np.sin(np.pi*(t[on]-.6)/2.2)**2
# Keep the clean target for objective error measurement; it is not available
# to the denoising algorithm itself.
clean=env*(.25*np.sin(2*np.pi*220*t)+
           .12*np.sin(2*np.pi*440*t)+.08*np.sin(2*np.pi*660*t))
# Add reproducible stationary white noise with standard deviation 0.08.
x=clean+.08*rng.normal(size=len(t))
# 2. Analyze 512-sample Hann windows at 128-sample hops.
st=signal.ShortTimeFFT(signal.windows.hann(512,sym=False),
                       hop=128,fs=fs,scale_to='magnitude')
Z=st.stft(x); times=st.t(len(x))
# Use only fully interior noise-only frames.
# Restrict calibration to frames entirely inside the noise-only interval.
noise_frames=(times>.05)&(times<.45)
# Average squared magnitude across calibration frames for each frequency bin.
# keepdims=True gives shape (bins,1), which broadcasts across all time frames.
noise=np.mean(abs(Z[:,noise_frames])**2,axis=1,keepdims=True)
# Instantaneous noisy power estimate, with shape (bins,frames).
P=abs(Z)**2
# 3. Estimate a Wiener-like amplitude gain. Oversubtraction factor 1.5
# adds suppression; floor 0.08 avoids fully muting bins. Epsilon avoids division by zero.
G=np.maximum(.08,1-1.5*noise/(P+1e-15))
# Smooth gains across 3 frequency bins and 5 time frames to reduce isolated changes.
# The kernel sums to one. Centered time smoothing uses future frames: offline processing.
G=signal.convolve2d(G,np.ones((3,5))/15,mode='same',boundary='symm')
# Apply the REAL, nonnegative gain to COMPLEX coefficients, preserving observed phase.
# This retains noisy phase; it does not recover the clean signal phase.
y=st.istft(G*Z,k1=len(x))
# Check unmodified reconstruction separately from the denoising algorithm.
assert np.max(abs(st.istft(Z,k1=len(x))-x))<1e-12
# 4. Evaluate both input and output over the SAME active-signal interval.
active=(t>.8)&(t<2.6)
# Compute clean-signal energy divided by reconstruction-error energy, in dB.
def snr(v):
    return 10*np.log10(np.sum(clean[active]**2)/
                       np.sum((v[active]-clean[active])**2))
# Require at least 3 dB improvement on this fixture, not all speech recordings.
assert snr(y)>snr(x)+3
fig,ax=plt.subplots(2,1,figsize=(4.5,5.5),layout='constrained')
# Keep color limits equal so visual improvements cannot come from rescaling.
for a,V,title in zip(ax,[Z,st.stft(y)],['Noisy','Processed']):
    im=a.pcolormesh(times,st.f,db(V),shading='auto',
                   cmap='gray_r',vmin=-75,vmax=-15)
    a.set(title=title,xlabel='Time (s)',ylabel='Frequency (Hz)',
          xlim=(0,3),ylim=(0,3000))
    fig.colorbar(im,ax=a,label='Magnitude (dB re 1)')
metrics=dict(input_snr_db=snr(x),output_snr_db=snr(y),
             improvement_db=snr(y)-snr(x),gain_min=float(G.min()),
             gain_max=float(G.max()))


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
