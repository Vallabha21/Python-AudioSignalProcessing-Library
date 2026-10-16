# Multi-resolution complex STFT
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

# 1. Create a two-second, 16 kHz fixture with competing time-frequency demands.
fs=16000
n=np.arange(2*fs); t=n/fs
# Two close tones, chirp, and a short broadband event.
# The two steady tones are only 80 Hz apart; short windows struggle to resolve them.
x=.15*np.sin(2*np.pi*1000*t)+.15*np.sin(2*np.pi*1080*t)
# Add a linear chirp sweeping from 300 Hz to 3000 Hz.
x+=.2*signal.chirp(t,300,t[-1],3000)
rng=np.random.default_rng(5)
# The Gaussian envelope concentrates broadband noise near t=1 s.
# A long window spreads this short event over time.
x+=.3*rng.normal(size=len(x))*np.exp(-((t-1)/.004)**2)
fig,ax=plt.subplots(2,1,figsize=(4.5,5.5),layout='constrained')
metrics={}
# 2. Compare 8 ms and 64 ms windows on identical data.
for a,N in zip(ax,[128,1024]):
    # Use a periodic Hann window and a quarter-window hop (75% overlap).
    # Magnitude scaling normalizes the transform; it is not an SPL calibration.
    st=signal.ShortTimeFFT(signal.windows.hann(N,sym=False),
           hop=N//4,fs=fs,scale_to='magnitude')
    # Z has shape (frequency bins, time frames) and retains complex phase.
    Z=st.stft(x)
    # 3. Reconstruct from the UNMODIFIED complex STFT. k1 trims to input length.
    back=st.istft(Z,k1=len(x))
    err=np.max(np.abs(back-x))
    # This validates the analysis/synthesis pair, including signal boundaries.
    assert err < 1e-12
    # Display only magnitudes, on an identical dB range for both panels.
    # gray_r maps stronger magnitudes to darker pixels. Never invert this image.
    im=a.pcolormesh(st.t(len(x)),st.f,db(Z),
          shading='auto',cmap='gray_r',vmin=-75,vmax=-15)
    a.set(title=f'{N} samples: {1000*N/fs:.0f} ms window',
          xlabel='Time (s)',ylabel='Frequency (Hz)',
          xlim=(0,2),ylim=(0,3500))
    fig.colorbar(im,ax=a,label='Magnitude (dB re 1)')
    # Bin spacing is fs/N because FFT length equals the window length here.
    # Zero-padding alone would make bins denser, not improve true resolving power.
    metrics[str(N)]=dict(bin_hz=fs/N,
             window_ms=1000*N/fs,reconstruction_error=err)


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
