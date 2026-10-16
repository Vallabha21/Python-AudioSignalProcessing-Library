# Regularized deconvolution
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

# 1. N is probe length; M is the known IR support, both in samples.
fs=16000; N=8192; M=257
rng=np.random.default_rng(10)
# A low-pass probe deliberately leaves weak high frequencies.
# Low-pass filtering makes high frequencies weakly excited and hard to identify.
x=signal.lfilter(signal.firwin(129,2500,fs=fs),[1],
                 rng.normal(size=N))
# Known direct path and two reflections provide ground truth for this teaching test.
h=np.zeros(M); h[0]=1; h[48]=.55; h[160]=-.25
# Generate the ideal system response before adding measurement noise.
clean=signal.fftconvolve(x,h)
noise=rng.normal(size=len(clean))
# Use noise standard deviation 0.05 for the main regularization comparison.
y=clean+.05*noise
# 2. Zero-pad beyond N+M-1=8448 samples so the model is linear convolution.
K=16384
X=np.fft.rfft(x,K); Y=np.fft.rfft(y,K)
# Small |X|^2 means the probe carries little information at that frequency.
power=abs(X)**2
# These are dimensionless ratios, not absolute regularization strengths.
lambdas=[1e-12,1e-6,1e-4,1e-2]
estimates=[]; errors=[]
for ratio in lambdas:
    # Scale regularization to the largest probe-bin power.
    lam=ratio*power.max()
    # Minimize |X*H-Y|^2 + lam*|H|^2 independently in each frequency bin.
    # The added denominator term limits noise amplification where X is weak.
    Hhat=np.conj(X)*Y/(power+lam)
    # Inverse FFT returns a periodic sequence. Retain the known first M samples.
    # Known support and time alignment are prior information in this experiment.
    hh=np.fft.irfft(Hhat,K)[:M]
    estimates.append(hh)
    # Relative L2 error includes amplitude and shape errors against the known IR.
    errors.append(float(np.linalg.norm(hh-h)/np.linalg.norm(h)))
# Require improvement for THIS noise level; lower noise may favor less regularization.
assert errors[2]<errors[0]
# Noise-free, fully excited probe checks normalization/alignment.
# 3. Broadband, noise-free excitation checks normalization and delay independently.
probe=rng.normal(size=N)
P=np.fft.rfft(probe,K)
Q=np.fft.rfft(signal.fftconvolve(probe,h),K)
# Direct division is safe for this numerical fixture, not generally for noisy probes.
exact=np.fft.irfft(Q/P,K)[:M]
assert np.max(abs(exact-h))<1e-10
# 4. Compare truth, almost-unregularized recovery, and the selected estimate.
# Axes autoscale separately: compare amplitudes using each panel's tick labels.
fig,ax=plt.subplots(3,1,figsize=(4.5,6),layout='constrained')
for a,v,title in zip(ax,[h,estimates[0],estimates[2]],
             ['Known IR','Almost unregularized',
              'Regularized: ratio 0.0001']):
    a.plot(np.arange(M)/fs*1000,v,color='black')
    a.set(title=title,xlabel='Time (ms)',ylabel='IR amplitude')
# 5. Reuse the same noise realization at multiple levels to expose the trade-off.
# At level 0.01 the selected regularization worsens error through bias.
noise_sweep={}
for level in [.01,.03,.05,.1]:
    # Only noise level changes; the clean response and excitation remain fixed.
    V=np.fft.rfft(clean+level*noise,K)
    errs=[]
    for ratio in lambdas:
        est=np.fft.irfft(X.conj()*V/
                (power+ratio*power.max()),K)[:M]
        errs.append(float(np.linalg.norm(est-h)/np.linalg.norm(h)))
    noise_sweep[str(level)]=errs
# Record the entire sweep, not just the favorable case.
# Choosing the ratio from ground truth is a teaching convenience, not a deployment method.
metrics=dict(noise_sweep=noise_sweep,
 relative_errors=dict(zip(map(str,lambdas),errors)),
 noise_free_error=float(np.max(abs(exact-h))),
 selected_ratio=1e-4)


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
