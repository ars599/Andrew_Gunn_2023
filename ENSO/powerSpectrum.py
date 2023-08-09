import numpy as np

def detrend(h, nord=1): 
   n = len(h)
   t = np.arange(n)
   idx = np.isfinite(h)
   p = np.polyfit(t[idx], h[idx], nord) 
   h_detrended = h - np.polyval(p, t) 
   return h_detrended

def taper(n,fr=0.1,window='hanning'): 
   """
   Window to taper fr/2 on both sides, with 'hanning' (default) or 'blackman' windows
   """

   if window.lower()=='welch':
      return quadwin(n)

   nfrh = int(n*fr/2)
   t = np.ones(n,dtype=float)
   if window.lower()=='blackman':
      wnf = np.blackman(2*nfrh)
   else:
      wnf = np.hanning(2*nfrh)
   t[0:nfrh] = t[0:nfrh]*wnf[0:nfrh]
   t[-nfrh:] = t[-nfrh:]*wnf[nfrh:]
   return t

"""
The following two functions are from QuantEcon.py package (pip install quantecon):
    https://quantecon.org/quantecon-py/
    https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/estspec.py

I changed the function name from "periodogram" to "spectrum"

Functions for working with periodograms of scalar data.
"""

def smooth(x, window_len=7, window='hanning'):
    """
    Smooth the data in x using convolution with a window of requested
    size and type.

    Parameters
    ----------
    x : array_like(float)
        A flat NumPy array containing the data to smooth
    window_len : scalar(int), optional
        An odd integer giving the length of the window.  Defaults to 7.
    window : string
        A string giving the window type. Possible values are 'flat',
        'hanning', 'hamming', 'bartlett' or 'blackman'

    Returns
    -------
    array_like(float)
        The smoothed values

    Notes
    -----
    Application of the smoothing window at the top and bottom of x is
    done by reflecting x around these points to extend it sufficiently
    in each direction.

    """
    if len(x) < window_len:
        raise ValueError("Input vector length must be >= window length.")

    if window_len < 3:
        raise ValueError("Window length must be at least 3.")

    if not window_len % 2:  # window_len is even
        window_len += 1
        print("Window length reset to {}".format(window_len))

    windows = {'hanning': np.hanning,
               'hamming': np.hamming,
               'bartlett': np.bartlett,
               'blackman': np.blackman,
               'flat': np.ones  # moving average
               }

    # === Reflect x around x[0] and x[-1] prior to convolution === #
    k = int(window_len / 2)
    xb = x[:k]   # First k elements
    xt = x[-k:]  # Last k elements
    s = np.concatenate((xb[::-1], x, xt[::-1]))

    # === Select window values === #
    if window in windows.keys():
        w = windows[window](window_len)
    else:
        msg = "Unrecognized window type '{}'".format(window)
        print(msg + " Defaulting to hanning")
        w = windows['hanning'](window_len)

    return np.convolve(w / w.sum(), s, mode='valid')


def spectrum(x0, window=None, window_len=11):
#def periodogram(x, window=None, window_len=7):
    r"""
    Computes the periodogram

    .. math::

        I(w) = \frac{1}{n} \Big[ \sum_{t=0}^{n-1} x_t e^{itw} \Big] ^2

    at the Fourier frequences :math:`w_j := \frac{2 \pi j}{n}`,
    :math:`j = 0, \dots, n - 1`, using the fast Fourier transform. Only the
    frequences :math:`w_j` in :math:`[0, \pi]` and corresponding values
    :math:`I(w_j)` are returned. If a window type is given then smoothing
    is performed.

    Parameters
    ----------
    x : 1-d array_like(float)
        A flat NumPy array containing the data to analyse
    window : string
        A string giving the window type. Possible values are 'flat',
        'hanning', 'hamming', 'bartlett' or 'blackman'
    window_len : scalar(int), optional(default=11)
        An odd integer giving the length of the window.  Defaults to 11.

    Returns
    -------
    w : array_like(float)
        Fourier frequences at which spectrum is evaluated
    I_w : array_like(float)
        Values of spectrum at the Fourier frequences

    """
    import pandas as pd

    n = len(x0)
    varx0 = x0.var()
    winweights = taper(n,0.1)
    x = x0 * winweights
    I_w = np.abs(np.fft.fft(x))**2 / n
    #w = 2 * np.pi * np.arange(n) / n  # Fourier frequencies
    w = np.arange(n) / n  # Fourier frequencies (I changed to linear freqs)
    w, I_w = w[:int(n/2)+1], I_w[:int(n/2)+1]  # Take only values on [0, pi]
    if window:
        I_w = smooth(I_w, window_len=window_len, window=window)
    spec_vals = 2*I_w*varx0/x.var()       # multiplying by 2 keeps the observed variance
    if isinstance(x,pd.Series):
        return pd.Series(spec_vals,index=w)
    else:
        return w, spec_vals

"""
A set of functions that compute power spectrum adding different steps one by one.
"""
def spectrum1(h, dt=1): 
   """
   First cut at spectral estimation: very crude.
   Returns frequencies, power spectrum, and
   power spectral density.
   Only positive frequencies between (and not including
   the Nyquist) are output.
   """
   nt = len(h)
   npositive = nt//2
   pslice = slice(1, npositive)
   freqs = np.fft.fftfreq(nt, d=dt)[pslice]
   ft = np.fft.fft(h)[pslice]
   psraw = np.abs(ft) ** 2
   # Double to account for the energy in the negative frequencies. 
   psraw *= 2
   # Normalization for Power Spectrum
   psraw /= nt**2
   # Convert PS to Power Spectral Density
   psdraw = psraw*dt*nt # nt*dt is record length
   return freqs, psraw, psdraw

def spectrum2(h, dt=1, nsmooth=11): 
   """
   Add simple boxcar smoothing to the raw periodogram.
   Chop off the ends to avoid end effects.
   """
 
   freqs, ps, psd = spectrum1(h, dt=dt)
   weights = np.ones(nsmooth, dtype=float) / nsmooth 

   nh = nsmooth//2
   xb = ps[:nh]   # First k elements
   xt = ps[-nh:]  # Last k elements
   ps = np.concatenate((xb[::-1], ps, xt[::-1])) # reflective boundaries
   yb = psd[:nh]   # First k elements
   yt = psd[-nh:]  # Last k elements
   psd = np.concatenate((yb[::-1], psd, yt[::-1]))

   cmode = 'valid'
   ps_s = np.convolve(ps, weights, mode=cmode)
   psd_s = np.convolve(psd, weights, mode=cmode)
   #nh = nsmooth//2
   #ps1 = np.full(ps.shape, np.nan)  # no longer needed, due to applying reflective BC
   #psd1 = np.full(psd.shape, np.nan)
   #ps1[nh:-nh] = ps_s
   #psd1[nh:-nh] = psd_s
   return freqs, ps_s, psd_s

def specx (h, dt=1, dtrend=1, nsmooth=11): 
   """
   This should give the same result as NCL's specx_anal function. Apply a
   tapering window and normalise as in specx_anal.

   Ref:
       https://currents.soest.hawaii.edu/ocn_data_analysis/_static/Spectrum.html 
   """

   nt = len(h)
   if dtrend in [1,2]:
      h_detrended = detrend(h,dtrend)
   else:
      h_detrended = h
   winweights = taper(nt,0.1)
   h_win = h_detrended * winweights
   varh = h_detrended.var()

   freqs, ps, psd = spectrum2(h_win, dt=dt, nsmooth=nsmooth)
   # Compensate for the energy suppressed by the window.
   psd *= nt / (winweights**2).sum()  # original normalisation as above
   ps[0] = 0.5*ps[0]
   #ps[-1] = 0.5*ps[-1]   # not needed, as ps doesn't include NyqFreq
   df = (freqs[1]-freqs[0])*dt 
   #print(len(varh*ps))
   #print(len(np.nansum(ps*df)))      
   ps = varh*ps/np.nansum(ps*df)     # NCL normalisation (roughly)

   return freqs, ps, psd

def specx_ci (ps,nsmooth,pval=0.05,window='boxcar'):
   """
   Compute the confidence interval for spectrum estimated in specx. May be plotted
   as:
   fig, ax = plt.subplots()
   ax.semilogy(freqs1, psd1, 'b', alpha=0.5)
   ax.semilogy(freqs1a, psd1a, 'r', alpha=0.5)

   ax.plot([conf_x, conf_x], conf, color='k', lw=1.5)
   ax.plot(conf_x, conf_y0, color='k', linestyle='none', 
        marker='_', ms=8, mew=2)

   Ref:
       https://currents.soest.hawaii.edu/ocn_data_analysis/_static/Spectrum.html
   Ref for DOFs:
       http://pordlabs.ucsd.edu/sgille/sioc221a_f17/lecture16_notes.pdf 
   """

   import scipy.stats as ss

   df = 2*nsmooth # DOF for nsmooth-point boxcar smoother
   if window == 'hanning':
       df = df*8/3    # DOF Hanning window (Ref: Table 1 of the 2nd Ref above)
   elif window == 'hamming':
       df = df*2.5164 # DOF Hamming window 
   elif window == 'bartlett':
       df = df*3.0
   elif window == 'parzen':
       df = df*3.708614

   ci = [pval/2,1-pval/2]
   conf = ps[:,np.newaxis] * df / ss.chi2.ppf(ci, df)

   return conf


