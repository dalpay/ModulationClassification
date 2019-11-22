import numpy as np
from scipy.stats import skew, kurtosis


def moments(inp, mom_order):

    inp -= np.mean(inp)
    moms = []
    
    for p in range(2, mom_order + 2, 2):
        for q in range(0, p//2 + 1):

            moms.append(np.sum(inp**(p - q) * np.conj(inp)**q)/len(inp))

    return np.array(moms)

def cumulants(inp):

    def idx(p, q):

        return (p//2 + 1) + q - 1

    moms = moments(inp, 8)

    c20 = moms[idx(2, 0)]
    c21 = moms[idx(2, 1)]
    c40 = moms[idx(4, 0)] - 3*moms[idx(2, 0)]**2
    c41 = moms[idx(4, 0)] - 3*moms[idx(2, 0)]*moms[idx(2, 1)]
    c42 = moms[idx(4, 2)] - np.abs(moms[idx(2, 0)]) - 2*moms[idx(2, 1)]**2
    c60 = moms[idx(6, 0)] - 15*moms[idx(2, 1)]*moms[idx(4, 0)] + 30*moms[idx(2, 0)]**3
    c61 = moms[idx(6, 1)] - 5*moms[idx(2, 1)]*moms[idx(4, 0)] \
        - 10*moms[idx(2, 0)]*moms[idx(4, 1)] + 30*moms[idx(2, 1)]*moms[idx(2, 0)]**2
    c62 = moms[idx(6, 2)] - 6*moms[idx(2, 0)]*moms[idx(4, 2)] - 8*moms[idx(2, 1)]*moms[idx(4, 1)] \
        - moms[idx(2, 2)]*moms[idx(4, 0)] + 6*moms[idx(2, 2)]*moms[idx(2, 0)]**2 \
        + 24*moms[idx(2, 0)]*moms[idx(2, 1)]**2
    c63 = moms[idx(6, 3)] - 9*moms[idx(2, 1)]*moms[idx(4, 2)] + 12*moms[idx(2, 1)]**3 \
        + 3*moms[idx(2, 0)]*moms[idx(4, 3)] - 3*moms[idx(2, 2)]*moms[idx(4, 1)] \
        + 18*moms[idx(2, 0)]*moms[idx(2, 1)]*moms[idx(2, 2)]

    cums = np.array([c20, c21, c40, c41, c42, c60, c61, c62, c63], dtype=np.complex64)
    mag_and_phase = np.hstack((np.abs(cums), np.angle(cums)))
    
    return mag_and_phase

def amplitude_stats(inp):

    amp = np.abs(inp)
    amp_mean = np.mean(amp)
    amp_mean_sq = np.mean(amp**2)
    amp_std =  np.sqrt(amp_mean_sq - amp_mean**2)

    return np.array([amp_mean, amp_std])

def phase_stats(inp):

    phase = np.angle(inp)
    phase_mean = np.mean(phase)
    phase_mean_sq = np.mean(phase**2)
    phase_std =  np.sqrt(phase_mean_sq - phase_mean**2)

    return np.array([phase_mean, phase_std])

def higher_stats(inp):

    skewn = skew(inp)
    kurto = kurtosis(inp)
    stats = np.array([np.real(skewn), np.imag(skewn), 
        np.real(kurto), np.imag(kurto)])

    return stats

def iq_ratio(inp):

    real = np.sum(np.real(inp))
    imag = np.sum(np.imag(inp))
    
    if (np.allclose(real, 0.0)):
        return 0.0
    else:
        return (imag / real)
    
def peak_to_mean(inp):

    peak = np.max(np.abs(inp))
    mean = np.abs(np.mean(inp))

    if (np.allclose(mean, 0.0)):
        return 0.0
    else:
        return (peak / mean)    

def peak_to_rms(inp):

    peak = np.max(np.abs(inp)**2)
    mean = np.abs(np.mean(inp**2))

    if (np.allclose(mean, 0.0)):
        return 0.0
    else:
        return (peak / mean) 
