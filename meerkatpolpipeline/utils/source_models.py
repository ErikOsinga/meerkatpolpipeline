from __future__ import annotations

import astropy.units as u  # noqa: F401 TODO: add unit handling
import numpy as np
from astropy.constants import c


def stokesI_model_3c286(freq: np.ndarray) -> np.ndarray:
    """
    Model for 3C286

    From Perley, Butler 2017 
    https://iopscience.iop.org/article/10.3847/1538-4365/aa6df9

    args:
        freq: frequency in Hz
    returns:
        flux in Jy as function of freq
    """

    Sa0 = 1.2480
    Sa1 = -0.4507
    Sa2 = -0.1798
    Sa3 = 0.0357
    return 10**(Sa0 + Sa1*np.log10(freq*1e-9) + Sa2*(np.log10(freq*1e-9))**2 + Sa3*(np.log10(freq*1e-9))**3)


def EVPA_model_3c286(frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    HP24 piecewise EVPA model for 3C286 (degrees) as a function of freq (Hz).

    See B. Hugo & R. Perley + 2024
    https://archive-gw-1.kat.ac.za/public/repository/10.48479/bqk7-aw53/data/Absolute_linear_polarization_angle_calibration_using_planetary_bodies_for_MeerKAT_and_JVLA-REV-C.pdf

    args:
        freq: frequency in Hz
    returns:
        lambdasq: wavelength squared in m^2 as function of freq
        EVPA:     evpa in degrees as function of freq
    """
    nu_GHz = frequencies * 1e-9
    lambdasq = (c.value / frequencies)**2
    evpa = np.zeros_like(lambdasq)

    # 1.7 GHz <= ν <= 12 GHz
    m1 = (nu_GHz >= 1.7) & (nu_GHz <= 12)
    evpa[m1] = 32.64 - 85.37 * lambdasq[m1]

    # ν < 1.7 GHz
    m2 = nu_GHz < 1.7
    evpa[m2] = 29.53 + lambdasq[m2] * (4005.88 * np.log10(nu_GHz[m2])**3 - 39.38)

    return lambdasq, evpa


def polfrac_model_3c286(frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HP24 piecewise polarisation fraction model for 3C286 as function of freq (Hz).
    
    See B. Hugo & R. Perley + 2024
    https://archive-gw-1.kat.ac.za/public/repository/10.48479/bqk7-aw53/data/Absolute_linear_polarization_angle_calibration_using_planetary_bodies_for_MeerKAT_and_JVLA-REV-C.pdf

    args:
        freq: frequency in Hz
    returns:
        lambdasq: wavelength squared in m^2 as function of freq
        polfrac : polfrac as function of freq
    """
    nu_GHz = frequencies * 1e-9
    lambdasq = (c.value / frequencies)**2
    polfrac = np.zeros_like(lambdasq)

    # 1.1 GHz <= ν <= 12 GHz
    m1 = (nu_GHz >= 1.1) & (nu_GHz <= 12)
    polfrac[m1] = 0.080 - 0.053 * lambdasq[m1] - 0.015 * np.log10(lambdasq[m1])

    # ν < 1.1 GHz
    m2 = nu_GHz < 1.1
    polfrac[m2] = 0.029 - 0.172 * lambdasq[m2] - 0.067 * np.log10(lambdasq[m2])

    return lambdasq, polfrac


def stokesI_model_3c138(freq: np.ndarray) -> np.ndarray:
    """
    Model for 3C138

    From Perley, Butler 2017 
    https://iopscience.iop.org/article/10.3847/1538-4365/aa6df9

    args:
        freq: frequency in Hz
    returns:
        flux in Jy as function of freq
    """
    Sa0 = 1.0088
    Sa1 = -0.4981
    Sa2 = -0.1552
    Sa3 = -0.0102
    Sa4 = -0.0223

    return 10**(Sa0 + Sa1*np.log10(freq*1e-9) + Sa2*(np.log10(freq*1e-9))**2 + Sa3*(np.log10(freq*1e-9))**3 + Sa4*(np.log10(freq*1e-9))**4 )

def EVPA_model_3c138(freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    3C138 EVPA model (degrees) as a function of freq (Hz),

    From Perley, Butler 2017 table at 1.05-2.45 GHz.
    https://iopscience.iop.org/article/10.3847/1538-4365/aa6df9
    Determined from a fit to chi vs lambdasq: chi(lambdasq) = chi0 + rm_deg * lambdasq
    
    args:
        freq: frequency in Hz
    returns:
        lambdasq: wavelength squared in m^2 as function of freq
        EVPA:     evpa in degrees as function of freq
    """
    lambdasq = (c.value / freq)**2
    # linear fit: slope in deg per m^2, intercept in deg
    rm_deg = -73.82118   # ≃ Delta chi / Delta lambdasq
    chi0   = -7.90190    # chi at lambdasq=0
    evpa_deg = chi0 + rm_deg * lambdasq
    return lambdasq, evpa_deg


def polfrac_model_3c138(freq):
    """
    3C138 polarisation-fraction model (dimensionless) vs freq (Hz),
    
    From Perley, Butler 2017 table at 1.05-2.45 GHz.
    https://iopscience.iop.org/article/10.3847/1538-4365/aa6df9
    from fit to polfrac vs lambdasq: polfrac = b_frac + m_frac * lambdasq
    
    args:
        freq: frequency in Hz
    returns:
        lambdasq: wavelength squared in m^2 as function of freq
        polfrac : polfrac as function of freq

    """
    lambdasq = (c.value / freq)**2
    # linear fit to percent–lambdasq
    m_frac = -67.29352    # % per m^2
    b_frac =  10.82183    # % at lambdasq=0
    polfrac = (m_frac * lambdasq + b_frac) / 100.0

    # convert to polarised intensity (Jy)
    # P_int = stokesI_model_3c138(freq) * polfrac # not returned atm

    return lambdasq, polfrac