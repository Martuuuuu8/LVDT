# --- B_spira_versione3.py ---
import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def m_field(I, RP, z0, r0, ztot, rtot):
    """ 
    Calcola il campo magnetico Bz generato da una singola spira di raggio RP.
    Versione accelerata con Numba (parallelizzata e stabile numericamente).
    """
    mu0 = 4 * np.pi * 1e-7

    Bz = np.zeros((len(ztot), len(rtot)))

    # Precalcolo di gamma
    Ngamma = 1000
    gamma = np.linspace(0.0, 2.0 * np.pi, Ngamma)
    cos_gamma = np.cos(gamma)
    dgamma = gamma[1] - gamma[0]

    eps = 1e-20  # stabilità numerica

    # Ciclo doppio parallelo (ztot × rtot)
    for i in prange(len(ztot)):
        z = ztot[i]
        for j in range(len(rtot)):
            r = rtot[j] -r0
            integral = 0.0

            for k in range(Ngamma):
                num = RP - r * cos_gamma[k]
                den = (r**2 + RP**2 - 2.0 * RP * r * cos_gamma[k] + (z-z0)**2)
                if den < 0.0:
                    den = 0.0
                den = den**1.5 + eps
                integral += num / den

            Bz[i, j] = mu0 / (4.0 * np.pi) * I * RP * integral * dgamma

    '''# Interpolazione semplice (facoltativa)
    for i in range(1, len(ztot) - 1):
        for j in range(1, len(rtot) - 1):
            if Bz[i, j] == 0.0:
                Bz[i, j] = 0.25 * (Bz[i - 1, j] + Bz[i + 1, j] + Bz[i, j - 1] + Bz[i, j + 1])'''

    return Bz

