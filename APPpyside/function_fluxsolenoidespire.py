import numpy as np
from scipy.interpolate import interp1d

def flusso_secondario(Bz_map, ztot, rtot, RS, dS, LS, hS, r0, z_center, iso, step):
    """
    Calcola il flusso concatenato al secondario.

    Parametri:
    -----------
    Bz_map : 2D np.array
        Campo Bz(z,r) generato dal primario
    ztot, rtot : 1D np.array
        Assi corrispondenti a Bz_map
    RS : float
        Raggio medio del secondario
    NS : int
        Numero totale di spire del secondario
    dS : float
        Diametro del filo secondario
    LS : float
        Lunghezza del secondario
    hS : float
        Spessore radiale del secondario
    z_center : float
        Coordinata assiale del centro del secondario
    step : float
        Passo della griglia

    Ritorna:
    --------
    phi_tot : float
        Flusso totale concatenato al secondario
    """
    
    pitch = dS * iso
    N_per_layer = max(1, int(LS / pitch))
    N_layers = max(1, int(2/np.sqrt(3) *hS / pitch))
    NS = N_per_layer*N_layers
    pref = NS/(LS*hS)
    # Creiamo SOLO UNA volta l’interpolatore lungo z
    Bz_interp_z = interp1d(ztot, Bz_map, axis=0, kind='cubic', bounds_error=False, fill_value="extrapolate")

    # Preallocazione
    phi_tot = 0.0

    # Precomputiamo tutte le maschere radiali per layer (sono poche)
    RS_layers = RS - hS/2 + np.arange(N_layers)*pitch + r0
    masks_r = [rtot <= RS_layer for RS_layer in RS_layers]

    for L in range(N_layers):

        r_mask = masks_r[L]
        r_vals = rtot[r_mask]

        for n in range(N_per_layer):

            z_single = -LS/2 + n*pitch + z_center

            # Interpola solo una riga (molto veloce)
            Bz_row = Bz_interp_z(z_single)[r_mask]

            # integrazione vettoriale
            phi_tot += np.sum(2*np.pi * r_vals * Bz_row * step)
    
    return phi_tot

    '''# --- Caso spira ideale ---
    if LS == 0 and hS == 0:
        # integrazione radiale fino a RS
        idx = np.searchsorted(rtot, RS, side='right') - 1
        integrand = 2 * np.pi * rtot[:idx+1] * Bz_map[:, :idx+1] * step  # shape: (len(ztot), len(rtot <= RS))
        phi_z = np.sum(integrand, axis=1)  # flusso cumulativo lungo r per ogni z
        # interpolazione per la posizione centrale z_center
        phi_interp = interp1d(ztot, phi_z, kind='cubic', fill_value="extrapolate")
        return float(phi_interp(z_center))

    # --- Caso bobina reale ---
    phi_tot = 0.0
    N_strati = max(1, int(np.floor(hS / (1.2*dS))))  # numero di strati radiali
    pitch = dS*1.2                   # passo assiale tra le spire
    RS_inner = RS - hS/2                        # raggio più interno

    for s in range(N_strati):
        R_layer = RS_inner + s*1.2*dS
        z_start = z_center - 0.5*LS
        for n in range(NS):
            z_i = z_start + n*pitch
            # interpolazione Bz in (z_i, R_layer)
            # prima interpoliamo lungo r per la z_i più vicina
            idx_z = np.searchsorted(ztot, z_i)
            idx_z = np.clip(idx_z, 0, len(ztot)-1)
            Bz_row = Bz_map[idx_z, :]
            Bz_val = np.interp(R_layer, rtot, Bz_row)
            phi_tot += Bz_val * np.pi * R_layer**2  # area spira
    return phi_tot'''
