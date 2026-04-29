import numpy as np
import math

rho = 1.724e-8  # resistività rame [Ω·m]

def coil_properties(awg, raggio_medio_mm, lunghezza_mm, altezza_mm,
                    V0, Imax, Pmax, Rcabling, iso):
    """
    Versione IDENTICA al codice originale.
    Calcola tutte le grandezze per un dato AWG.
    """

    # Diametri
    d_mm = 0.127 * (92 ** ((36.0 - awg) / 39.0))
    d_iso_mm = d_mm * iso
    rmi_mm = d_iso_mm / 2
    d_m = d_mm * 1e-3 
    A_filo = np.pi * (d_m / 2)**2

    # Numero spire
    N_per_layer = int(lunghezza_mm / (2 * rmi_mm))
    N_layers = int(4/(2*np.sqrt(3)) * altezza_mm / (2 * rmi_mm))
    N_cont = int(N_per_layer * N_layers)

    if N_cont < 1:
        return None

    # Raggi
    r_int_m = (raggio_medio_mm - altezza_mm/2) * 1e-3
    r_ext_m = (raggio_medio_mm + altezza_mm/2) * 1e-3
    r_eff_m = (r_int_m + r_ext_m) / 2

    # Resistenza per spira (ATTENZIONE: usa raggio_medio_mm come nel codice originale)
    R_spira = rho * 2 * np.pi * (raggio_medio_mm * 1e-3) / A_filo
    R_tot = N_cont * R_spira

    # Resistenza target e N_ideal (calcolati QUI come nel codice originale)
    R_Imax = (V0 / Imax) - Rcabling
    R_Pmax = 2 * Pmax / (Imax**2)
    R_target = max(R_Imax, R_Pmax)
    N_ideal = R_target / R_spira

    return {
        "awg": awg,
        "d_mm": d_mm,
        "d_iso_mm": d_iso_mm,
        "R_spira": R_spira,
        "R_tot": R_tot,
        "N_cont": N_cont,
        "R_target": R_target,
        "N_ideal": N_ideal,
        "N_layers": N_layers,
        "N_per_layer": N_per_layer,
        "r_int_m": r_int_m,
        "r_ext_m": r_ext_m,
        "r_eff_m": r_eff_m
    }


def dimensionamento_primario(V0, Imax, Pmax, Rcabling,
                             raggio_medio_mm, lunghezza_mm, altezza_mm,
                             dP_AWG, f0_kHz, iso):
    """
    Versione IDENTICA al codice originale.
    """

    # Configurazione iniziale
    ref = coil_properties(dP_AWG, raggio_medio_mm, lunghezza_mm, altezza_mm,
                          V0, Imax, Pmax, Rcabling, iso)
    if ref is None:
        return {"error": "AWG troppo grande: nessuna spira avvolgibile."}

    # Induttanza e capacità (identiche all’originale)
    L_uH = (31.6 * ref["N_cont"]**2 * ref["r_int_m"]**2) / (
        6*ref["r_int_m"] + 9*lunghezza_mm*1e-3 + 10*altezza_mm*1e-3
    )
    L_H = L_uH * 1e-6
    C_F = 1 / (L_H * (2*np.pi*f0_kHz*1e3)**2)
    C_nF = C_F * 1e9

    # Ottimizzazione AWG (identica all’originale)
    best = None
    min_diff = float("inf")

    for awg in range(1, 41):
        c = coil_properties(awg, raggio_medio_mm, lunghezza_mm, altezza_mm,
                            V0, Imax, Pmax, Rcabling, iso)
        if c is None:
            continue
        diff = abs(c["N_ideal"] - c["N_cont"])
        if diff < min_diff:
            min_diff = diff
            best = c

    # Induttanza ottimale (identica all’originale)
    L_opt_uH = (31.6 * best["N_cont"]**2 * best["r_int_m"]**2) / (
        6*best["r_int_m"] + 9*lunghezza_mm*1e-3 + 10*altezza_mm*1e-3
    )
    L_opt_H = L_opt_uH * 1e-6
    C_opt_F = 1 / (L_opt_H * (2*np.pi*f0_kHz*1e3)**2)
    C_opt_nF = C_opt_F * 1e9

    return {
        "ref": ref,
        "L_uH": L_uH,
        "C_nF": C_nF,
        "best": best,
        "best_awg": best["awg"],
        "L_opt_uH": L_opt_uH,
        "C_opt_nF": C_opt_nF
    }
