import numpy as np

def coil_properties_secondario(e_n_nV, i_n_pA, raggio_medio_mm, lunghezza_mm, altezza_mm, awg, iso, tipo_collegamento):
    rho = 1.724e-8

    raggio_medio_m = raggio_medio_mm * 1e-3

    # AWG → diametro
    d_mm = 0.127 * (92 ** ((36.0 - awg) / 39.0))
    d_iso_mm = d_mm * iso
    rmi_mm = d_iso_mm / 2
    d_m = d_mm * 1e-3
    Sm = np.pi * (d_m / 2)**2

    # Resistenza per spira
    R_spira = 2 * np.pi * raggio_medio_m * rho / Sm

    # Numero spire
    N_per_layer = int(lunghezza_mm / (2 * rmi_mm))
    N_layers = int(4/(2*np.sqrt(3))*altezza_mm / (2 * rmi_mm))
    N_cont = int(N_per_layer * N_layers)
    R_tot = N_cont * R_spira

    # Resistenza di riferimento da e_n, i_n
    R_rif = (e_n_nV / i_n_pA) * 1e3  # Ω

    if tipo_collegamento == "S":
        R_eff = R_rif / 2
    elif tipo_collegamento == "P":
        R_eff = 2 * R_rif
    else:
        R_eff = R_rif

    r1_m = (raggio_medio_mm - altezza_mm/2) * 1e-3

    return {
        "awg": awg,
        "d_mm": d_mm,
        "d_iso_mm": d_iso_mm,
        "N_cont": N_cont,
        "N_per_layer": N_per_layer,
        "N_layers": N_layers,
        "R_spira": R_spira,
        "R_tot": R_tot,
        "R_rif": R_rif,
        "R_eff": R_eff,
        "r1_m": r1_m
    }

def dimensionamento_secondario(f0_kHz, e_n_nV, i_n_pA, raggio_medio_mm, lunghezza_mm, altezza_mm, awg, iso, tipo_collegamento):
    
    ref = coil_properties_secondario(e_n_nV, i_n_pA, raggio_medio_mm, lunghezza_mm, altezza_mm, awg, iso, tipo_collegamento)
    if ref is None:
        return {"error": "AWG troppo grande: nessuna spira avvolgibile."}
    
    #induttanza e capacità
    N_cont = ref["N_cont"]
    r1_m = ref["r1_m"]

    L_uH = (31.6 * N_cont**2 * r1_m**2) / (6*r1_m + 9*lunghezza_mm*1e-3 + 10*altezza_mm*1e-3)
    L_mH = L_uH * 1e-3
    #L_H = L_uH * 1e-6
    omega0 = 2*np.pi*f0_kHz*1e3
    C_nF = 1/(L_mH * omega0**2)*1e12

    #ottimizzazione awg
    best = None
    min_diff = float("inf")

    for awg_test in range(1, 41):
        c = coil_properties_secondario(e_n_nV, i_n_pA, raggio_medio_mm, lunghezza_mm, altezza_mm, awg_test, iso, tipo_collegamento)
        if c is None:
            continue

        diff = abs(c["R_eff"] - c["R_tot"])
        if diff < min_diff:
            min_diff = diff
            best = c
    
    #induttanza ottimale 
    N_best = best["N_cont"]
    r1_best = best["r1_m"]

    L_opt_uH = (31.6 * N_best**2 * r1_best**2) / (
        6*r1_best + 9*lunghezza_mm*1e-3 + 10*altezza_mm*1e-3
    )
    L_opt_mH = L_opt_uH * 1e-3
    C_opt_nF = 1/(L_opt_mH * omega0**2)*1e12

    return {
        "ref": ref,
        "L_mH": L_mH,
        "C_nF": C_nF,
        "best": best,
        "best_awg": best["awg"],
        "L_opt_mH": L_opt_mH,
        "C_opt_nF": C_opt_nF
    }
