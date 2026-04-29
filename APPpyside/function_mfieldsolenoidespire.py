import numpy as np
from function_mfieldspira_3 import m_field

def Bz_solenoide(I, RP, dP, LP, hP, z0, r0, ztot, rtot, iso, step):
    """
    Calcola Bz del solenoide usando le griglie ztot e rtot già definite.
    """

    Bz_tot = np.zeros((len(ztot), len(rtot)))
    pitch = dP * iso

    # Numero stimato di spire per layer
    N_per_layer = max(1, int(np.round(LP / pitch)))
    
    # Numero di layer radiali
    N_layers = max(1, int(np.round((4/(2*np.sqrt(3)) * hP / pitch))))
    
    NP = N_layers*N_per_layer

    RP_layer = RP + (np.arange(N_layers) - (N_layers - 1)/2) * pitch + r0
    z = (np.arange(N_per_layer) - (N_per_layer - 1)/2) * pitch + z0

    for i in range(N_layers):
        # Raggio del layer L
        RP_layer_single = RP_layer[i]
        for j in range(N_per_layer):
            # posizione assiale della spira n
            z_single = z[j]
            Bz_tot += (NP / (LP * hP)) * m_field(I, RP_layer_single, z_single, r0, ztot, rtot) *pitch**2
    return Bz_tot


    
    
    
    
    
    
    
    
    
    
    
    """ Bz_tot = np.zeros((len(ztot), len(rtot)))

    N_strati = max(1, int(np.floor(hP / (1.2*dP))))
    RP_inner = RP - hP/2

    # Numero di spire per strato (distribuzione uniforme)
    spire_per_strato = NP // N_strati
    spire_extra = NP % N_strati  # eventuali spire rimanenti

    spire_coords = []
    for s in range(N_strati):
        RP_layer = RP_inner + s * 1.2*dP
        z_start = z0 - 0.5 * LP
        # Se ci sono spire extra, aggiungine una in questo strato
        n_spire_in_layer = spire_per_strato + (1 if s < spire_extra else 0)

        if n_spire_in_layer <=1:
            pitch = 0
        else:
            pitch = dP*1.2
        for n in range(n_spire_in_layer):
            z_pos = z_start + n * pitch
            spire_coords.append((RP_layer, z_pos, r0, z0))

    for RP_loop, z_loop, r0_loop, z0_loop in spire_coords:
        Bz = m_field(I, RP_loop, z_loop, r0_loop, ztot, rtot, ztot, rtot, step)
        Bz_tot += Bz

    return Bz_tot """
