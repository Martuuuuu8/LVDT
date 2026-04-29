import os
import time
import numpy as np
from joblib import Parallel, delayed
import gc
import copy
from tqdm import tqdm
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QFormLayout, QDoubleSpinBox, QSpinBox, QApplication, QSizePolicy,
    QDialog, QScrollArea, QGridLayout, QProgressBar, QFileDialog
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFormLayout, QDoubleSpinBox, QSpinBox, QScrollArea,
    QGridLayout, QFileDialog, QComboBox
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QWidget as QW
from PySide6.QtWidgets import QTabWidget

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d

from function_mfieldsolenoidespire import Bz_solenoide
from function_fluxsolenoidespire import flusso_secondario
from dimensionamentoelettricoprimario import dimensionamento_primario
from dimensionamentoelettricosecondario import dimensionamento_secondario

def cylinder_surface(R, z_min, z_max, n_theta=200, n_z=64):
    theta = np.linspace(0, 2*np.pi, n_theta)
    z = np.linspace(z_min, z_max, n_z)
    theta, z = np.meshgrid(theta, z)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    return x, y, z

# ---------------------------------------------------------
# PARALLEL HELPER CON PROGRESSO (NO SIGNAL NELLE TASK)
# ---------------------------------------------------------
class TqdmParallel(Parallel):
    def __call__(self, *args, **kwargs):
        with tqdm(total=len(args[0])) as self._pbar:
            return super().__call__(*args, **kwargs)

    def print_progress(self):
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
# ---------------------------------------------------------
# COMPUTE FIELD (PARALLELO + PROGRESSO)
# ---------------------------------------------------------
def compute_field(RMAX, LMAX, dmax, resolution, dP_AWG, dS_AWG, I, iso,
                  progress_signal, message_signal):

    # --- pulizia ---
    for name in ['Br_all', 'Bz_all', 'flux', 'dflux_dz', 'dflux_max',
                 'z_max', 'RPtot', 'RStot', 'rtot', 'ztot']:
        if name in globals():
            del globals()[name]
    gc.collect()

    # --- parametri ---
    dP = 0.127 * 92**((36 - dP_AWG)/39) * 1e-3
    dS = 0.127 * 92**((36 - dS_AWG)/39) * 1e-3
    step = resolution

    # --- generazione combinazioni ---
    RPtot = np.arange(resolution*20, RMAX + resolution, resolution*10)
    RStot = np.arange(resolution*20, RMAX + resolution, resolution*10)
    LPtot = np.arange(resolution*20, LMAX + resolution, resolution*10)
    LStot = np.arange(resolution*20, LMAX + resolution, resolution*10)

    real_comb = []

    for RP in RPtot:
        for RS in RStot:
            for LP in LPtot:
                for LS in LStot:
                    spaz = (LMAX/2 - LS)*2
                    if spaz <= 2*LP: continue
                    if (spaz/2 - LP/2) <= dmax: continue
                    hS = 2*(RMAX - RS)
                    if hS < 4*dS*iso:
                        continue
                    if hS > 1.1*LS:
                        hS = 1.1*LS
                    hP = 2*(RMAX - RP)
                    if hP < 4*dP*iso:
                        continue
                    if hP > 1.1*LP:
                        hP = 1.1*LP
                    if not (np.isclose(RS + hS/2, RMAX) or np.isclose(RP + hP/2, RMAX)):
                        continue
                    rtot = np.arange(0, RMAX + resolution, resolution)
                    zmin = -(spaz + LS - LP/2)
                    zmax = -zmin
                    ztot = np.arange(zmin, zmax + resolution, resolution)
                    real_comb.append((RP, RS, LP, LS, hP, hS, spaz, rtot, ztot))
    message_signal.emit(f"Valid combinations: {len(real_comb)}")
    print("Valid combinations:", len(real_comb))

    if len(real_comb) == 0:
        return [], []

    # --- estrai triple uniche (RP, LP, hP) ---
    RP_LP_hP_set = []
    seen = set()
    for RP, RS, LP, LS, hP, hS, spaz, rtot, ztot in real_comb:
        key = (RP, LP, hP)
        if key not in seen:
            seen.add(key)
            RP_LP_hP_set.append(key)

    # --- calcolo Bz seriale con progress ---
    RP_LP_hP_Bz_set = []
    n_tot = len(RP_LP_hP_set)

    def compute_Bz(i):
        message_signal.emit(f"Computing field {i+1}/{n_tot}")
        progress_signal.emit(int(100 * (i+1) / n_tot))
        RP, LP, hP = RP_LP_hP_set[i]
        steplargo = resolution*5
        ztot = np.arange(0, LMAX - LP/2 + resolution, steplargo)
        rtot = np.arange(0, RMAX + resolution, steplargo)
        Bz = Bz_solenoide(I, RP, dP, LP, hP, 0, 0, ztot, rtot, iso, step)
        Bz_smooth = np.zeros_like(Bz)
        for j in range(len(rtot)):
            Bz_smooth[:, j] = gaussian_filter1d(Bz[:, j], sigma=5)
        return Bz_smooth

    tasks = [delayed(compute_Bz)(i) for i in range(len(RP_LP_hP_set))]
    Bz_tot = TqdmParallel(n_jobs=-1, prefer="threads")(tasks)
    
    RP_LP_hP_Bz_set = []
    for i in range(len(RP_LP_hP_set)):
        RP, LP, hP = RP_LP_hP_set[i]
        RP_LP_hP_Bz_set.append([RP, LP, hP, Bz_tot[i]])

    #print("!!!")
    # --- lookup ---
    Bz_lookup = {(c[0], c[1], c[2]): c[3] for c in RP_LP_hP_Bz_set}

    # --- ricostruisci tutte le combinazioni ---
    RPRS_LPLS_hPhS_spaz_Bz_set = []
    for c in real_comb:
        RP, RS, LP, LS, hP, hS, spaz, rtot, ztot = c
        Bz = Bz_lookup[(RP, LP, hP)]
        RPRS_LPLS_hPhS_spaz_Bz_set.append([RP, RS, LP, LS, hP, hS, spaz, Bz])
    #print("campo fatto")
    return RP_LP_hP_Bz_set, RPRS_LPLS_hPhS_spaz_Bz_set

def compute_flux_and_filter(RPRS_LPLS_hPhS_spaz_Bz_set,
                            RMAX, LMAX, dmax, resolution, dP_AWG, dS_AWG, I, iso, f0_kHz,
                            progress_signal, message_signal):

    dP = 0.127 * 92**((36 - dP_AWG)/39) * 1e-3
    dS = 0.127 * 92**((36 - dS_AWG)/39) * 1e-3

    step = resolution
    r0 = 0

    n_tot = len(RPRS_LPLS_hPhS_spaz_Bz_set)

    def compute_flux(i):
        message_signal.emit(f"Computing flux {i+1}/{n_tot}")
        progress_signal.emit(int(100 * (i+1) / n_tot))
        RP, RS, LP, LS, hP, hS, spaz, Bz = RPRS_LPLS_hPhS_spaz_Bz_set[i]
        steplargo = resolution*5

        ztot = np.arange(0, LMAX - LP/2 + resolution, steplargo)
        rtot = np.arange(0, RMAX + resolution, steplargo)

        dzmax = (spaz/2 - LP/2)
        dz_shift = np.arange(-dzmax, dzmax + step, steplargo)
        zS1_center = spaz/2 + LS/2

        flux_S1_array = np.zeros_like(dz_shift)
        for j, dzs in enumerate(dz_shift):
            zS1 = zS1_center + dzs
            flux_S1_array[j] = flusso_secondario(
                Bz, ztot, rtot, RS, dS, LS, hS, r0, zS1, iso, step
            )
        return [flux_S1_array, zS1_center, dz_shift]

    tasks = [delayed(compute_flux)(i) for i in range(len(RPRS_LPLS_hPhS_spaz_Bz_set))]
    flux_results = TqdmParallel(n_jobs=-1, prefer="threads")(tasks)

    RPRS_LPLS_hPhS_spaz_Bz_flux_set = []
    for i in range(len(RPRS_LPLS_hPhS_spaz_Bz_set)):
        RP, RS, LP, LS, hP, hS, spaz, Bz = RPRS_LPLS_hPhS_spaz_Bz_set[i]
        flux_S1_array, zS1_center, dz_shift = flux_results[i]
        RPRS_LPLS_hPhS_spaz_Bz_flux_set.append(
            [RP, RS, LP, LS, hP, hS, spaz, Bz, flux_S1_array, zS1_center, dz_shift]
        )

    RPRS_filtered = []
    #toll = resolution * 10

    for c in RPRS_LPLS_hPhS_spaz_Bz_flux_set:
        RP, RS, LP, LS, hP, hS, spaz, Bz, flux_S1_array, zS1_center, dz_shift = c

        dflux1 = np.gradient(flux_S1_array, step)
        dflux1_abs = np.abs(dflux1)

        z_axis = zS1_center + dz_shift
        idxxx = np.argmin(np.abs(z_axis - zS1_center))
        dfluxsens = dflux1_abs[idxxx]

        efmnorm = -2*np.pi*f0_kHz*(flux_S1_array - np.flip(flux_S1_array))
        # 1) Ricerca automatica del punto più lineare tramite finestra mobile
        window = 5   # punti per il fit locale
        errors = []

        for k in range(window, len(dz_shift)-window):
            dz_loc = dz_shift[k-window:k+window]
            ef_loc = efmnorm[k-window:k+window]
            m_loc, q_loc = np.polyfit(dz_loc, ef_loc, 1)
            y_fit = m_loc * dz_loc + q_loc
            err = np.mean(np.abs(ef_loc - y_fit))
            errors.append(err)

        k_best = np.argmin(errors) + window
        dz0 = dz_shift[k_best]   # punto di massima linearità

        # 2) Fit locale nel punto migliore
        dz_loc = dz_shift[k_best-window:k_best+window]
        ef_loc = efmnorm[k_best-window:k_best+window]
        m_loc, q_loc = np.polyfit(dz_loc, ef_loc, 1)

        # 3) Soglia basata sulla pendenza (molto più fisica)
        eps = 0.015 * abs(m_loc)* np.max(np.abs(dz_shift))  # 2% della variazione attesa

        # 4) Espansione simmetrica finché i residui restano sotto soglia
        i_left = k_best
        i_right = k_best

        while i_left > 0:
            y_pred = m_loc * dz_shift[i_left] + q_loc
            if abs(efmnorm[i_left] - y_pred) < eps:
                i_left -= 1
            else:
                break

        while i_right < len(dz_shift)-1:
            y_pred = m_loc * dz_shift[i_right] + q_loc
            if abs(efmnorm[i_right] - y_pred) < eps:
                i_right += 1
            else:
                break
        
        dz_lin = dz_shift[i_left:i_right+1]

        if len(dz_lin) >= 2:
            dz_min = dz_shift[i_left]
            dz_max = dz_shift[i_right]
            lin = min(abs(dz_min - dz0), abs(dz_max - dz0))  # semi-intervallo simmetrico
            mask_fit = (dz_shift >= -lin) & (dz_shift <= lin)
            x_fit = dz_shift[mask_fit]
            y_fit = efmnorm[mask_fit]
            m, q = np.polyfit(x_fit, y_fit, 1)
            fit_ok = True
        else:
            lin = 0.0
            m, q = np.nan, np.nan
            fit_ok = False

        if lin > dmax + resolution:
            RPRS_filtered.append((RP, RS, LP, LS, hP, hS, spaz, Bz, flux_S1_array, zS1_center, dz_shift, dfluxsens, lin))


    RPRS_filtered_sortedbysens = sorted(
        RPRS_filtered,
        key=lambda c: c[11],
        reverse=True
    )
    
    RPRS_filtered_sortedbylin = sorted(
        RPRS_filtered,
        key=lambda c: c[12],
        reverse=True
    )

    filtered_real_comb_fff_sortedbysens = []
    filtered_real_comb_fff_sortedbylin = []

    for c in RPRS_filtered_sortedbysens:
        RP, RS, LP, LS, hP, hS, spaz, Bz, flux_S1_array, zS1_center, dz_shift, dfluxsens, lin = c

        zmin = -(spaz + LS - LP/2)
        zmax = -zmin
        ztot = np.arange(zmin, zmax + resolution, resolution) 

        rmax = max(RS + hS/2, RP + hP/2, resolution)
        rtot = np.arange(0, rmax + resolution, resolution)

        filtered_real_comb_fff_sortedbysens.append(
            (RP, RS, LP, LS, hP, hS, spaz, lin, dfluxsens, rtot, ztot)
        )

    for c in RPRS_filtered_sortedbylin:
        RP, RS, LP, LS, hP, hS, spaz, Bz, flux_S1_array, zS1_center, dz_shift, dfluxsens, lin = c

        zmin = -(spaz + LS - LP/2)
        zmax = -zmin
        ztot = np.arange(zmin, zmax + resolution, resolution)

        rmax = max(RS + hS/2, RP + hP/2, resolution)
        rtot = np.arange(0, rmax + resolution, resolution)

        filtered_real_comb_fff_sortedbylin.append(
            (RP, RS, LP, LS, hP, hS, spaz, lin, dfluxsens, rtot, ztot)
        )

    return filtered_real_comb_fff_sortedbysens, filtered_real_comb_fff_sortedbylin

# ---------------------------------------------------------
# PROGRESS DIALOG
# ---------------------------------------------------------
class ProgressDialog(QDialog):
    def __init__(self, title="Running...", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(400, 120)

        layout = QVBoxLayout(self)

        self.label = QLabel("Starting...")
        self.label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.label)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        layout.addWidget(self.bar)

    def update_message(self, text):
        self.label.setText(text)
        QApplication.processEvents()

    def update_progress(self, value):
        self.bar.setValue(value)
        QApplication.processEvents()

# ---------------------------------------------------------
# WORKER THREAD
# ---------------------------------------------------------
class DesignComputationWorker(QThread):
    finished = Signal(dict)
    message = Signal(str)
    progress = Signal(int)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        p = self.params

        # converti in metri
        RMAX = p["RMAX"] * 1e-3
        LMAX = p["LMAX"] * 1e-3
        dmax = p["dmax"] * 1e-3
        resolution = p["resolution_mm"] * 1e-3


        # STEP 1
        self.message.emit("Computing field...")
        RP_LP_hP_Bz_set, RPRS_LPLS_hPhS_spaz_Bz_set = compute_field(
            RMAX, LMAX, dmax, resolution,
            p["dP_AWG"], p["dS_AWG"], p["I"], p["iso"],
            self.progress, self.message
        )
        
        if len(RPRS_LPLS_hPhS_spaz_Bz_set) == 0:
            self.finished.emit({"error": "No valid combinations"})
            return

        # STEP 2
        self.message.emit("Computing flux and filtering...")
        filtered_sens, filtered_lin = compute_flux_and_filter(
            RPRS_LPLS_hPhS_spaz_Bz_set, RMAX, LMAX, dmax, resolution,
            p["dP_AWG"], p["dS_AWG"], p["I"], p["iso"], p["f0_kHz"],
            self.progress, self.message
        )
        
        if len(filtered_sens) == 0:
            self.finished.emit({"error": "No valid combinations"})
            return
        
        self.finished.emit({
            "RP_LP_hP_Bz_set": RP_LP_hP_Bz_set,
            "RPRS_LPLS_hPhS_spaz_Bz_set": RPRS_LPLS_hPhS_spaz_Bz_set,
            "filtered_sens": filtered_sens,
            "filtered_lin": filtered_lin,
            "params": p
        })
    
#----------------------------------------------------------
# OTTIMIZZAZIONE FINESTRA
# ---------------------------------------------------------
class OptimizeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimization options")
        self.setModal(True)
        self.resize(400, 250)

        layout = QVBoxLayout(self)

        title = QLabel("Choose optimization mode")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.btn_primary = QPushButton("Optimize primary coil only")
        self.btn_secondary = QPushButton("Optimize secondary coils only")
        self.btn_total = QPushButton("Optimize both (full optimization)")
        self.btn_back = QPushButton("Back")

        layout.addWidget(self.btn_primary)
        layout.addWidget(self.btn_secondary)
        layout.addWidget(self.btn_total)
        layout.addWidget(self.btn_back)

# ---------------------------------------------------------
# WIDGET PRINCIPALE
# ---------------------------------------------------------
class DesignAlignedWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._param_descriptions = {
            "resolution": "Simulation discretization step in millimeters.",
            "RMAX": "Maximum radial dimension of the LVDT (mm).",
            "LMAX": "Maximum axial length of the LVDT (mm).",
            "dmax": "Maximum displacement to be measured (mm).",
            "dP_AWG": "Primary coil wire diameter (AWG).",
            "dS_AWG": "Secondary coil wire diameter (AWG).",
            "iso": "Insulation ratio.",
            "I": "Primary current (A).",
            "V0": "Primary excitation voltage amplitude (V).",
            "Imax": "Maximum allowable primary current (A).",
            "Pmax": "Maximum allowable primary power (W).",
            "Rcabling": "Cable resistance (Ω).",
            "f0_kHz": "Resonance frequency (kHz).",
            "e_n": "Op-amp voltage noise (nV/√Hz).",
            "i_n": "Op-amp current noise (pA/√Hz).",
            "corr": "Correction factor for electronics (V).",
            "tipo_coll": "Secondary coil connection type (series or parallel)",
        }

        self._results = {}
        self._view_mode = "3d"

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(6)
        self.setLayout(self.layout)

        self.fig = Figure(figsize=(16, 10), dpi=120, facecolor='black')
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: black;")
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(150)
        self._update_timer.timeout.connect(self.render_current_view)

        self.build_ui()
        self.setup_signals()
        #self.render_current_view()
        self._optimization_mode = None
        self._original_dP_AWG = None
        self._original_dS_AWG = None

    def build_ui(self):
        self._selected_mode = False
        title = QLabel("Coaxial aligned - Design parameters")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: white;")
        self.title_label = title

        title_hbox = QHBoxLayout()
        title_hbox.addWidget(self.title_label)
        title_hbox.addStretch()

        help_btn = QPushButton("?")
        help_btn.setFixedSize(28, 28)
        help_btn.clicked.connect(self.show_param_help_dialog)
        title_hbox.addWidget(help_btn)

        form = QFormLayout()
        form.setVerticalSpacing(4)
        form.setHorizontalSpacing(12)

        def spinD(value, step=0.1):
            s = QDoubleSpinBox()
            s.setDecimals(3)
            s.setRange(0, 1000000)
            s.setValue(value)
            s.setSingleStep(step)
            return s

        self.resolution = spinD(0.1)
        form.addRow("resolution [mm]", self.resolution)

        self.RMAX = spinD(50.0)
        form.addRow("RMAX [mm]", self.RMAX)

        self.LMAX = spinD(25.0)
        form.addRow("LMAX [mm]", self.LMAX)

        self.dmax = spinD(8.0)
        form.addRow("dmax [mm]", self.dmax)

        self.dP_AWG = QSpinBox(); self.dP_AWG.setRange(1, 40); self.dP_AWG.setValue(25)
        form.addRow("dP [AWG]", self.dP_AWG)

        self.dS_AWG = QSpinBox(); self.dS_AWG.setRange(1, 40); self.dS_AWG.setValue(32)
        form.addRow("dS [AWG]", self.dS_AWG)

        self.iso = spinD(1.1, step=0.1)
        form.addRow("Insulation ratio", self.iso)

        self.I = spinD(0.001, step=0.0001)
        form.addRow("I [A]", self.I)

        self.V0 = spinD(10.0)
        form.addRow("V0 [V]", self.V0)

        self.Imax = spinD(0.3)
        form.addRow("Imax [A]", self.Imax)

        self.Pmax = spinD(2.121320344)
        form.addRow("Pmax [W]", self.Pmax)

        self.Rcabling = spinD(1.0)
        form.addRow("Rcabling [Ω]", self.Rcabling)

        self.f0_kHz = spinD(46.5)
        form.addRow("Resonance frequency [kHz]", self.f0_kHz)

        self.e_n = spinD(1.1)
        form.addRow("eₙ [nV/√Hz]", self.e_n)

        self.i_n = spinD(1.7)
        form.addRow("iₙ [pA/√Hz]", self.i_n)

        self.corr = spinD(0.25)
        form.addRow("corr [V]", self.corr)

        self.tipo_coll = QComboBox()
        self.tipo_coll.addItems(["Series", "Parallel"])
        form.addRow("Secondary connection type", self.tipo_coll)

        self.left_container = QW()
        left_stack = QVBoxLayout(self.left_container)

        self.input_panel = QW()
        input_layout = QVBoxLayout(self.input_panel)
        input_layout.addLayout(form)

        run_button = QPushButton("START DESIGN")
        run_button.clicked.connect(self.start_simulation)
        input_layout.addWidget(run_button)

        self.results_panel = QW()
        results_layout = QVBoxLayout(self.results_panel)

        btn_resultssens = QPushButton("Show configurations sorted by sensitivity")
        btn_resultslin = QPushButton("Show configurarions sorted by linearity")

        btn_resultssens.clicked.connect(self.show_resultssens_dialog)
        btn_resultslin.clicked.connect(self.show_resultslin_dialog)

        for b in (btn_resultssens, btn_resultslin):
            b.setStyleSheet("font-size: 16px; padding: 10px;")

        results_layout.addWidget(btn_resultssens)
        results_layout.addWidget(btn_resultslin)
        results_layout.addStretch()

        self.results_panel.hide()

        left_stack.addWidget(self.input_panel)
        left_stack.addWidget(self.results_panel)
        self.left_container.setFixedWidth(360)

        right_container = QW()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas_3d = FigureCanvas(Figure(figsize=(4,4)))
        self.ax3d = self.canvas_3d.figure.add_subplot(111, projection='3d')
        right_layout.addWidget(self.canvas_3d)

        self.canvas_front = FigureCanvas(Figure(figsize=(4,4)))
        self.ax_front = self.canvas_front.figure.add_subplot(111)
        right_layout.addWidget(self.canvas_front)

        self.canvas_side = FigureCanvas(Figure(figsize=(4,4)))
        self.ax_side = self.canvas_side.figure.add_subplot(111)
        right_layout.addWidget(self.canvas_side)

        # 3D
        self.canvas_3d.figure.patch.set_facecolor("black")
        self.ax3d.set_facecolor("black")

        # FRONT
        self.canvas_front.figure.patch.set_facecolor("black")
        self.ax_front.set_facecolor("black")

        # SIDE
        self.canvas_side.figure.patch.set_facecolor("black")
        self.ax_side.set_facecolor("black")

        
        self.canvas_front.hide()
        self.canvas_side.hide()

        self.render_current_view()

        view_btns = QHBoxLayout()
        self.btn_view_3d = QPushButton("3D view")
        self.btn_view_front = QPushButton("Front view")
        self.btn_view_side = QPushButton("Side view")
        for b in (self.btn_view_3d, self.btn_view_front, self.btn_view_side):
            b.setStyleSheet("color:white; background-color:#333; padding:6px;")
        self.btn_view_3d.clicked.connect(lambda: self.set_view_mode("3d"))
        self.btn_view_front.clicked.connect(lambda: self.set_view_mode("front"))
        self.btn_view_side.clicked.connect(lambda: self.set_view_mode("side"))
        view_btns.addWidget(self.btn_view_3d)
        view_btns.addWidget(self.btn_view_front)
        view_btns.addWidget(self.btn_view_side)
        right_layout.addLayout(view_btns)

        save_btns = QHBoxLayout()
        self.btn_save_3d = QPushButton("Save 3D view")
        self.btn_save_front = QPushButton("Save front view")
        self.btn_save_side = QPushButton("Save side view")
        for b in (self.btn_save_3d, self.btn_save_front, self.btn_save_side):
            b.setStyleSheet("color:white; background-color:#333; padding:6px;")
        self.btn_save_3d.clicked.connect(self.save_3d_pdf)
        self.btn_save_front.clicked.connect(self.save_front_pdf)
        self.btn_save_side.clicked.connect(self.save_side_pdf)
        save_btns.addWidget(self.btn_save_3d)
        save_btns.addWidget(self.btn_save_front)
        save_btns.addWidget(self.btn_save_side)
        right_layout.addLayout(save_btns)

        h = QHBoxLayout()
        h.addWidget(self.left_container)
        h.addWidget(right_container, stretch=1)

        self.layout.addLayout(title_hbox)
        self.layout.addLayout(h)
        self.layout.addStretch()

        self.add_param_tooltips()

    # ---------------------------------------------------------
    # VIEW MODE
    # ---------------------------------------------------------
    def set_view_mode(self, mode):
        self._view_mode = mode

        # nascondi tutto
        self.canvas_3d.hide()
        self.canvas_front.hide()
        self.canvas_side.hide()

        # mostra solo quello giusto
        if mode == "3d":
            self.canvas_3d.show()
        elif mode == "front":
            self.canvas_front.show()
        elif mode == "side":
            self.canvas_side.show()

        self.render_current_view()

    def render_current_view(self):
        if self._view_mode == "3d":
            self.render_schema_3d_mob(self.ax3d)

        elif self._view_mode == "front":
            self.render_front_view(self.ax_front)

        elif self._view_mode == "side":
            self.render_side_view(self.ax_side)

    # -------------------------
    # Tooltips
    # -------------------------
    def add_param_tooltips(self):
        for key, widget in {
            "resolution": self.resolution,
            "RMAX": self.RMAX,
            "LMAX": self.LMAX,
            "dmax": self.dmax,
            "dP_AWG": self.dP_AWG,
            "dS_AWG": self.dS_AWG,
            "iso": self.iso,
            "I": self.I,
            "V0": self.V0,
            "Imax": self.Imax,
            "Pmax": self.Pmax,
            "Rcabling": self.Rcabling,
            "f0_kHz": self.f0_kHz,
            "e_n": self.e_n,
            "i_n": self.i_n,
            "corr": self.corr,
            "tipo_coll": self.tipo_coll,
        }.items():
            widget.setToolTip(self._param_descriptions[key])

    # -------------------------
    # Help dialog
    # -------------------------
    def show_param_help_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Parameters and descriptions")
        dlg.resize(560, 420)

        layout = QVBoxLayout(dlg)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout(content)

        for i, (k, desc) in enumerate(self._param_descriptions.items()):
            name = QLabel(f"<b>{k}</b>")
            d = QLabel(desc)
            d.setWordWrap(True)
            d.setStyleSheet("color: #EEE;")
            grid.addWidget(name, i, 0)
            grid.addWidget(d, i, 1)

        scroll.setWidget(content)
        layout.addWidget(scroll)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)

        dlg.exec()

    # -------------------------
    # Signals
    # -------------------------
    def setup_signals(self):
        widgets = [
            self.resolution, self.RMAX, self.LMAX, self.dmax,
            self.dP_AWG, self.dS_AWG, self.iso, self.I,
            self.V0, self.Imax, self.Pmax, self.Rcabling,
            self.f0_kHz, self.e_n, self.i_n, self.corr
        ]
        for w in widgets:
            w.valueChanged.connect(self._update_timer.start)

    # ---------------------------------------------------------
    # Params
    # ---------------------------------------------------------
    def get_params(self):
        return {
            "resolution_mm": self.resolution.value(),
            "RMAX": self.RMAX.value(),
            "LMAX": self.LMAX.value(),
            "dmax": self.dmax.value(),
            "dP_AWG": self.dP_AWG.value(),
            "dS_AWG": self.dS_AWG.value(),
            "iso": self.iso.value(),
            "I": self.I.value(),
            "V0": self.V0.value(),
            "Imax": self.Imax.value(),
            "Pmax": self.Pmax.value(),
            "Rcabling": self.Rcabling.value(),
            "f0_kHz": self.f0_kHz.value(),
            "e_n": self.e_n.value(),
            "i_n": self.i_n.value(),
            "corr": self.corr.value(),
            "tipo_coll": self.tipo_coll.currentText(), 
        }

    # -------------------------
    # Start design (solo worker + dialog)
    # -------------------------
    def start_simulation(self):
        self._update_timer.stop()

        params = self.get_params()

        self.progress_dialog = ProgressDialog("Running design simulation...", self)
        self.progress_dialog.show()

        self.worker = DesignComputationWorker(params)
        self.worker.message.connect(self.progress_dialog.update_message)
        self.worker.progress.connect(self.progress_dialog.update_progress)
        self.worker.finished.connect(self.on_simulation_finished)
        self.worker.start()

    def on_simulation_finished(self, result):
        self.progress_dialog.close()

        if "error" in result:
            dlg = QDialog(self)
            dlg.setWindowTitle("No valid combinations")
            layout = QVBoxLayout(dlg)
            layout.addWidget(QLabel("No valid coil geometries satisfy constraints."))
            btn = QPushButton("Close")
            btn.clicked.connect(dlg.accept)
            layout.addWidget(btn)
            dlg.exec()
            return

        self._results = result

        self.show_results_panel()

    def show_input_panel(self):
        self.title_label.setText("Coaxial aligned – Design parameters")
        self.input_panel.show()
        self.results_panel.hide()

    def show_results_panel(self):
        self.title_label.setText("Coaxial aligned – Design results")
        self.input_panel.hide()
        self.results_panel.show()

    # -------------------------
    # Render 3D (le altre view puoi tenerle come le avevi)
    # -------------------------
    def render_schema_3d_mob(self, ax=None):
        if ax is None:
            ax = self.ax3d

        ax.clear()
        ax.figure.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)
        ax.set_title("3D view", color='white')
        ax.set_xlabel("z [mm]", color='white')
        ax.set_ylabel("x [mm]", color='white')
        ax.set_zlabel("y [mm]", color='white')
        ax.xaxis.label.set_color('white')
        ax.tick_params(colors='white')

        # ============================
        # MODALITÀ NORMALE (RMAX / LMAX)
        # ============================
        if not self._selected_mode:
            R = self.RMAX.value()
            L = self.LMAX.value()

            R = max(0, R)
            L = max(0, L)

            color = '#00AAFF'

            x, y, z = cylinder_surface(R, -L/2, L/2)
            ax.plot_surface(z, x, y, color=color, alpha=0.8)

            theta = np.linspace(0, 2*np.pi, 240)
            xcap = R * np.cos(theta)
            ycap = R * np.sin(theta)

            ax.plot_surface(np.full((2, len(theta)), -L/2),
                            np.array([xcap, xcap]),
                            np.array([ycap, ycap]),
                            color=color, alpha=0.8)

            ax.plot_surface(np.full((2, len(theta)), L/2),
                            np.array([xcap, xcap]),
                            np.array([ycap, ycap]),
                            color=color, alpha=0.8)

            lim = max(R, L/2) * 1.1
            ax.set_aspect('equal', 'box')
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.set_box_aspect([3, 1, 1])
            ax.view_init(elev=10, azim=110)

            self.canvas_3d.draw()
            return

        # ============================
        # MODALITÀ SELEZIONATA (RP/RS/LP/LS/hP/hS/spaz)
        # ============================
        RP, RS, LP, LS, hP, hS, spaz, lin, sens, rtot, ztot = self.selected_config
        RP = RP*1e3
        RS = RS*1e3
        LP = LP*1e3
        LS = LS*1e3
        hP = hP*1e3
        hS = hS*1e3
        spaz = spaz*1e3

        RPo_out = RP + hP / 2
        RPo_in = max(0, RP - hP / 2)
        RSo_out = RS + hS / 2
        RSo_in = max(0, RS - hS / 2)

        zP_min, zP_max = -LP / 2, LP / 2
        zS1_min = spaz / 2
        zS1_max = LS + spaz / 2
        zS2_max = -zS1_max
        zS2_min = -zS1_min

        theta = np.linspace(0, 2 * np.pi, 240)

        color_primary = '#00FF88'
        color_secondary = '#66CCFF'
        sec_alpha = 0.6

        # SECONDARY 1
        xS1o, yS1o, zS1o = cylinder_surface(RSo_out, zS1_min, zS1_max)
        ax.plot_surface(zS1o, xS1o, yS1o, color=color_secondary, alpha=sec_alpha)
        xS1i, yS1i, zS1i = cylinder_surface(RSo_in, zS1_min, zS1_max)
        ax.plot_surface(zS1i, xS1i, yS1i, color=color_secondary, alpha=sec_alpha)

        xcap_s = np.array([RSo_out * np.cos(theta), RSo_in * np.cos(theta)])
        ycap_s = np.array([RSo_out * np.sin(theta), RSo_in * np.sin(theta)])
        ax.plot_surface(np.full_like(xcap_s, zS1_min), xcap_s, ycap_s, color=color_secondary, alpha=sec_alpha)
        ax.plot_surface(np.full_like(xcap_s, zS1_max), xcap_s, ycap_s, color=color_secondary, alpha=sec_alpha)

        # SECONDARY 2
        xS2o, yS2o, zS2o = cylinder_surface(RSo_out, zS2_min, zS2_max)
        ax.plot_surface(zS2o, xS2o, yS2o, color=color_secondary, alpha=sec_alpha)
        xS2i, yS2i, zS2i = cylinder_surface(RSo_in, zS2_min, zS2_max)
        ax.plot_surface(zS2i, xS2i, yS2i, color=color_secondary, alpha=sec_alpha)

        ax.plot_surface(np.full_like(xcap_s, zS2_min), xcap_s, ycap_s, color=color_secondary, alpha=sec_alpha)
        ax.plot_surface(np.full_like(xcap_s, zS2_max), xcap_s, ycap_s, color=color_secondary, alpha=sec_alpha)

        # PRIMARY
        xPo, yPo, zPo = cylinder_surface(RPo_out, zP_min, zP_max)
        ax.plot_surface(zPo, xPo, yPo, color=color_primary, alpha=1.0)
        xPi, yPi, zPi = cylinder_surface(RPo_in, zP_min, zP_max)
        ax.plot_surface(zPi, xPi, yPi, color=color_primary, alpha=1.0)

        xcap = np.array([RPo_out * np.cos(theta), RPo_in * np.cos(theta)])
        ycap = np.array([RPo_out * np.sin(theta), RPo_in * np.sin(theta)])
        ax.plot_surface(np.full_like(xcap, zP_min), xcap, ycap, color=color_primary, alpha=1.0)
        ax.plot_surface(np.full_like(xcap, zP_max), xcap, ycap, color=color_primary, alpha=1.0)

        lim = max(RPo_out, RSo_out, abs(zS1_max), abs(zS2_min)) * 1.05
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect([3, 1, 1])
        ax.view_init(elev=10, azim=110)

        self.canvas_3d.draw()

    # ---------------------------------------------------------
    # FRONT VIEW (radiale)
    # ---------------------------------------------------------
    def render_front_view(self, ax=None):
        if ax is None:
            ax = self.ax_front

        ax.clear()
        
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.figure.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)
        
        if not self._selected_mode:
            R = self.RMAX.value()

            secondary_color = '#66CCFF'
            text_color = 'white'

            ax.add_patch(plt.Circle((0, 0), R, fill=True, color=secondary_color, alpha=0.9))
            ax.add_patch(plt.Circle((0, 0), R-0.1, fill=True, color="black", alpha=1))

            ax.plot([0, R], [0, 0], color=text_color, lw=2)
            ax.text(R/2, R*0.05, "RMAX", color=text_color, ha='center')

            ax.set_xlim(-R*1.1, R*1.1)
            ax.set_ylim(-R*1.1, R*1.1)

            ax.set_title("Front view (radial section)", color='white')
            ax.set_xlabel("x [mm]", color='white')
            ax.set_ylabel("y [mm]", color='white')
            ax.tick_params(colors='white')
            ax.grid(False)

            self.canvas_front.draw()
            return

        # QUI metti la versione con RP, RS, hP, hS
        RP, RS, LP, LS, hP, hS, spaz, lin, sens, rtot, ztot = self.selected_config
        RP = RP*1e3
        RS = RS*1e3
        hP = hP*1e3
        hS = hS*1e3
        primary_color = '#00FF88'
        secondary_color = '#66CCFF'
        text_color = 'white'
        
        if RP > RS:
            # disegna prima primario, poi secondario
            # -------------------------
            # PRIMARY RING (anello)
            # -------------------------
            ax.add_patch(plt.Circle((0, 0), RP + hP/2,
                                    fill=True, color=primary_color, alpha=0.9))
            ax.add_patch(plt.Circle((0, 0), RP - hP/2,
                                    fill=True, color='black'))
            # -------------------------
            # SECONDARY RING (anello)
            # -------------------------
            # Outer radius
            ax.add_patch(plt.Circle((0, 0), RS + hS/2,
                                    fill=True, color=secondary_color, alpha=0.6))
            # Inner radius (hole)
            ax.add_patch(plt.Circle((0, 0), RS - hS/2,
                                    fill=True, color='black'))

        else:
            # disegna prima secondario, poi primario
            # -------------------------
            # SECONDARY RING (anello)
            # -------------------------
            # Outer radius
            ax.add_patch(plt.Circle((0, 0), RS + hS/2,
                                    fill=True, color=secondary_color, alpha=0.6))
            # Inner radius (hole)
            ax.add_patch(plt.Circle((0, 0), RS - hS/2,
                                    fill=True, color='black'))
            # -------------------------
            # PRIMARY RING (anello)
            # -------------------------
            ax.add_patch(plt.Circle((0, 0), RP + hP/2,
                                    fill=True, color=primary_color, alpha=0.9))
            ax.add_patch(plt.Circle((0, 0), RP - hP/2,
                                    fill=True, color='black'))

        # -------------------------
        # PARAMETER LINES
        # -------------------------
        # RP
        ax.plot([0, RP], [0, 0], color=text_color, lw=2)
        ax.text(RP/2, 1, "RP", color=text_color, ha='center')

        # RS
        ax.plot([0, 0], [0, RS], color=text_color, lw=2)
        ax.text(1, RS/2, "RS", color=text_color, va='center')

        # hP
        x1 = -(RP - hP/2)
        x2 = -(RP + hP/2)
        ax.plot([x1, x2], [0, 0], color=text_color, lw=2)
        ax.text((x1 + x2)/2, -5, "hP", color=text_color, ha='center')

        # hS
        y1 = -(RS - hS/2)
        y2 = -(RS + hS/2)
        ax.plot([0, 0], [y1, y2], color=text_color, lw=2)
        ax.text(2, (y1 + y2)/2, "hS", color=text_color, va='center')

        # -------------------------
        # LIMITS
        # -------------------------
        Rmax = max(RS + hS, RP + hP)
        ax.set_xlim(-Rmax*1.4, Rmax*1.4)
        ax.set_ylim(-Rmax*1.4, Rmax*1.4)
        ax.margins(0.1)
        
        ax.set_title("Front view (radial section)", color='white')
        ax.set_xlabel("x [mm]", color='white')
        ax.set_ylabel("y [mm]", color='white')
        ax.tick_params(colors='white')
        ax.get_tightbbox
        ax.grid(False)

        self.canvas_front.draw()

    # ---------------------------------------------------------
    # SIDE VIEW (longitudinale)
    # ---------------------------------------------------------
    def render_side_view(self, ax=None):
        if ax is None:
            ax = self.ax_side

        ax.clear()
        ax.set_facecolor('black')
        ax.figure.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)

        if not self._selected_mode:
            R = self.RMAX.value()
            L = self.LMAX.value()

            secondary_color = '#66CCFF'
            text_color = 'white'

            ax.fill_between([-L/2, L/2], -R/2, R/2, color=secondary_color, alpha=0.6)

            ax.plot([-L/2, L/2], [R/2 + 2, R/2 + 2], color=text_color, lw=2)
            ax.text(0, R/2 + 4, "LMAX", color=text_color, ha='center')

            ax.plot([L/2 + 2, L/2 + 2], [-R/2, R/2], color=text_color, lw=2)
            ax.text(L/2 + 10, 0, "RMAX", color=text_color, ha='center')

            ax.set_xlim(-L, L)
            ax.set_ylim(-R, R)
            #ax.set_aspect('equal', 'box')
            ax.set_title("Side view (longitudinal section)", color='white')
            ax.set_xlabel("z [mm]", color='white')
            ax.set_ylabel("y [mm]", color='white')
            ax.tick_params(colors='white')
            ax.grid(False)

            self.canvas_side.draw()
            return

        # QUI metti la versione con LP, LS, spaz
        RP, RS, LP, LS, hP, hS, spaz, lin, sens, rtot, ztot = self.selected_config
        RP = RP*1e3
        RS = RS*1e3
        LP = LP*1e3
        LS = LS*1e3
        hP = hP*1e3
        hS = hS*1e3
        spaz = spaz*1e3

        primary_color = '#00FF88'
        secondary_color = '#66CCFF'
        text_color = 'white'

        # PRIMARY
        ax.fill_between([-LP/2, LP/2], RP - hP/2, RP + hP/2, color=primary_color, alpha=0.9)
        ax.fill_between([-LP/2, LP/2], -(RP - hP/2), -(RP + hP/2), color=primary_color, alpha=0.9)

        # SECONDARIES
        ax.fill_between([spaz/2, spaz/2 + LS], RS - hS/2, RS + hS/2, color=secondary_color, alpha=0.6)
        ax.fill_between([-spaz/2 - LS, -spaz/2], RS - hS/2, RS + hS/2, color=secondary_color, alpha=0.6)
        ax.fill_between([spaz/2, spaz/2 + LS], -(RS - hS/2), -(RS + hS/2), color=secondary_color, alpha=0.6)
        ax.fill_between([-spaz/2 - LS, -spaz/2], -(RS - hS/2), -(RS + hS/2), color=secondary_color, alpha=0.6)

        # LP
        ax.plot([-LP/2, LP/2], [RP + hP/2 + 1.5, RP + hP/2 + 1.5], color=text_color, lw=2)
        ax.text(0, RP + hP/2 + 2.5, "LP", color=text_color, ha='center')

        # LS
        ax.plot([spaz/2, spaz/2 + LS], [(RS + hS/2 + 1.5), (RS + hS/2 + 1.5)], color=text_color, lw=2)
        ax.text(spaz/2 + LS/2, RS + hS/2 + 2.5, "LS", color=text_color, ha='center')

        # spacing
        y_spaz = min(-(RS - hS/2 - 1), -(RP - hP/2 - 1))
        ax.plot([-spaz/2, spaz/2], [y_spaz, y_spaz], color=text_color, lw=2)
        ax.text(0, min(-(RS + hS/2 + 2), -(RP + hP/2 + 2)), "spaz", color=text_color, ha='center')
        
        # RP
        ax.plot([0, 0], [0, RP], color=text_color, lw=2)
        ax.text(- 1.5, RP/2, "RP", color=text_color, ha='center')

        # RS
        ax.plot([(spaz/2 + LS/2), (spaz/2 + LS/2)], [0, RS], color=text_color, lw=2)
        ax.text((spaz/2 + LS/2 + 1), RS/2, "RS", color=text_color, ha='center')
        
        # hP
        ax.plot([-LP/2 - 1, -LP/2 - 1], [RP - hP/2, RP + hP/2], color=text_color, lw=2)
        ax.text(-LP/2 - 2, RP, "hP", color=text_color, ha='center')

        # hS
        ax.plot([(spaz/2 + LS + 1), (spaz/2 + LS + 1)], [RS - hS/2, + RS + hS/2], color=text_color, lw=2)
        ax.text((spaz/2 + LS + 2), RS, "hS", color=text_color, ha='center')

        ax.set_xlim(-(spaz/2 + LS)*1.5, (spaz/2 + LS)*1.5)
        ax.set_ylim(-max(RS + hS/2, RP + hP/2)*1.5, max(RS + hS/2, RP + hP/2)*1.5)

        ax.set_title("Side view (longitudinal section)", color='white')
        #ax.set_aspect('equal', 'box')
        ax.set_ylabel("y [mm]", color='white')
        ax.set_xlabel("z [mm]", color='white')
        ax.tick_params(colors='white')
        ax.get_tightbbox
        ax.grid(False)
        ax.set_title("Side view (longitudinal section)", color='white')

        self.canvas_side.draw()
 
    # ---------------------------------------------------------
    # SAVE PDF FUNCTIONS
    # ---------------------------------------------------------
    def save_3d_pdf(self):
        # 1) forza il renderer a ridisegnare la vista corrente
        self.render_current_view()
        self.canvas_3d.draw()

        fig = self.canvas_3d.figure

        fname, _ = QFileDialog.getSaveFileName(self, "Save 3D view", "", "PDF Files (*.pdf)")
        if fname:
            fig.savefig(
                fname,
                dpi=300,
                bbox_inches="tight",
                facecolor="black",   # <-- PDF nero
                edgecolor="black"
            )

    def save_front_pdf(self):
        # forza il renderer a ridisegnare la vista corrente
        self.render_current_view()
        self.canvas_front.draw()

        fig = self.canvas_front.figure

        fname, _ = QFileDialog.getSaveFileName(self, "Save front view", "", "PDF Files (*.pdf)")
        if fname:
            fig.savefig(
                fname,
                dpi=300,
                bbox_inches="tight",
                facecolor="black",   # PDF nero
                edgecolor="black"
            )

    def save_side_pdf(self):
        # forza il renderer a ridisegnare la vista corrente
        self.render_current_view()
        self.canvas_side.draw()

        fig = self.canvas_side.figure

        fname, _ = QFileDialog.getSaveFileName(self, "Save side view", "", "PDF Files (*.pdf)")
        if fname:
            fig.savefig(
                fname,
                dpi=300,
                bbox_inches="tight",
                facecolor="black",   # PDF nero
                edgecolor="black"
            )

    # ---------------------------------------------------------
    # RESULTS DIALOG: configurations sorted by linearity or sensitivity
    # ---------------------------------------------------------
    def show_resultssens_dialog(self):
        if not self._results or "filtered_sens" not in self._results:
            dlg = QDialog(self)
            dlg.setWindowTitle("No results available")
            layout = QVBoxLayout(dlg)
            layout.addWidget(QLabel("Run the simulation first."))
            btn = QPushButton("Close")
            btn.clicked.connect(dlg.accept)
            layout.addWidget(btn)
            dlg.exec()
            return

        configs = self._results["filtered_sens"]
        params = self._results["params"]

        dlg = QDialog(self)
        dlg.setWindowTitle("Configurations sorted by sensitivity")
        dlg.resize(600, 700)

        main_layout = QVBoxLayout(dlg)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        vbox = QVBoxLayout(content)

        # ============================
        # 1) INPUT SUMMARY (copiabile)
        # ============================
        summary_text = (
            "<b>INPUT SUMMARY</b><br>"
            f"RMAX = {params['RMAX']:.3f} mm<br>"
            f"LMAX = {params['LMAX']:.3f} mm<br>"
            f"dmax = {params['dmax']:.3f} mm<br>"
            f"resolution = {params['resolution_mm']:.3f} mm<br>"
            f"dP_AWG = {params['dP_AWG']}<br>"
            f"dS_AWG = {params['dS_AWG']}<br>"
            f"iso = {params['iso']:.3f}<br>"
            f"I = {params['I']:.6f} A<br>"
            f"f0 = {params['f0_kHz']:.3f} kHz<br>"
        )

        lbl_summary = QLabel(summary_text)
        lbl_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        vbox.addWidget(lbl_summary)
        vbox.addSpacing(20)

        # Per ogni configurazione
        for i, c in enumerate(configs):
            RP, RS, LP, LS, hP, hS, spaz, lin, sens, rtot, ztot = c

            # Contenitore orizzontale
            row = QHBoxLayout()

            # Testo copiabile
            text = (
                f"Config {i}\n"
                f"  RP = {RP*1e3:.3f} mm\n"
                f"  RS = {RS*1e3:.3f} mm\n"
                f"  LP = {LP*1e3:.3f} mm\n"
                f"  LS = {LS*1e3:.3f} mm\n"
                f"  hP = {hP*1e3:.3f} mm\n"
                f"  hS = {hS*1e3:.3f} mm\n"
                f"  spaz = {spaz*1e3:.3f} mm\n"
                f"  linearity = {lin*1e3:.3f} mm\n"
                f"  sensitivity = {2*sens*2*np.pi*params['f0_kHz']*1e3:.6e} V/m\n"
            )

            lbl = QLabel(text)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            row.addWidget(lbl)

            # Bottone selezione
            btn = QPushButton("Select")
            btn.clicked.connect(lambda _, idx=i: self.on_select_configuration(idx, "filtered_sens"))
            row.addWidget(btn)

            vbox.addLayout(row)
            vbox.addSpacing(10)

        content.setLayout(vbox)
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        main_layout.addWidget(close_btn)

        dlg.exec()
        
    def show_resultslin_dialog(self):
        if not self._results or "filtered_lin" not in self._results:
            dlg = QDialog(self)
            dlg.setWindowTitle("No results available")
            layout = QVBoxLayout(dlg)
            layout.addWidget(QLabel("Run the simulation first."))
            btn = QPushButton("Close")
            btn.clicked.connect(dlg.accept)
            layout.addWidget(btn)
            dlg.exec()
            return

        configs = self._results["filtered_lin"]
        params = self._results["params"]
        
        dlg = QDialog(self)
        dlg.setWindowTitle("Configurations sorted by linearity")
        dlg.resize(600, 700)

        main_layout = QVBoxLayout(dlg)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        vbox = QVBoxLayout(content)

        # ============================
        # 1) INPUT SUMMARY (copiabile)
        # ============================
        summary_text = (
            "<b>INPUT SUMMARY</b><br>"
            f"RMAX = {params['RMAX']:.3f} mm<br>"
            f"LMAX = {params['LMAX']:.3f} mm<br>"
            f"dmax = {params['dmax']:.3f} mm<br>"
            f"resolution = {params['resolution_mm']:.3f} mm<br>"
            f"dP_AWG = {params['dP_AWG']}<br>"
            f"dS_AWG = {params['dS_AWG']}<br>"
            f"iso = {params['iso']:.3f}<br>"
            f"I = {params['I']:.6f} A<br>"
            f"f0 = {params['f0_kHz']:.3f} kHz<br>"
        )
        
        lbl_summary = QLabel(summary_text)
        lbl_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        vbox.addWidget(lbl_summary)
        vbox.addSpacing(20)

        # Per ogni configurazione
        for i, c in enumerate(configs):
            RP, RS, LP, LS, hP, hS, spaz, lin, sens, rtot, ztot = c

            # Contenitore orizzontale
            row = QHBoxLayout()

            # Testo copiabile
            text = (
                f"Config {i}\n"
                f"  RP = {RP*1e3:.3f} mm\n"
                f"  RS = {RS*1e3:.3f} mm\n"
                f"  LP = {LP*1e3:.3f} mm\n"
                f"  LS = {LS*1e3:.3f} mm\n"
                f"  hP = {hP*1e3:.3f} mm\n"
                f"  hS = {hS*1e3:.3f} mm\n"
                f"  spaz = {spaz*1e3:.3f} mm\n"
                f"  linearity = {lin*1e3:.3f} mm\n"
                f"  sensitivity = {2*sens*2*np.pi*params['f0_kHz']*1e3:.6e} V/m\n"
            )

            lbl = QLabel(text)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            row.addWidget(lbl)

            # Bottone selezione
            btn = QPushButton("Select")
            btn.clicked.connect(lambda _, idx=i: self.on_select_configuration(idx, "filtered_lin"))
            row.addWidget(btn)

            vbox.addLayout(row)
            vbox.addSpacing(10)

        content.setLayout(vbox)
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        main_layout.addWidget(close_btn)

        dlg.exec()

    #---------------------------------------------------------
    # ALTRA SCHERMATA: simulazione della configurazione selezionata
    #---------------------------------------------------------  
    def compute_from_params(self, p, bar=None, label=None):
        # se non passi bar/label, non fare progress bar
        use_bar = bar is not None and label is not None
        
        RP, RS, LP, LS, hP, hS, spaz, lin, sens, rtot, ztot = self.selected_config

        resolution = p["resolution_mm"] * 1e-3
        I = p["I"]
        f0 = p["f0_kHz"]
        omega_0 = 2 * np.pi * f0 * 1e3

        iso = p["iso"]
        corr = p["corr"]

        dP_AWG = p["dP_AWG"]
        dS_AWG = p["dS_AWG"]

        dP = 0.127 * 92 ** ((36 - dP_AWG) / 39) * 1e-3
        dS = 0.127 * 92 ** ((36 - dS_AWG) / 39) * 1e-3

        if use_bar:
            bar.setValue(10); QApplication.processEvents()

        step = resolution
        rmax = max(RP + hP/2, RS + hS/2)
        zmin = -(spaz + LS - LP/2)
        zmax =  (spaz + LS - LP/2)

        ztot = np.arange(zmin, zmax + step, step)
        rtot = np.arange(0, rmax + step, step)

        if use_bar:
            label.setText("Computing magnetic field Bz...")
            bar.setValue(20); QApplication.processEvents()

        Bz = Bz_solenoide(I, RP, dP, LP, hP, 0, 0, ztot, rtot, iso, step)

        Bz_smooth = np.zeros_like(Bz)
        for j in range(len(rtot)):
            Bz_smooth[:, j] = gaussian_filter1d(Bz[:, j], sigma=5)
        Bz = Bz_smooth

        if use_bar:
            bar.setValue(40); QApplication.processEvents()

        rho = 1.724e-8

        Nlat_pri = int(LP / (dP * iso))
        Nvert_pri = int(4/(2*np.sqrt(3)) * hP / (dP*iso))
        N_pri = Nlat_pri * Nvert_pri
        len_pri = 2*np.pi*RP*N_pri
        sec_pri = np.pi * (dP/2)**2
        R_pri = rho * len_pri / sec_pri
        L_pri_uH = (31.6 * N_pri**2 * (RP - hP/2)**2) / (6*(RP - hP/2) + 9*LP + 10*hP)

        Nlat_sec = int(LS / (dS * iso))
        Nvert_sec = int(4/(2*np.sqrt(3)) * hS / (dS*iso))
        N_sec = Nlat_sec * Nvert_sec
        len_sec = 2*np.pi*RS*N_sec
        sec_sec = np.pi * (dS/2)**2
        R_sec = rho * len_sec / sec_sec
        L_sec_uH = (31.6 * N_sec**2 * (RS - hS/2)**2) / (6*(RS - hS/2) + 9*LS + 10*hS)

        if use_bar:
            bar.setValue(60); QApplication.processEvents()

        dzmax = spaz/2 - LP
        dz_shift = np.arange(-dzmax, dzmax + step, step)

        zS1_center = spaz/2 + LS/2

        flux_S1 = np.zeros_like(dz_shift)

        for i, dzs in enumerate(dz_shift):
            zS1 = zS1_center + dzs
            flux_S1[i] = flusso_secondario(Bz, ztot, rtot, RS, dS, LS, hS, 0, zS1, iso, step)

            if use_bar and i % max(1, len(dz_shift)//10) == 0:
                bar.setValue(60 + int(20*i/len(dz_shift)))
                QApplication.processEvents()

        flux_S2 = flux_S1[::-1]

        if use_bar:
            bar.setValue(85); QApplication.processEvents()

        efm1 = -omega_0 * flux_S1
        efm2 = -omega_0 * flux_S2
        deltaefm = efm2 - efm1

        window = 25
        errors = []
        for k in range(window, len(dz_shift)-window):
            dz_loc = dz_shift[k-window:k+window]
            ef_loc = deltaefm[k-window:k+window]
            m_loc, q_loc = np.polyfit(dz_loc, ef_loc, 1)
            y_fit = m_loc * dz_loc + q_loc
            err = np.mean(np.abs(ef_loc - y_fit))
            errors.append(err)

        if len(errors) == 0:
            # fallback se qualcosa va storto
            lin = 0.0
            m = q = m_corr = np.nan
            dz0 = 0.0
            dz_lin = np.array([])
        else:
            k_best = np.argmin(errors) + window
            dz0 = dz_shift[k_best]

            dz_loc = dz_shift[k_best-window:k_best+window]
            ef_loc = deltaefm[k_best-window:k_best+window]
            m_loc, q_loc = np.polyfit(dz_loc, ef_loc, 1)

            eps = 0.025 * abs(m_loc) * np.max(np.abs(dz_shift))

            i_left = k_best
            i_right = k_best

            while i_left > 0:
                y_pred = m_loc * dz_shift[i_left] + q_loc
                if abs(deltaefm[i_left] - y_pred) < eps:
                    i_left -= 1
                else:
                    break

            while i_right < len(dz_shift)-1:
                y_pred = m_loc * dz_shift[i_right] + q_loc
                if abs(deltaefm[i_right] - y_pred) < eps:
                    i_right += 1
                else:
                    break

            dz_lin = dz_shift[i_left:i_right+1]

            if len(dz_lin) >= 2:
                dz_min = dz_shift[i_left]
                dz_max = dz_shift[i_right]
                lin = min(abs(dz_min - dz0), abs(dz_max - dz0))
                mask_fit = (dz_shift >= -lin) & (dz_shift <= lin)
                x_fit = dz_shift[mask_fit]
                y_fit = deltaefm[mask_fit]
                m, q = np.polyfit(x_fit, y_fit, 1)
            else:
                lin = 0.0
                m, q = np.nan, np.nan

        m_corr = m * corr

        kT = 26e-3 * 1.6e-19
        NoiseResSec = np.sqrt(4 * kT * R_sec)
        DisplacementNoise = np.sqrt(2) * NoiseResSec
        EquivalentDisplacement = DisplacementNoise/abs(m) if np.isfinite(m) and m != 0 else np.nan
        EquivalentDisplacementcorr = DisplacementNoise/abs(m_corr) if np.isfinite(m_corr) and m_corr != 0 else np.nan

        idx_center = np.abs(zS1_center + dz_shift - (spaz/2 + LS/2)).argmin()
        dflux1 = np.gradient(flux_S1, step)
        sens_t = 2 * abs(dflux1[idx_center]) * omega_0

        deltaefm_readout = deltaefm * corr

        if use_bar:
            bar.setValue(100); QApplication.processEvents()
            time.sleep(0.1)

        self._selected_results = {
            "params": p,
            "dP_mm": dP * 1e3,
            "dS_mm": dS * 1e3,
            "ztot": ztot,
            "rtot": rtot,
            "Bz": Bz,
            "N_pri": N_pri,
            "Nvert_pri": Nvert_pri,
            "Nlat_pri": Nlat_pri,
            "N_sec": N_sec,
            "Nvert_sec": Nvert_sec,
            "Nlat_sec": Nlat_sec,
            "R_pri": R_pri,
            "R_sec": R_sec,
            "L_pri_uH": L_pri_uH,
            "L_sec_uH": L_sec_uH,
            "flux_S1": flux_S1,
            "flux_S2": flux_S2,
            "efm1": efm1,
            "efm2": efm2,
            "deltaefm": deltaefm,
            "deltaefm_readout": deltaefm_readout,
            "dz_shift": dz_shift,
            "sens_t": sens_t,
            "sens_fit": m,
            "q": q,
            "qcorr": q*corr if np.isfinite(q) else np.nan,
            "sens_fit_corr": m_corr,
            "linear_range": lin,
            "linear_center": dz0,
            "Noise_voltage": DisplacementNoise,
            "EquivalentDisplacement": EquivalentDisplacement,
            "EquivalentDisplacementcorr": EquivalentDisplacementcorr
        }

    def _make_plot_figures(self):
        r = self._selected_results
        p = r["params"]

        figs = []

        # Unpack
        ztot = r["ztot"]
        rtot = r["rtot"]
        Bz = r["Bz"]
        flux_S1 = r["flux_S1"]
        flux_S2 = r["flux_S2"]
        efm1 = r["efm1"]
        efm2 = r["efm2"]
        deltaefm = r["deltaefm"]
        deltaefm_readout = r["deltaefm_readout"]
        dz_shift = r["dz_shift"]

        # Fit results
        m = r["sens_fit"]
        q = r["q"]
        m_corr = r["sens_fit_corr"]
        q_corr = r["qcorr"]
        lin = r["linear_range"]
        fit_ok = not np.isnan(m)

        # Geometry
        LP = p["LP_m"] 
        LS = p["LS_m"] 
        spaz = p["spaz_m"] 

        # ============================================================
        # FIG 1 — Bz FIELD
        # ============================================================
        fig1, ax = plt.subplots(figsize=(8,5))
        indices = [0, int(len(rtot)/60), int(len(rtot)/50), int(len(rtot)/40),
                int(len(rtot)/30), int(len(rtot)/20), int(len(rtot)/10),
                int(len(rtot)/5), int(len(rtot)/2), len(rtot)-1]

        for idx in indices:
            ax.plot(ztot*1e3, Bz[:, idx], label=f"r={rtot[idx]*1e3:.1f} mm")
        
        ax.axvline((0 + LP/2)*1e3, color='darkgreen', linestyle='--', linewidth=2, label='Primary')
        ax.axvline((0 - LP/2)*1e3, color='darkgreen', linestyle='--', linewidth=2)
        ax.axvspan((0 + LP/2)*1e3, (0 - LP/2)*1e3, color='darkgreen', alpha=0.2)

        ax.axvline(spaz/2*1e3,          color='midnightblue', linestyle='--', linewidth=2, label='Secondary')
        ax.axvline((LS + spaz/2)*1e3,   color='midnightblue', linestyle='--', linewidth=2)
        ax.axvspan(spaz/2*1e3, (LS + spaz/2)*1e3, color='midnightblue', alpha=0.2)

        ax.axvline(-spaz/2*1e3,         color='midnightblue', linestyle='--', linewidth=2)
        ax.axvline(-(LS + spaz/2)*1e3,  color='midnightblue', linestyle='--', linewidth=2)
        ax.axvspan(-(LS + spaz/2)*1e3, -spaz/2*1e3, color='midnightblue', alpha=0.2)

        ax.set_xlabel("z [mm]")
        ax.set_ylabel("Bz [T]")
        ax.set_title("Bz field")
        ax.grid()
        ax.legend()
        figs.append(fig1)

        # ============================================================
        # FIG 2 — FLUX
        # ============================================================
        fig2, ax = plt.subplots(figsize=(8,5))
        ax.plot(dz_shift*1e3, flux_S1, color='blue', label='Flux S1')
        ax.plot(dz_shift*1e3, flux_S2, color='black', label='Flux S2')
        ax.set_xlabel("Δz [mm]")
        ax.set_ylabel("Flux [T·m²]")
        ax.set_title("Flux in secondaries")
        ax.grid()
        ax.legend()
        figs.append(fig2)

        # ============================================================
        # FIG 3 — EFM
        # ============================================================
        fig3, ax = plt.subplots(figsize=(8,5))
        ax.plot(dz_shift*1e3, efm1, color='blue', label='efm1')
        ax.plot(dz_shift*1e3, efm2, color='black', label='efm2')
        ax.set_xlabel("Δz [mm]")
        ax.set_ylabel("efm [V]")
        ax.set_title("EMF in secondaries")
        ax.grid()
        ax.legend()
        figs.append(fig3)

        # ============================================================
        # FIG 4 — ΔEMF + FIT
        # ============================================================
        fig4, ax = plt.subplots(figsize=(8,5))
        ax.plot(dz_shift*1e3, deltaefm, color='black', label='Simulation')

        if fit_ok:
            x_line = np.linspace(-lin, lin, 200)
            y_line = m * x_line + q
            ax.plot(x_line*1e3, y_line, color='red', lw=2,
                    label=f"Fit: m={m:.3e} V/m, q={q:.3e} V")
            ax.axvline(lin*1e3, color="red", linestyle="--", label="Fitted linear range")
            ax.axvline(-lin*1e3, color="red", linestyle="--")
        ax.set_xlabel("Δz [mm]")
        ax.set_ylabel("Δemf [V]")
        ax.set_title("Differential emf and linear fit")
        ax.grid()
        ax.legend()
        figs.append(fig4)

        # ============================================================
        # FIG 5 — ΔEMF CORRECTED
        # ============================================================
        fig5, ax = plt.subplots(figsize=(8,5))
        ax.plot(dz_shift*1e3, deltaefm_readout, color='black', label='Simulation')

        if fit_ok:
            y_line_corr = m_corr * x_line + q_corr
            ax.plot(x_line*1e3, y_line_corr, color='red', lw=2,
                    label=f"Fit corrected: m={m_corr:.3e} V/m, q={q_corr:.3e} V")
            ax.axvline(lin*1e3, color="red", linestyle="--", label="Fitted linear range")
            ax.axvline(-lin*1e3, color="red", linestyle="--")
        ax.set_xlabel("Δz [mm]")
        ax.set_ylabel("Δ emf corrected [V]")
        ax.set_title("Δ emf corrected")
        ax.grid()
        ax.legend()
        figs.append(fig5)

        return figs

    def on_select_configuration(self, idx, array_name):
        self.selected_config = self._results[array_name][idx]
        self._selected_mode = True 
        # Salva AWG originali appena selezioni la configurazione
        self._original_dP_AWG = self.dP_AWG.value()
        self._original_dS_AWG = self.dS_AWG.value()

        # --- CREA E MOSTRA IL POPUP ---
        dlg = ProgressDialog("Computing selected configuration...", self)
        dlg.show()
        QApplication.processEvents()

        # useremo direttamente la barra e la label del dialog
        bar = dlg.bar
        label = dlg.label

        # --- PREPARA PARAMETRI ---
        p_gui = self.get_params()
        RP, RS, LP, LS, hP, hS, spaz, lin, sens, rtot, ztot = self.selected_config

        p = {
            "resolution_mm": p_gui["resolution_mm"],
            "I": p_gui["I"],
            "f0_kHz": p_gui["f0_kHz"],
            "iso": p_gui["iso"],
            "corr": p_gui["corr"],
            "dP_AWG": p_gui["dP_AWG"],
            "dS_AWG": p_gui["dS_AWG"],
            "RP_m": RP,
            "RS_m": RS,
            "LP_m": LP,
            "LS_m": LS,
            "hP_m": hP,
            "hS_m": hS,
            "spaz_m": spaz
        }

        # --- CALCOLO (aggiorna popup dentro compute_from_params) ---
        self.compute_from_params(p, bar=bar, label=label)
        self._original_selected_results = copy.deepcopy(self._selected_results)

        # --- CHIUDI POPUP ---
        dlg.close()

        # --- PULIZIA SCHERMATA ---
        self.input_panel.hide()
        self.results_panel.hide()

        # --- CREA I BOTTONI ---
        btn_results = QPushButton("Show results")
        btn_plots   = QPushButton("Show plots")
        btn_pdf     = QPushButton("Export PDF")
        self.btn_opt= QPushButton("Optimize")
        self.btn_back    = QPushButton("Back to configurations")
        self.btn_back_selected = QPushButton("Back to selected configuration")

        btn_results.clicked.connect(self.show_selected_results)
        btn_plots.clicked.connect(self.show_selected_plots)
        self.btn_opt.clicked.connect(self.open_optimize_dialog)
        btn_pdf.clicked.connect(self.export_selected_pdf)
        self.btn_back.clicked.connect(self.back_to_configurations)
        self.btn_back_selected.clicked.connect(self.cancel_optimization_and_return_selected)
        self.btn_back_selected.hide()  # nascosto finché non ottimizzi

        # --- CREA IL PANNELLO SELEZIONE ---
        self.selected_panel = QWidget()
        layout = QVBoxLayout(self.selected_panel)
        layout.addWidget(btn_results)
        layout.addWidget(btn_plots)
        layout.addWidget(btn_pdf)
        layout.addWidget(self.btn_opt)
        layout.addWidget(self.btn_back)
        layout.addWidget(self.btn_back_selected)

        self.left_container.layout().addWidget(self.selected_panel)

        # --- AGGIORNA LE VISTE ---
        self.update_views_from_selected()

    def update_views_from_selected(self):
        RP, RS, LP, LS, hP, hS, spaz, lin, sens, rtot, ztot = self.selected_config

        geom = {
            "RP": RP, "RS": RS, "LP": LP, "LS": LS,
            "hP": hP, "hS": hS, "spaz": spaz,
            "rtot": rtot, "ztot": ztot
        }

        # 3D
        self.ax3d.clear()
        self.render_schema_3d_mob(self.ax3d)
        self.canvas_3d.draw()

        # Front
        self.ax_front.clear()
        self.render_front_view(self.ax_front)
        self.canvas_front.draw()

        # Side
        self.ax_side.clear()
        self.render_side_view(self.ax_side)
        self.canvas_side.draw()

    def back_to_configurations(self):
        self._selected_mode = False

        # Nascondi il pannello della configurazione selezionata
        if hasattr(self, "selected_panel"):
            self.selected_panel.hide()

        # Mostra di nuovo la schermata con i due bottoni
        self.results_panel.show()

    def show_selected_results(self):
        if not self._selected_results:
            dlg = QDialog(self)
            dlg.setWindowTitle("No results available")
            layout = QVBoxLayout(dlg)
            layout.addWidget(QLabel("Run the simulation first."))
            btn = QPushButton("Close")
            btn.clicked.connect(dlg.accept)
            layout.addWidget(btn)
            dlg.exec()
            return

        r = self._selected_results
        p = r["params"]

        # HTML RESULTS
        if self._optimization_mode == None:
            html = f"""
            <div style='color:white; font-size:14px;'>

            <h2>INPUT</h2>
            dP = {p['dP_AWG']} AWG ({r['dP_mm']:.3f} mm)<br>
            dS = {p['dS_AWG']} AWG ({r['dS_mm']:.3f} mm)<br>
            RP = {p['RP_m']*1e3:.3f} mm, RS = {p['RS_m']*1e3:.3f} mm<br>
            LP = {p['LP_m']*1e3:.3f} mm, LS = {p['LS_m']*1e3:.3f} mm<br>
            hP = {p['hP_m']*1e3:.3f} mm, hS = {p['hS_m']*1e3:.3f} mm<br>
            spaz = {p['spaz_m']*1e3:.3f} mm<br>
            I = {p['I']:.3f} A, f0 = {p['f0_kHz']:.3f} kHz<br>

            <h2>WINDINGS</h2>
            Primary: Np = {r['N_pri']} ({r['Nvert_pri']} × {r['Nlat_pri']})<br>
            Secondary: Ns = {r['N_sec']} ({r['Nvert_sec']} × {r['Nlat_sec']})<br>

            <h2>ELECTRICAL PARAMETERS</h2>
            R<sub>pri</sub> = {r['R_pri']:.6f} Ω<br>
            R<sub>sec</sub> = {r['R_sec']:.6f} Ω<br>
            L<sub>pri</sub> = {r['L_pri_uH']:.3f} µH<br>
            L<sub>sec</sub> = {r['L_sec_uH']:.3f} µH<br>

            <h2>SENSITIVITY</h2>
            Theoretical = {r['sens_t']:.6f} V/m<br>
            Fit = {abs(r['sens_fit']):.6f} V/m<br>
            Corrected = {abs(r['sens_fit_corr']):.6f} V/m<br>
            Linearity range = ±{r['linear_range']*1e3:.6f} mm<br>

            <h2>NOISE</h2>
            Voltage noise = {r['Noise_voltage'] if r['Noise_voltage'] else 'computed in run':.3e} V/sqrt(Hz)<br>
            Equivalent displacement = {r['EquivalentDisplacement'] if r['EquivalentDisplacement'] else 'computed in run':.3e} m/sqrt(Hz)<br>
            Equivalent displacement corrected = {r['EquivalentDisplacementcorr'] if r['EquivalentDisplacementcorr'] else 'computed in run':.3e} m/sqrt(Hz)<br>
            </div>
            """
        if self._optimization_mode == "primary":
            html = f"""
            <div style='color:white; font-size:14px;'>

            <h2>PRIMARY OPTIMIZED</h2>
            <h2>INPUT</h2>
            dP = {p['dP_AWG']} AWG ({r['dP_mm']:.3f} mm)<br>
            dS = {p['dS_AWG']} AWG ({r['dS_mm']:.3f} mm)<br>
            RP = {p['RP_m']*1e3:.3f} mm, RS = {p['RS_m']*1e3:.3f} mm<br>
            LP = {p['LP_m']*1e3:.3f} mm, LS = {p['LS_m']*1e3:.3f} mm<br>
            hP = {p['hP_m']*1e3:.3f} mm, hS = {p['hS_m']*1e3:.3f} mm<br>
            spaz = {p['spaz_m']*1e3:.3f} mm<br>
            I = {p['I']:.3f} A, f0 = {p['f0_kHz']:.3f} kHz<br>

            <h2>WINDINGS</h2>
            Primary: Np = {r['N_pri']} ({r['Nvert_pri']} × {r['Nlat_pri']})<br>
            Secondary: Ns = {r['N_sec']} ({r['Nvert_sec']} × {r['Nlat_sec']})<br>

            <h2>ELECTRICAL PARAMETERS</h2>
            R<sub>pri</sub> = {r['R_pri']:.6f} Ω<br>
            R<sub>sec</sub> = {r['R_sec']:.6f} Ω<br>
            L<sub>pri</sub> = {r['L_pri_uH']:.3f} µH<br>
            L<sub>sec</sub> = {r['L_sec_uH']:.3f} µH<br>

            <h2>SENSITIVITY</h2>
            Theoretical = {r['sens_t']:.6f} V/m<br>
            Fit = {abs(r['sens_fit']):.6f} V/m<br>
            Corrected = {abs(r['sens_fit_corr']):.6f} V/m<br>
            Linearity range = ±{r['linear_range']*1e3:.6f} mm<br>

            <h2>NOISE</h2>
            Voltage noise = {r['Noise_voltage'] if r['Noise_voltage'] else 'computed in run':.3e} V/sqrt(Hz)<br>
            Equivalent displacement = {r['EquivalentDisplacement'] if r['EquivalentDisplacement'] else 'computed in run':.3e} m/sqrt(Hz)<br>
            Equivalent displacement corrected = {r['EquivalentDisplacementcorr'] if r['EquivalentDisplacementcorr'] else 'computed in run':.3e} m/sqrt(Hz)<br>
            </div>
            """
        if self._optimization_mode == "secondary":
            html = f"""
            <div style='color:white; font-size:14px;'>

            <h2>SECONDARY OPTIMIZED</h2>
            <h2>INPUT</h2>
            dP = {p['dP_AWG']} AWG ({r['dP_mm']:.3f} mm)<br>
            dS = {p['dS_AWG']} AWG ({r['dS_mm']:.3f} mm)<br>
            RP = {p['RP_m']*1e3:.3f} mm, RS = {p['RS_m']*1e3:.3f} mm<br>
            LP = {p['LP_m']*1e3:.3f} mm, LS = {p['LS_m']*1e3:.3f} mm<br>
            hP = {p['hP_m']*1e3:.3f} mm, hS = {p['hS_m']*1e3:.3f} mm<br>
            spaz = {p['spaz_m']*1e3:.3f} mm<br>
            I = {p['I']:.3f} A, f0 = {p['f0_kHz']:.3f} kHz<br>

            <h2>WINDINGS</h2>
            Primary: Np = {r['N_pri']} ({r['Nvert_pri']} × {r['Nlat_pri']})<br>
            Secondary: Ns = {r['N_sec']} ({r['Nvert_sec']} × {r['Nlat_sec']})<br>

            <h2>ELECTRICAL PARAMETERS</h2>
            R<sub>pri</sub> = {r['R_pri']:.6f} Ω<br>
            R<sub>sec</sub> = {r['R_sec']:.6f} Ω<br>
            L<sub>pri</sub> = {r['L_pri_uH']:.3f} µH<br>
            L<sub>sec</sub> = {r['L_sec_uH']:.3f} µH<br>

            <h2>SENSITIVITY</h2>
            Theoretical = {r['sens_t']:.6f} V/m<br>
            Fit = {abs(r['sens_fit']):.6f} V/m<br>
            Corrected = {abs(r['sens_fit_corr']):.6f} V/m<br>
            Linearity range = ±{r['linear_range']*1e3:.6f} mm<br>

            <h2>NOISE</h2>
            Voltage noise = {r['Noise_voltage'] if r['Noise_voltage'] else 'computed in run':.3e} V/sqrt(Hz)<br>
            Equivalent displacement = {r['EquivalentDisplacement'] if r['EquivalentDisplacement'] else 'computed in run':.3e} m/sqrt(Hz)<br>
            Equivalent displacement corrected = {r['EquivalentDisplacementcorr'] if r['EquivalentDisplacementcorr'] else 'computed in run':.3e} m/sqrt(Hz)<br>
            </div>
            """
        if self._optimization_mode == "total":
            html = f"""
            <div style='color:white; font-size:14px;'>

            <h2>PRIMARY and SECONDARY OPTIMIZED</h2>
            <h2>INPUT</h2>
            dP = {p['dP_AWG']} AWG ({r['dP_mm']:.3f} mm)<br>
            dS = {p['dS_AWG']} AWG ({r['dS_mm']:.3f} mm)<br>
            RP = {p['RP_m']*1e3:.3f} mm, RS = {p['RS_m']*1e3:.3f} mm<br>
            LP = {p['LP_m']*1e3:.3f} mm, LS = {p['LS_m']*1e3:.3f} mm<br>
            hP = {p['hP_m']*1e3:.3f} mm, hS = {p['hS_m']*1e3:.3f} mm<br>
            spaz = {p['spaz_m']*1e3:.3f} mm<br>
            I = {p['I']:.3f} A, f0 = {p['f0_kHz']:.3f} kHz<br>

            <h2>WINDINGS</h2>
            Primary: Np = {r['N_pri']} ({r['Nvert_pri']} × {r['Nlat_pri']})<br>
            Secondary: Ns = {r['N_sec']} ({r['Nvert_sec']} × {r['Nlat_sec']})<br>

            <h2>ELECTRICAL PARAMETERS</h2>
            R<sub>pri</sub> = {r['R_pri']:.6f} Ω<br>
            R<sub>sec</sub> = {r['R_sec']:.6f} Ω<br>
            L<sub>pri</sub> = {r['L_pri_uH']:.3f} µH<br>
            L<sub>sec</sub> = {r['L_sec_uH']:.3f} µH<br>

            <h2>SENSITIVITY</h2>
            Theoretical = {r['sens_t']:.6f} V/m<br>
            Fit = {abs(r['sens_fit']):.6f} V/m<br>
            Corrected = {abs(r['sens_fit_corr']):.6f} V/m<br>
            Linearity range = ±{r['linear_range']*1e3:.6f} mm<br>

            <h2>NOISE</h2>
            Voltage noise = {r['Noise_voltage'] if r['Noise_voltage'] else 'computed in run':.3e} V/sqrt(Hz)<br>
            Equivalent displacement = {r['EquivalentDisplacement'] if r['EquivalentDisplacement'] else 'computed in run':.3e} m/sqrt(Hz)<br>
            Equivalent displacement corrected = {r['EquivalentDisplacementcorr'] if r['EquivalentDisplacementcorr'] else 'computed in run':.3e} m/sqrt(Hz)<br>
            </div>
            """

        # -------------------------
        # DIALOG
        # -------------------------
        dlg = QDialog(self)
        dlg.setWindowTitle("Simulation results")
        dlg.resize(600, 800)

        layout = QVBoxLayout(dlg)

        # SCROLL AREA
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        vbox = QVBoxLayout(container)

        # LABEL SELEZIONABILE
        label = QLabel(html)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)

        vbox.addWidget(label)
        scroll.setWidget(container)

        layout.addWidget(scroll)

        btn = QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)

        dlg.exec()

    def show_selected_plots(self):
        #print("Mostro i grafici della configurazione selezionata")
        if not self._selected_results:
            dlg = QDialog(self)
            dlg.setWindowTitle("No results available")
            layout = QVBoxLayout(dlg)
            layout.addWidget(QLabel("Run the simulation first."))
            btn = QPushButton("Close")
            btn.clicked.connect(dlg.accept)
            layout.addWidget(btn)
            dlg.exec()
            return

        r = self._selected_results
        p = r["params"]

        # Unpack
        ztot = r["ztot"]
        rtot = r["rtot"]
        Bz = r["Bz"]
        flux_S1 = r["flux_S1"]
        flux_S2 = r["flux_S2"]
        efm1 = r["efm1"]
        efm2 = r["efm2"]
        deltaefm = r["deltaefm"]
        deltaefm_readout = r["deltaefm_readout"]
        dz_shift = r["dz_shift"]

        # Fit results
        m = r["sens_fit"]
        q = r["q"]
        m_corr = r["sens_fit_corr"]
        q_corr = r["qcorr"]
        lin = r["linear_range"]
        fit_ok = not np.isnan(m)

        # Geometry
        LP = p["LP_m"] 
        LS = p["LS_m"] 
        spaz = p["spaz_m"]
        corr = p["corr"]

        zS1_center = spaz/2 + LS/2
        zS2_center = -(spaz/2 + LS/2)

        dlg = QDialog(self)
        dlg.setWindowTitle("Simulation plots")
        dlg.resize(900, 700)

        layout = QVBoxLayout(dlg)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Helper to create a tab with a figure
        def make_tab(fig):
            w = QWidget()
            v = QVBoxLayout(w)
            canvas = FigureCanvas(fig)
            v.addWidget(canvas)

            btn = QPushButton("Save PDF")
            def save_pdf():
                fname, _ = QFileDialog.getSaveFileName(self, "Save plot", "", "PDF Files (*.pdf)")
                if fname:
                    fig.savefig(fname, facecolor=fig.get_facecolor())
            btn.clicked.connect(save_pdf)
            v.addWidget(btn)

            return w
        
        figs = self._make_plot_figures()

        names = [
            "Bz field",
            "Flux",
            "EMF",
            "Δemf + fit",
            "Δemf corrected"
        ]

        for fig, name in zip(figs, names):
            tabs.addTab(make_tab(fig), name)
        
        dlg.exec()

    def export_selected_pdf(self):
        #print("Esporto PDF della configurazione selezionata")
        if not self._selected_results:
            dlg = QDialog(self)
            dlg.setWindowTitle("No results available")
            layout = QVBoxLayout(dlg)
            layout.addWidget(QLabel("Run the simulation first."))
            btn = QPushButton("Close")
            btn.clicked.connect(dlg.accept)
            layout.addWidget(btn)
            dlg.exec()
            return

        fname, _ = QFileDialog.getSaveFileName(
            self, "Export full report", "", "PDF Files (*.pdf)"
        )
        if not fname:
            return

        old_mode = self._view_mode

        with PdfPages(fname) as pdf:

            # ---------------------------------------------------------
            # PAGINA 1 — RISULTATI (testo)
            # ---------------------------------------------------------
            fig_txt = plt.figure(figsize=(8.27, 11.69))  # A4
            fig_txt.patch.set_facecolor("white")
            ax_txt = fig_txt.add_subplot(111)
            ax_txt.axis("off")

            txt = self._build_results_text()
            ax_txt.text(
                0.02, 0.98, txt,
                va="top", ha="left",
                fontsize=9, family="monospace"
            )

            pdf.savefig(fig_txt)
            plt.close(fig_txt)

            # ---------------------------------------------------------
            # PAGINE 2–4 — 3D, FRONT, SIDE
            # ---------------------------------------------------------
            #3d
            self._view_mode = "3d"
            self.render_current_view()
            self.canvas_3d.draw()
            pdf.savefig(
                self.canvas_3d.figure,
                facecolor="black",
                edgecolor="black"
            )

            # FRONT
            self._view_mode = "front"
            self.render_current_view()
            self.canvas_front.draw()
            pdf.savefig(
                self.canvas_front.figure,
                facecolor="black",
                edgecolor="black"
            )

            # SIDE
            self._view_mode = "side"
            self.render_current_view()
            self.canvas_side.draw()
            pdf.savefig(
                self.canvas_side.figure,
                facecolor="black",
                edgecolor="black"
            )


            # ---------------------------------------------------------
            # PAGINE SUCCESSIVE — TUTTI I PLOT
            # ---------------------------------------------------------
            figs = self._make_plot_figures()
            for f in figs:
                pdf.savefig(f)
                plt.close(f)

        # Ripristina la vista originale
        self._view_mode = old_mode
        self.render_current_view()
        self.canvas.draw()

    def _build_results_text(self):
        r = self._selected_results
        p = r["params"]

        lines = []
        
        if self._optimization_mode == None:
            lines.append("=== INPUT PARAMETERS (NOT OPTIMIZE)===")
        if self._optimization_mode == "primary":
            awg = self.dP_AWG.value()
            lines.append(f"=== INPUT PARAMETERS (PRIMARY OPTIMIZED (AWG {awg}))===")
        if self._optimization_mode == "secondary":
            awg = self.dS_AWG.value()
            lines.append(f"=== INPUT PARAMETERS (SECONDARY OPTIMIZED (AWG {awg}))===")
        if self._optimization_mode == "total":
            awgprim = self.dP_AWG.value()
            awgsec = self.dS_AWG.value()
            lines.append(f"=== INPUT PARAMETERS (PRIMARY AND SECONDARY OPTIMIZED (AWG {awgprim} - AWG {awgsec}))===")
        #lines.append("=== INPUT PARAMETERS (NOT OPTIMIZE)===")
        lines.append(f"RMAX                  : {self.RMAX.value():.3f} mm")
        lines.append(f"LMAX                  : {self.LMAX.value():.3f} mm")
        lines.append(f"dmax                  : {self.dmax.value():.3f} mm")
        lines.append(f"resolution            : {self.resolution.value():.3f} mm")
        lines.append(f"dP_AWG                : {p['dP_AWG']}  ({r['dP_mm']:.3f} mm)")
        lines.append(f"dS_AWG                : {p['dS_AWG']}  ({r['dS_mm']:.3f} mm)")
        lines.append(f"iso                   : {self.iso.value():.3f}")
        lines.append(f"I [A], f0 [kHz]       : {p['I']}, {p['f0_kHz']}")

        lines.append("")
        lines.append("=== CONFIGURATION SELECTED (NOT OPTIMIZE)===")
        lines.append(f"dP_AWG                : {p['dP_AWG']}  ({r['dP_mm']:.3f} mm)")
        lines.append(f"dS_AWG                : {p['dS_AWG']}  ({r['dS_mm']:.3f} mm)")
        lines.append(f"RP, RS [mm]           : {p['RP_m']*1e3:.3f}, {p['RS_m']*1e3:.3f}")
        lines.append(f"LP, LS [mm]           : {p['LP_m']*1e3:.3f}, {p['LS_m']*1e3:.3f}")
        lines.append(f"hP, hS [mm]           : {p['hP_m']*1e3:.3f}, {p['hS_m']*1e3:.3f}")
        lines.append(f"spacing [mm]          : {p['spaz_m']*1e3:.3f}")

        lines.append("")
        lines.append("=== WINDINGS ===")
        lines.append(f"N_pri                 : {r['N_pri']}  ({r['Nvert_pri']} × {r['Nlat_pri']})")
        lines.append(f"N_sec                 : {r['N_sec']}  ({r['Nvert_sec']} × {r['Nlat_sec']})")

        lines.append("")
        lines.append("=== ELECTRICAL PARAMETERS ===")
        lines.append(f"R_pri [Ω]             : {r['R_pri']:.6f}")
        lines.append(f"R_sec [Ω]             : {r['R_sec']:.6f}")
        lines.append(f"L_pri [µH]            : {r['L_pri_uH']:.3f}")
        lines.append(f"L_sec [µH]            : {r['L_sec_uH']:.3f}")

        lines.append("")
        lines.append("=== SENSITIVITY ===")
        lines.append(f"Theoretical [V/m]     : {r['sens_t']:.6f}")
        lines.append(f"Fit [V/m]             : {abs(r['sens_fit']):.6f}")
        lines.append(f"Corrected [V/m]       : {abs(r['sens_fit_corr']):.6f}")
        lines.append(f"Linearity range [mm]  : ±{r['linear_range']*1e3:.6f}")

        lines.append("")
        lines.append("=== NOISE ===")
        lines.append(f"Noise voltage [V/√Hz] : {r['Noise_voltage']}")
        lines.append(f"Eq. disp [m/√Hz]      : {r['EquivalentDisplacement']}")
        lines.append(f"Eq. disp corr [m/√Hz] : {r['EquivalentDisplacementcorr']}")

        return "\n".join(lines)
    
    ############
    ## ottimizzazione 
    ############
    def cancel_optimization_and_return_selected(self):
        # Ripristina AWG originale se esiste
        if self._original_dP_AWG is not None:
            self.dP_AWG.setValue(self._original_dP_AWG)
        if self._original_dS_AWG is not None:
            self.dS_AWG.setValue(self._original_dS_AWG)

         # Ripristina anche i RISULTATI originali
        if hasattr(self, "_original_selected_results"):
            self._selected_results = copy.deepcopy(self._original_selected_results)

        # Reset stato ottimizzazione
        self._original_dP_AWG = None
        self._original_dS_AWG = None
        self._optimization_mode = None
        
        # Nascondi bottone back to selected configuration
        self.btn_back_selected.hide()
        self.btn_back.show()
        self.btn_opt.show()

        # Ripristina titolo
        self.title_label.setText("Coaxial aligned – Design results")

        # Mostra pannello selezionato
        self.selected_panel.show()

        # Nascondi pannello configurazioni
        self.results_panel.hide()

        # Aggiorna viste con i valori originali
        self.update_views_from_selected()

    def open_optimize_dialog(self):
        #print("Apro il dialogo di ottimizzazione")  # debug

        dlg = OptimizeDialog(self)

        dlg.btn_primary.clicked.connect(lambda: (dlg.close(), self.optimize_primary()))
        dlg.btn_secondary.clicked.connect(lambda: (dlg.close(), self.optimize_secondary()))
        dlg.btn_total.clicked.connect(lambda: (dlg.close(), self.optimize_total()))
        dlg.btn_back.clicked.connect(dlg.close)

        dlg.exec()

    def optimize_primary(self):
        #print("ottimizzo primario")

        # Estrai parametri attuali
        RP, RS, LP, LS, hP, hS, spaz, lin, sens, rtot, ztot = self.selected_config
        p = self.get_params()

        # Calcolo ottimizzazione primario
        prim = dimensionamento_primario(
            p["V0"],
            p["Imax"],
            p["Pmax"],
            p["Rcabling"],
            RP*1e3,
            LP*1e3,
            hP*1e3,
            p["dP_AWG"],
            p["f0_kHz"],
            p["iso"]
        )

        if "error" in prim:
            dlg = QDialog(self)
            layout = QVBoxLayout(dlg)
            layout.addWidget(QLabel(prim["error"]))
            btn = QPushButton("Close")
            btn.clicked.connect(dlg.accept)
            layout.addWidget(btn)
            dlg.exec()
            return

        best = prim["best"]
        new_awg = best["awg"]

        # Aggiorna AWG primario nella GUI (temporaneamente)
        self.dP_AWG.setValue(new_awg)

        # Ricalcolo completo
        dlg = ProgressDialog("Recomputing with optimized primary...", self)
        dlg.show()
        QApplication.processEvents()

        bar = dlg.bar
        label = dlg.label

        p2 = {
            "resolution_mm": p["resolution_mm"],
            "I": p["I"],
            "f0_kHz": p["f0_kHz"],
            "iso": p["iso"],
            "corr": p["corr"],
            "dP_AWG": new_awg,
            "dS_AWG": p["dS_AWG"],
            "RP_m": RP,
            "RS_m": RS,
            "LP_m": LP,
            "LS_m": LS,
            "hP_m": hP,
            "hS_m": hS,
            "spaz_m": spaz
        }

        self.compute_from_params(p2, bar=bar, label=label)
        dlg.close()

        # Aggiorna viste
        self.update_views_from_selected()

        # Segna che abbiamo ottimizzato il primario
        self._optimization_mode = "primary"

        # Cambia titolo
        self.title_label.setText("Coaxial aligned – Design results (primary optimized)")

        # nascondi il bottone Optimize e back to configurations perché ora è inutile e mostra il back to selected configuration
        if hasattr(self, "btn_opt"):
            self.btn_opt.hide()
            self.btn_back.hide()
            self.btn_back_selected.show()

    def optimize_secondary(self):
        # Estrai parametri attuali
        RP, RS, LP, LS, hP, hS, spaz, lin, sens, rtot, ztot = self.selected_config
        p = self.get_params()

        # Calcolo ottimizzazione secondario
        sec = dimensionamento_secondario(
            p["f0_kHz"],
            p["e_n"],
            p["i_n"],
            RS*1e3,
            LS*1e3,
            hS*1e3,
            p["dS_AWG"],
            p["iso"],
            p["tipo_coll"]
        )

        if "error" in sec:
            dlg = QDialog(self)
            layout = QVBoxLayout(dlg)
            layout.addWidget(QLabel(sec["error"]))
            btn = QPushButton("Close")
            btn.clicked.connect(dlg.accept)
            layout.addWidget(btn)
            dlg.exec()
            return

        best = sec["best"]
        new_awg = best["awg"]

        # Aggiorna AWG secondario nella GUI (temporaneamente)
        self.dS_AWG.setValue(new_awg)

        # Ricalcolo completo
        dlg = ProgressDialog("Recomputing with optimized secondary...", self)
        dlg.show()
        QApplication.processEvents()

        bar = dlg.bar
        label = dlg.label

        p2 = {
            "resolution_mm": p["resolution_mm"],
            "I": p["I"],
            "f0_kHz": p["f0_kHz"],
            "iso": p["iso"],
            "corr": p["corr"],
            "dP_AWG": p["dP_AWG"],
            "dS_AWG": new_awg,
            "RP_m": RP,
            "RS_m": RS,
            "LP_m": LP,
            "LS_m": LS,
            "hP_m": hP,
            "hS_m": hS,
            "spaz_m": spaz
        }

        self.compute_from_params(p2, bar=bar, label=label)
        dlg.close()

        # Aggiorna viste
        self.update_views_from_selected()

        # Segna che abbiamo ottimizzato il primario
        self._optimization_mode = "secondary"

        # Cambia titolo
        self.title_label.setText("Coaxial aligned – Design results (secondary optimized)")

        # nascondi il bottone Optimize e back to configurations perché ora è inutile e mostra il back to selected configuration
        if hasattr(self, "btn_opt"):
            self.btn_opt.hide()
            self.btn_back.hide()
            self.btn_back_selected.show()

    def optimize_total(self):
        # Estrai parametri attuali
        RP, RS, LP, LS, hP, hS, spaz, lin, sens, rtot, ztot = self.selected_config
        p = self.get_params()
        
        # Calcolo ottimizzazione primario
        prim = dimensionamento_primario(
            p["V0"],
            p["Imax"],
            p["Pmax"],
            p["Rcabling"],
            RP*1e3,
            LP*1e3,
            hP*1e3,
            p["dP_AWG"],
            p["f0_kHz"],
            p["iso"]
        )

        if "error" in prim:
            dlg = QDialog(self)
            layout = QVBoxLayout(dlg)
            layout.addWidget(QLabel(prim["error"]))
            btn = QPushButton("Close")
            btn.clicked.connect(dlg.accept)
            layout.addWidget(btn)
            dlg.exec()
            return
        
        # Calcolo ottimizzazione secondario
        sec = dimensionamento_secondario(
            p["f0_kHz"],
            p["e_n"],
            p["i_n"],
            RS*1e3,
            LS*1e3,
            hS*1e3,
            p["dS_AWG"],
            p["iso"],
            p["tipo_coll"]
        )

        if "error" in sec:
            dlg = QDialog(self)
            layout = QVBoxLayout(dlg)
            layout.addWidget(QLabel(sec["error"]))
            btn = QPushButton("Close")
            btn.clicked.connect(dlg.accept)
            layout.addWidget(btn)
            dlg.exec()
            return
        
        bestprim = prim["best"]
        new_awgprim = bestprim["awg"]
        bestsec = sec["best"]
        new_awgsec = bestsec["awg"]

        # Aggiorna AWG nella GUI (temporaneamente)
        self.dP_AWG.setValue(new_awgprim)
        self.dS_AWG.setValue(new_awgsec)

        # Ricalcolo completo
        dlg = ProgressDialog("Recomputing with optimized secondary...", self)
        dlg.show()
        QApplication.processEvents()

        bar = dlg.bar
        label = dlg.label

        p2 = {
            "resolution_mm": p["resolution_mm"],
            "I": p["I"],
            "f0_kHz": p["f0_kHz"],
            "iso": p["iso"],
            "corr": p["corr"],
            "dP_AWG": new_awgprim,
            "dS_AWG": new_awgsec,
            "RP_m": RP,
            "RS_m": RS,
            "LP_m": LP,
            "LS_m": LS,
            "hP_m": hP,
            "hS_m": hS,
            "spaz_m": spaz
        }

        self.compute_from_params(p2, bar=bar, label=label)
        dlg.close()

        # Aggiorna viste
        self.update_views_from_selected()

        # Segna che abbiamo ottimizzato il primario
        self._optimization_mode = "total"

        # Cambia titolo
        self.title_label.setText("Coaxial aligned – Design results (primary and secondary optimized)")

        # nascondi il bottone Optimize e back to configurations perché ora è inutile e mostra il back to selected configuration
        if hasattr(self, "btn_opt"):
            self.btn_opt.hide()
            self.btn_back.hide()
            self.btn_back_selected.show()
            
