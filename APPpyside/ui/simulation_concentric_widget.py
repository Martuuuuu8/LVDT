import os
import time
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QFormLayout, QDoubleSpinBox, QSpinBox, QApplication, QSizePolicy,
    QDialog, QScrollArea, QGridLayout, QProgressBar, QFileDialog
)
from PySide6.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d


from function_mfieldsolenoidespire import Bz_solenoide
from function_fluxsolenoidespire import flusso_secondario


def cylinder_surface(R, z_min, z_max, n_theta=200, n_z=64):
    theta = np.linspace(0, 2*np.pi, n_theta)
    z = np.linspace(z_min, z_max, n_z)
    theta, z = np.meshgrid(theta, z)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    return x, y, z


class SimulationConcentricWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._param_descriptions = {
            "Resolution": "Simulation discretization step in millimeters.",
            "I": "Primary current in amperes.",
            "f0": "Excitation frequency in kHz.",
            "RP": "Primary mean radius (mm).",
            "hP": "Primary thickness (mm).",
            "LP": "Primary length (mm).",
            "dP_AWG": "Primary wire diameter (AWG).",
            "RS": "Secondary mean radius (mm).",
            "hS": "Secondary thickness (mm).",
            "LS": "Secondary length (mm).",
            "dS_AWG": "Secondary wire diameter (AWG).",
            "spaz": "Spacing between the closer ends of secondaries (mm).",
            "iso": "Insulation ratio.",
            "corr": "Offset correction in Volts.",
        }

        self._results = {}
        self._view_mode = "3d"  # "3d", "front", "side"

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(6)
        self.setLayout(self.layout)

        # FIGURE + CANVAS
        self.fig = Figure(figsize=(16, 10), dpi=120, facecolor='black')
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: black;")
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # TIMER PER REDRAW
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(150)
        self._update_timer.timeout.connect(self.render_current_view)

        # UI
        self.build_ui()
        self.setup_signals()
        self.render_current_view()

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------
    def build_ui(self):
        self.title_label = QLabel("Coaxial concentric – Simulation parameters")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: white;")

        title_hbox = QHBoxLayout()
        title_hbox.addWidget(self.title_label)
        title_hbox.addStretch()

        help_btn = QPushButton("?")
        help_btn.setFixedSize(28, 28)
        help_btn.clicked.connect(self.show_param_help_dialog)
        title_hbox.addWidget(help_btn)

        # FORM
        form = QFormLayout()
        form.setVerticalSpacing(4)
        form.setHorizontalSpacing(12)

        def spinD(value):
            s = QDoubleSpinBox()
            s.setDecimals(3)
            s.setRange(0, 1000000)
            s.setValue(value)
            return s

        self.resolution = spinD(0.1); form.addRow("Resolution [mm]", self.resolution)
        self.I = spinD(1.000); form.addRow("I [A]", self.I)
        self.f0 = spinD(46.500); form.addRow("f0 [kHz]", self.f0)

        self.RP = spinD(12.000); form.addRow("RP [mm]", self.RP)
        self.dP = QSpinBox(); self.dP.setRange(1, 40); self.dP.setValue(25); form.addRow("dP [AWG]", self.dP)
        self.LP = spinD(17.000); form.addRow("LP [mm]", self.LP)
        self.hP = spinD(1.800); form.addRow("hP [mm]", self.hP)

        self.RS = spinD(16.000); form.addRow("RS [mm]", self.RS)
        self.dS = QSpinBox(); self.dS.setRange(1, 40); self.dS.setValue(32); form.addRow("dS [AWG]", self.dS)
        self.LS = spinD(11.000); form.addRow("LS [mm]", self.LS)
        self.hS = spinD(1.000); form.addRow("hS [mm]", self.hS)

        self.iso = spinD(1.1); form.addRow("Insulation ratio", self.iso)
        self.spaz = spinD(1.000); form.addRow("spaz [mm]", self.spaz)
        self.corr = spinD(0.250); form.addRow("corr [V]", self.corr)

        # LEFT PANEL
        from PySide6.QtWidgets import QWidget as QW
        self.left_container = QW()
        left_stack = QVBoxLayout(self.left_container)

        self.input_panel = QW()
        input_layout = QVBoxLayout(self.input_panel)
        input_layout.addLayout(form)

        run_button = QPushButton("START SIMULATION")
        run_button.clicked.connect(self.start_simulation)
        input_layout.addWidget(run_button)

        self.results_panel = QW()
        results_layout = QVBoxLayout(self.results_panel)

        btn_results = QPushButton("Show results")
        btn_graphs = QPushButton("Show plots")
        btn_pdf = QPushButton("Export PDF")

        btn_results.clicked.connect(self.show_results_dialog)
        btn_graphs.clicked.connect(self.show_plots_dialog)
        btn_pdf.clicked.connect(self.export_pdf)

        for b in (btn_results, btn_graphs, btn_pdf):
            b.setStyleSheet("font-size: 16px; padding: 10px;")

        results_layout.addWidget(btn_results)
        results_layout.addWidget(btn_graphs)
        results_layout.addWidget(btn_pdf)
        results_layout.addStretch()

        self.results_panel.hide()

        left_stack.addWidget(self.input_panel)
        left_stack.addWidget(self.results_panel)
        self.left_container.setFixedWidth(360)

        # RIGHT PANEL
        right_container = QW()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Canvas
        right_layout.addWidget(self.canvas)

        # VIEW SWITCH BUTTONS
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

        # SAVE BUTTONS
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

        # MAIN LAYOUT
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
    def set_view_mode(self, mode: str):
        self._view_mode = mode
        self.render_current_view()

    def render_current_view(self):
        if self._view_mode == "3d":
            self.render_schema_3d()
        elif self._view_mode == "front":
            self.render_front_view()
        elif self._view_mode == "side":
            self.render_side_view()

    # ---------------------------------------------------------
    # Tooltips
    # ---------------------------------------------------------
    def add_param_tooltips(self):
        mapping = {
            "Resolution": self.resolution,
            "I": self.I,
            "f0": self.f0,
            "RP": self.RP,
            "dP_AWG": self.dP,
            "LP": self.LP,
            "hP": self.hP,
            "RS": self.RS,
            "dS_AWG": self.dS,
            "LS": self.LS,
            "hS": self.hS,
            "iso": self.iso,
            "spaz": self.spaz,
            "corr": self.corr,
        }
        for key, widget in mapping.items():
            widget.setToolTip(self._param_descriptions[key])

    # ---------------------------------------------------------
    # Help dialog
    # ---------------------------------------------------------
    def show_param_help_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Parameters and descriptions")
        dlg.resize(560, 420)

        layout = QVBoxLayout(dlg)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout(content)

        keys = [
            "Resolution","I","f0","RP","hP","LP","dP_AWG",
            "RS","hS","LS","dS_AWG","spaz","iso","corr"
        ]

        for i, k in enumerate(keys):
            name = QLabel(f"<b>{k}</b>")
            desc = QLabel(self._param_descriptions[k])
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #EEE;")
            grid.addWidget(name, i, 0)
            grid.addWidget(desc, i, 1)

        scroll.setWidget(content)
        layout.addWidget(scroll)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)

        dlg.exec()

    # ---------------------------------------------------------
    # Signals
    # ---------------------------------------------------------
    def setup_signals(self):
        widgets = [
            self.resolution, self.I, self.f0,
            self.RP, self.dP, self.LP, self.hP,
            self.RS, self.dS, self.LS, self.hS,
            self.iso, self.spaz, self.corr
        ]
        for w in widgets:
            w.valueChanged.connect(self._update_timer.start)

    # ---------------------------------------------------------
    # Params
    # ---------------------------------------------------------
    def get_params(self):
        return {
            "resolution_mm": self.resolution.value(),
            "I": self.I.value(),
            "f0_kHz": self.f0.value(),
            "RP_mm": self.RP.value(),
            "dP_AWG": self.dP.value(),
            "LP_mm": self.LP.value(),
            "hP_mm": self.hP.value(),
            "RS_mm": self.RS.value(),
            "dS_AWG": self.dS.value(),
            "LS_mm": self.LS.value(),
            "hS_mm": self.hS.value(),
            "iso": self.iso.value(),
            "spaz_mm": self.spaz.value(),
            "corr": self.corr.value(),
        }

    # ---------------------------------------------------------
    # START SIMULATION
    # ---------------------------------------------------------
    def start_simulation(self):
        self._update_timer.stop()
        self.render_current_view()

        dlg = QDialog(self)
        dlg.setWindowTitle("Running simulation...")
        dlg.setModal(True)
        dlg.resize(420, 110)

        v = QVBoxLayout(dlg)
        label = QLabel("Computing magnetic field and flux...")
        label.setAlignment(Qt.AlignCenter)

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)

        v.addWidget(label)
        v.addWidget(bar)

        dlg.show()
        QApplication.processEvents()

        try:
            self.run_simulation(bar, label)
        finally:
            dlg.close()

        self.show_results_panel()

    def show_input_panel(self):
        self.title_label.setText("Coaxial concentric – Simulation parameters")
        self.input_panel.show()
        self.results_panel.hide()

    def show_results_panel(self):
        self.title_label.setText("Coaxial concentric – Simulation results")
        self.input_panel.hide()
        self.results_panel.show()

    # ---------------------------------------------------------
    # 3D VIEW
    # ---------------------------------------------------------
    def render_schema_3d(self):
        params = self.get_params()
        RP = params["RP_mm"]; RS = params["RS_mm"]
        LP = params["LP_mm"]; LS = params["LS_mm"]
        hP = params["hP_mm"]; hS = params["hS_mm"]
        spaz = params["spaz_mm"]

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

        self.fig.clf()
        ax = self.fig.add_axes([0, 0, 1, 1], projection='3d', facecolor='black')
        ax.figure.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)
        ax.set_title("3D view", color='white')
        
        color_primary = '#00FF88'
        color_secondary = '#66CCFF'
        sec_alpha = 0.6

        # SECONDARY 1 (pieno: cilindro + cappi)
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

        # PRIMARY (pieno)
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
        ax.set_ylabel("x [mm]", color='white')
        ax.set_xlabel("z [mm]", color='white')
        ax.set_zlabel("y [mm]", color='white')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect([3, 1, 1])

        ax.view_init(elev=10, azim=110)
        #ax.set_axis_off()

        self.canvas.draw()

    # ---------------------------------------------------------
    # FRONT VIEW (radiale)
    # ---------------------------------------------------------
    def render_front_view(self):
        params = self.get_params()
        RP = params["RP_mm"]; RS = params["RS_mm"]
        hP = params["hP_mm"]; hS = params["hS_mm"]

        self.fig.clf()
        ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)
        ax.set_facecolor('black')
        ax.set_aspect('equal')

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
        ax.text((x1 + x2)/2, -2, "hP", color=text_color, ha='center')

        # hS
        y1 = -(RS - hS/2)
        y2 = -(RS + hS/2)
        ax.plot([0, 0], [y1, y2], color=text_color, lw=2)
        ax.text(2, (y1 + y2)/2, "hS", color=text_color, va='center')

        # -------------------------
        # LIMITS
        # -------------------------
        Rmax = max(RS + hS, RP + hP)
        ax.set_xlim(-Rmax*1.1, Rmax*1.1)
        ax.set_ylim(-Rmax*1.1, Rmax*1.1)

        ax.set_title("Front view (radial section)", color='white')
        ax.set_xlabel("x [mm]", color='white')
        ax.set_ylabel("y [mm]", color='white')
        ax.tick_params(colors='white')
        ax.get_tightbbox
        ax.grid(False)

        self.canvas.draw()

    # ---------------------------------------------------------
    # SIDE VIEW (longitudinale)
    # ---------------------------------------------------------
    def render_side_view(self):
        params = self.get_params()
        LP = params["LP_mm"]; LS = params["LS_mm"]; spaz = params["spaz_mm"]; RP = params["RP_mm"]; RS = params["RS_mm"]; hP = params["hP_mm"]; hS = params["hS_mm"]

        self.fig.clf()
        ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.15)
        ax.set_facecolor('black')

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
        ax.plot([-LP/2, LP/2], [RP + hP/2 + (spaz/2 + LS)*1.5/10.5, RP + hP/2 + (spaz/2 + LS)*1.5/10.5], color=text_color, lw=2)
        ax.text(0, RP + hP/2 + (spaz/2 + LS)*1.5/10.5*2, "LP", color=text_color, ha='center')

        # LS
        ax.plot([spaz/2, spaz/2 + LS], [-(RS + hS/2 + (spaz/2 + LS)*1.5/10.5), -(RS + hS/2 + (spaz/2 + LS)*1.5/10.5)], color=text_color, lw=2)
        ax.text(spaz/2 + LS/2, -(RS + hS/2 + (spaz/2 + LS)*1.5/10.5*4), "LS", color=text_color, ha='center')

        # spacing
        y_spaz = min(-(RS - hS/2 - (spaz/2 + LS)*1.5/10.5), -(RP - hP/2 - (spaz/2 + LS)*1.5/10.5))
        ax.plot([-spaz/2, spaz/2], [y_spaz, y_spaz], color=text_color, lw=2)
        ax.text(0, min(-(RS - hS/2 - (spaz/2 + LS)*1.5/10.5*2), -(RP - hP/2 - (spaz/2 + LS)*1.5/10.5*2)), "spaz", color=text_color, ha='center')
        
        # RP
        ax.plot([0, 0], [0, RP], color=text_color, lw=2)
        ax.text(- (spaz/2 + LS)*1.5/10.5*1.5, RP/2, "RP", color=text_color, ha='center')

        # RS
        ax.plot([(spaz/2 + LS/2), (spaz/2 + LS/2)], [-RS, 0], color=text_color, lw=2)
        ax.text((spaz/2 + LS/2 + max(RS + hS/2, RP + hP/2)*1.5/10.5*2), -RS/2, "RS", color=text_color, ha='center')
        
        # hP
        ax.plot([-LP/2 - max(RS + hS/2, RP + hP/2)*1.5/25.5, -LP/2 - max(RS + hS/2, RP + hP/2)*1.5/25.5], [RP - hP/2, RP + hP/2], color=text_color, lw=2)
        ax.text(-LP/2 - max(RS + hS/2, RP + hP/2)*1.5/25.5*3, RP, "hP", color=text_color, ha='center')

        # hS
        ax.plot([(spaz/2 + LS + max(RS + hS/2, RP + hP/2)*1.5/25.5), (spaz/2 + LS + max(RS + hS/2, RP + hP/2)*1.5/25.5)], [- RS - hS/2, - RS + hS/2], color=text_color, lw=2)
        ax.text((spaz/2 + LS + max(RS + hS/2, RP + hP/2)*1.5/25.5*3), -RS, "hS", color=text_color, ha='center')

        ax.set_xlim(-(spaz/2 + LS)*1.5, (spaz/2 + LS)*1.5)
        ax.set_ylim(-max(RS + hS/2, RP + hP/2)*1.5, max(RS + hS/2, RP + hP/2)*1.5)

        ax.set_title("Front view (radial section)", color='white')
        ax.set_aspect('equal', 'box')
        ax.set_ylabel("y [mm]", color='white')
        ax.set_xlabel("z [mm]", color='white')
        ax.tick_params(colors='white')
        ax.get_tightbbox
        ax.grid(False)
        ax.set_title("Side view (longitudinal section)", color='white')

        self.canvas.draw()

    # ---------------------------------------------------------
    # SAVE PDF FUNCTIONS
    # ---------------------------------------------------------
    def save_3d_pdf(self):
        old_mode = self._view_mode
        self._view_mode = "3d"
        self.render_current_view()
        fname, _ = QFileDialog.getSaveFileName(self, "Save 3D view", "", "PDF Files (*.pdf)")
        if fname:
            self.fig.savefig(fname, facecolor=self.fig.get_facecolor())
        self._view_mode = old_mode
        self.render_current_view()

    def save_front_pdf(self):
        old_mode = self._view_mode
        self._view_mode = "front"
        self.render_current_view()
        fname, _ = QFileDialog.getSaveFileName(self, "Save Front view", "", "PDF Files (*.pdf)")
        if fname:
            self.fig.savefig(fname, facecolor=self.fig.get_facecolor())
        self._view_mode = old_mode
        self.render_current_view()

    def save_side_pdf(self):
        old_mode = self._view_mode
        self._view_mode = "side"
        self.render_current_view()
        fname, _ = QFileDialog.getSaveFileName(self, "Save Side view", "", "PDF Files (*.pdf)")
        if fname:
            self.fig.savefig(fname, facecolor=self.fig.get_facecolor())
        self._view_mode = old_mode
        self.render_current_view()

    # ---------------------------------------------------------
    # SIMULATION ENGINE (come versione semplificata precedente)
    # ---------------------------------------------------------
    def run_simulation(self, bar: QProgressBar, label: QLabel):
        p = self.get_params()

        resolution = p["resolution_mm"] * 1e-3
        I = p["I"]
        f0 = p["f0_kHz"]
        omega_0 = 2 * np.pi * f0 * 1e3

        RP = p["RP_mm"] * 1e-3
        LP = p["LP_mm"] * 1e-3
        hP = p["hP_mm"] * 1e-3

        RS = p["RS_mm"] * 1e-3
        LS = p["LS_mm"] * 1e-3
        hS = p["hS_mm"] * 1e-3

        iso = p["iso"]
        spaz = p["spaz_mm"] * 1e-3
        corr = p["corr"]

        dP_AWG = p["dP_AWG"]
        dS_AWG = p["dS_AWG"]

        dP = 0.127 * 92 ** ((36 - dP_AWG) / 39) * 1e-3
        dS = 0.127 * 92 ** ((36 - dS_AWG) / 39) * 1e-3

        bar.setValue(10)
        QApplication.processEvents()

        step = resolution
        x0, y0, z0 = 0, 0, 0
        rmin = 0
        rmax = max(RP + hP/2, RS + hS/2)
        zmin = -(spaz + 2*LS - LP/2)
        zmax = spaz + 2*LS - LP/2

        ztot = np.arange(zmin, zmax + step, step)
        rtot = np.arange(0, rmax + step, step)

        label.setText("Computing magnetic field Bz...")
        bar.setValue(20)
        QApplication.processEvents()

        Bz = Bz_solenoide(I, RP, dP, LP, hP, 0, 0, ztot, rtot, iso, step)

        Bz_smooth = np.zeros_like(Bz)
        for j in range(len(rtot)):
            Bz_smooth[:, j] = gaussian_filter1d(Bz[:, j], sigma=5)
        Bz = Bz_smooth

        bar.setValue(40)
        QApplication.processEvents()

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

        bar.setValue(60)
        QApplication.processEvents()

        dzmax = spaz/2 + LS - LP/2
        dz_shift = np.arange(-dzmax, dzmax + step, step)

        zS1_center = spaz/2 + LS/2
        #zS2_center = -(spaz/2 + LS/2)

        flux_S1 = np.zeros_like(dz_shift)
        #flux_S2 = np.zeros_like(dz_shift)

        for i, dzs in enumerate(dz_shift):
            zS1 = zS1_center + dzs
            #zS2 = zS2_center + dzs

            flux_S1[i] = flusso_secondario(Bz, ztot, rtot, RS, dS, LS, hS, 0, zS1, iso, step)
            #flux_S2[i] = flusso_secondario(Bz, ztot, rtot, RS, dS, LS, hS, 0, zS2, iso, step)

            if i % max(1, len(dz_shift)//10) == 0:
                bar.setValue(60 + int(20*i/len(dz_shift)))
                QApplication.processEvents()

        flux_S2 = flux_S1[::-1]
        bar.setValue(85)
        QApplication.processEvents()

        efm1 = -omega_0 * flux_S1
        efm2 = -omega_0 * flux_S2
        deltaefm = efm2 - efm1

        # ---------------------------------------------------------
        # FIT LOCALE + LINEARITÀ
        # ---------------------------------------------------------

        window = 5   # punti per il fit locale
        errors = []

        # 1) Trova il punto più lineare
        for k in range(window, len(dz_shift)-window):
            dz_loc = dz_shift[k-window:k+window]
            ef_loc = deltaefm[k-window:k+window]
            m_loc, q_loc = np.polyfit(dz_loc, ef_loc, 1)
            y_fit = m_loc * dz_loc + q_loc
            err = np.mean(np.abs(ef_loc - y_fit))
            errors.append(err)

        k_best = np.argmin(errors) + window
        dz0 = dz_shift[k_best]   # punto di massima linearità

        # 2) Fit locale nel punto migliore
        dz_loc = dz_shift[k_best-window:k_best+window]
        ef_loc = deltaefm[k_best-window:k_best+window]
        m_loc, q_loc = np.polyfit(dz_loc, ef_loc, 1)

        # 3) Soglia basata sulla pendenza (molto più fisica)
        eps = 0.025 * abs(m_loc) * np.max(np.abs(dz_shift))  # 2% della variazione attesa

        # 4) Espansione simmetrica finché i residui restano sotto soglia
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

        # 5) Fit finale nella zona lineare
        if len(dz_lin) >= 2:
            dz_min = dz_shift[i_left]
            dz_max = dz_shift[i_right]
            lin = min(abs(dz_min - dz0), abs(dz_max - dz0))  # semi-intervallo simmetrico
            mask_fit = (dz_shift >= -lin) & (dz_shift <= lin)
            x_fit = dz_shift[mask_fit]
            y_fit = deltaefm[mask_fit]
            m, q = np.polyfit(x_fit, y_fit, 1)
            fit_ok = True
        else:
            lin = 0.0
            m, q = np.nan, np.nan
            fit_ok = False

        # Sensibilità corretta
        m_corr = m * corr
        
        kT = 26e-3 * 1.6e-19
        NoiseResSec = np.sqrt(4 * kT * R_sec)
        DisplacementNoise = np.sqrt(2) * NoiseResSec
        EquivalentDisplacement = DisplacementNoise/abs(m)
        EquivalentDisplacementcorr = DisplacementNoise/abs(m_corr)

        idx_center = np.abs(zS1_center + dz_shift - (spaz/2 + LS/2)).argmin()
        dflux1 = np.gradient(flux_S1, step)
        sens_t = 2 * abs(dflux1[idx_center]) * omega_0

        deltaefm_readout = deltaefm * corr

        bar.setValue(100)
        QApplication.processEvents()
        time.sleep(0.1)

        self._results = {
            # -------------------------
            # INPUT
            # -------------------------
            "params": p,
            "dP_mm": dP * 1e3,   # in mm
            "dS_mm": dS * 1e3,   # in mm

            # -------------------------
            # GEOMETRY
            # -------------------------
            "ztot": ztot,
            "rtot": rtot,

            # -------------------------
            # MAGNETIC FIELD
            # -------------------------
            "Bz": Bz,

            # -------------------------
            # WINDINGS
            # -------------------------
            "N_pri": N_pri,
            "Nvert_pri": Nvert_pri,
            "Nlat_pri": Nlat_pri,
            "N_sec": N_sec,
            "Nvert_sec": Nvert_sec,
            "Nlat_sec": Nlat_sec,

            # -------------------------
            # ELECTRICAL PARAMETERS
            # -------------------------
            "R_pri": R_pri,
            "R_sec": R_sec,
            "L_pri_uH": L_pri_uH,
            "L_sec_uH": L_sec_uH,

            # -------------------------
            # FLUX AND EMF
            # -------------------------
            "flux_S1": flux_S1,
            "flux_S2": flux_S2,
            "efm1": efm1,
            "efm2": efm2,
            "deltaefm": deltaefm,
            "deltaefm_readout": deltaefm_readout,
            "dz_shift": dz_shift,

            # -------------------------
            # SENSITIVITY
            # -------------------------
            "sens_t": sens_t,          # teorica
            "sens_fit": m,             # dal fit
            "q": q,
            "qcorr": q*corr,
            "sens_fit_corr": m_corr,   # corretta
            "linear_range": lin,       # ±lin
            "linear_center": dz0,      # centro

            # -------------------------
            # NOISE
            # -------------------------
            "Noise_voltage": DisplacementNoise,  
            "EquivalentDisplacement": EquivalentDisplacement,
            "EquivalentDisplacementcorr": EquivalentDisplacementcorr  
        }
 
    def _make_plot_figures(self):
        r = self._results
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
        LP = p["LP_mm"] * 1e-3
        LS = p["LS_mm"] * 1e-3
        spaz = p["spaz_mm"] * 1e-3

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

    # ---------------------------------------------------------
    # RESULTS DIALOG
    # ---------------------------------------------------------
    def show_results_dialog(self):
        if not self._results:
            dlg = QDialog(self)
            dlg.setWindowTitle("No results available")
            layout = QVBoxLayout(dlg)
            layout.addWidget(QLabel("Run the simulation first."))
            btn = QPushButton("Close")
            btn.clicked.connect(dlg.accept)
            layout.addWidget(btn)
            dlg.exec()
            return

        r = self._results
        p = r["params"]

        # HTML RESULTS
        html = f"""
        <div style='color:white; font-size:14px;'>

        <h2>INPUT</h2>
        dP = {p['dP_AWG']} AWG ({r['dP_mm']:.3f} mm)<br>
        dS = {p['dS_AWG']} AWG ({r['dS_mm']:.3f} mm)<br>
        RP = {p['RP_mm']} mm, RS = {p['RS_mm']} mm<br>
        LP = {p['LP_mm']} mm, LS = {p['LS_mm']} mm<br>
        hP = {p['hP_mm']} mm, hS = {p['hS_mm']} mm<br>
        I = {p['I']} A, f0 = {p['f0_kHz']} kHz<br>
        spacing = {p['spaz_mm']} mm<br>

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
        Voltage noise = {r['Noise_voltage'] if r['Noise_voltage'] else 'computed in run'} V/sqrt(Hz)<br>
        Equivalent displacement = {r['EquivalentDisplacement'] if r['EquivalentDisplacement'] else 'computed in run'} m/sqrt(Hz)<br>
        Equivalent displacement corrected = {r['EquivalentDisplacementcorr'] if r['EquivalentDisplacementcorr'] else 'computed in run'} m/sqrt(Hz)<br>
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

    def show_plots_dialog(self):
        if not self._results:
            dlg = QDialog(self)
            dlg.setWindowTitle("No results available")
            layout = QVBoxLayout(dlg)
            layout.addWidget(QLabel("Run the simulation first."))
            btn = QPushButton("Close")
            btn.clicked.connect(dlg.accept)
            layout.addWidget(btn)
            dlg.exec()
            return

        r = self._results
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
        LP = p["LP_mm"] * 1e-3
        LS = p["LS_mm"] * 1e-3
        spaz = p["spaz_mm"] * 1e-3
        corr = p["corr"]

        zS1_center = spaz/2 + LS/2
        zS2_center = -(spaz/2 + LS/2)

        dlg = QDialog(self)
        dlg.setWindowTitle("Simulation plots")
        dlg.resize(900, 700)

        layout = QVBoxLayout(dlg)

        from PySide6.QtWidgets import QTabWidget
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

    def export_pdf(self):
        if not self._results:
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
            for mode in ["3d", "front", "side"]:
                self._view_mode = mode
                self.render_current_view()
                self.canvas.draw()
                pdf.savefig(self.fig)

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


    
        r = self._results
        p = r.get("params", {})

        def g(key, default=np.nan):
            return r.get(key, default)

        lines = []

        lines.append("=== INPUT PARAMETERS ===")
        for k, v in p.items():
            lines.append(f"{k:20s} : {v}")

        lines.append("")
        lines.append("=== WINDINGS ===")
        lines.append(f"{'Primary turns':20s} : {g('N_pri', 0)}")
        lines.append(f"{'Secondary turns':20s} : {g('N_sec', 0)}")
        lines.append(f"{'R_primary [Ω]':20s} : {g('R_pri', np.nan):.3f}")
        lines.append(f"{'R_secondary [Ω]':20s} : {g('R_sec', np.nan):.3f}")

        lines.append("")
        lines.append("=== INDUCTANCES ===")
        lines.append(f"{'L_primary [µH]':20s} : {g('L_pri_uH', np.nan):.3f}")
        lines.append(f"{'L_secondary [µH]':20s} : {g('L_sec_uH', np.nan):.3f}")

        lines.append("")
        lines.append("=== SENSITIVITY ===")
        lines.append(f"{'Theoretical [V/m]':20s} : {g('sens_t', np.nan):.6e}")
        lines.append(f"{'Fit [V/m]':20s} : {abs(g('sens_fit', np.nan)):.6e}")
        lines.append(f"{'Corrected [V/m]':20s} : {abs(g('sens_fit_corr', np.nan)):.6e}")
        lines.append(f"{'Linear range [m]':20s} : ±{g('linear_range', np.nan):.6e}")

        lines.append("")
        lines.append("=== NOISE ===")
        lines.append(f"{'NoiseResSec [V/√Hz]':20s} : {g('NoiseResSec', np.nan):.3e}")
        lines.append(f"{'Eq. disp [m/√Hz]':20s} : {g('EquivalentDisplacement', np.nan):.3e}")
        lines.append(f"{'Eq. disp corr [m/√Hz]':20s} : {g('EquivalentDisplacementcorr', np.nan):.3e}")

        return "\n".join(lines)

    def _build_results_text(self):
        r = self._results
        p = r["params"]

        lines = []

        lines.append("=== INPUT PARAMETERS ===")
        lines.append(f"dP_AWG                : {p['dP_AWG']}  ({r['dP_mm']:.3f} mm)")
        lines.append(f"dS_AWG                : {p['dS_AWG']}  ({r['dS_mm']:.3f} mm)")
        lines.append(f"RP, RS [mm]           : {p['RP_mm']}, {p['RS_mm']}")
        lines.append(f"LP, LS [mm]           : {p['LP_mm']}, {p['LS_mm']}")
        lines.append(f"hP, hS [mm]           : {p['hP_mm']}, {p['hS_mm']}")
        lines.append(f"I [A], f0 [kHz]       : {p['I']}, {p['f0_kHz']}")
        lines.append(f"spacing [mm]          : {p['spaz_mm']}")

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
