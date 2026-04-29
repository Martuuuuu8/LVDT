# main.py
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QPushButton, QVBoxLayout, QHBoxLayout, QLabel
)
from PySide6.QtCore import Qt

from ui.simulation_widget import SimulationWidget
from ui.design_widget import DesignWidget
from ui.simulation_aligned_widget import SimulationAlignedWidget
from ui.simulation_concentric_widget import SimulationConcentricWidget
from ui.design_aligned_widget import DesignAlignedWidget
from ui.design_concentric_widget import DesignConcentricWidget


# ---------------------------------------------------------
# START SCREEN
# ---------------------------------------------------------
class StartScreen(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Simulation and Design of LVDT")
        title.setStyleSheet("font-size: 28px; font-weight: bold; margin-bottom: 40px;")

        self.btn_sim = QPushButton("Simulation of an LVDT")
        self.btn_design = QPushButton("Design an LVDT")

        for b in (self.btn_sim, self.btn_design):
            b.setStyleSheet("font-size: 20px; padding: 14px 24px;")

        layout.addWidget(title)
        layout.addWidget(self.btn_sim)
        layout.addWidget(self.btn_design)

        self.setLayout(layout)


# ---------------------------------------------------------
# MAIN WINDOW
# ---------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Simulation and Design of LVDT")
        self.resize(1400, 900)

        # ------------------------------
        # LAYOUT PRINCIPALE VERTICALE
        # ------------------------------
        self.main_layout = QVBoxLayout()

        # ------------------------------
        # BARRA SUPERIORE (menu + sidebar)
        # ------------------------------
        self.top_bar = QHBoxLayout()

        # MENU BUTTON
        self.btn_menu = QPushButton("☰")
        self.btn_menu.setFixedWidth(40)
        self.btn_menu.clicked.connect(self.toggle_sidebar)

        self.top_bar.addWidget(self.btn_menu)

        # SIDEBAR (inizialmente nascosta)
        self.sidebar_layout = QVBoxLayout()
        self.btn_sim_sidebar = QPushButton("Simulation")
        self.btn_design_sidebar = QPushButton("Design")

        for b in (self.btn_sim_sidebar, self.btn_design_sidebar):
            b.setStyleSheet("font-size: 16px; padding: 8px 12px;")

        self.sidebar_layout.addWidget(self.btn_sim_sidebar)
        self.sidebar_layout.addWidget(self.btn_design_sidebar)
        self.sidebar_layout.addStretch()

        self.sidebar_container = QWidget()
        self.sidebar_container.setLayout(self.sidebar_layout)
        self.sidebar_container.hide()

        self.top_bar.addWidget(self.sidebar_container)
        self.top_bar.addStretch()

        # ------------------------------
        # AREA CENTRALE
        # ------------------------------
        self.central_layout = QVBoxLayout()
        self.central_layout.setAlignment(Qt.AlignCenter)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.central_layout)

        # ------------------------------
        # ASSEMBLA TUTTO
        # ------------------------------
        self.main_layout.addLayout(self.top_bar)
        self.main_layout.addWidget(self.central_widget)

        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        # ------------------------------
        # START SCREEN
        # ------------------------------
        self.start_screen = StartScreen()
        self.central_layout.addWidget(self.start_screen)

        # Connessioni
        self.start_screen.btn_sim.clicked.connect(self.load_simulation)
        self.start_screen.btn_design.clicked.connect(self.load_design)

        self.btn_sim_sidebar.clicked.connect(self.load_simulation)
        self.btn_design_sidebar.clicked.connect(self.load_design)

        self.sidebar_visible = False

    # ---------------------------------------------------------
    # TOGGLE SIDEBAR
    # ---------------------------------------------------------
    def toggle_sidebar(self):
        if self.sidebar_visible:
            self.sidebar_container.hide()
        else:
            self.sidebar_container.show()

        self.sidebar_visible = not self.sidebar_visible

    # ---------------------------------------------------------
    # CAMBIA CONTENUTO CENTRALE
    # ---------------------------------------------------------
    def _set_central_widget(self, new_widget):
        for i in reversed(range(self.central_layout.count())):
            item = self.central_layout.itemAt(i)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        self.central_layout.addWidget(new_widget)

    # ---------------------------------------------------------
    # CARICA SIMULATION (schermata iniziale con le due immagini)
    # ---------------------------------------------------------
    def load_simulation(self):
        widget = SimulationWidget()
        widget.open_aligned.connect(self.load_simulation_aligned)
        widget.open_concentric.connect(self.load_simulation_concentric)
        self._set_central_widget(widget)

    # ---------------------------------------------------------
    # CARICA DESIGN (schermata iniziale con le due immagini)
    # ---------------------------------------------------------
    def load_design(self):
        widget = DesignWidget()
        widget.open_design_aligned.connect(self.load_design_aligned)
        widget.open_design_concentric.connect(self.load_design_concentric)
        self._set_central_widget(widget)

    # ---------------------------------------------------------
    # CARICA SIMULAZIONE COAXIAL ALIGNED
    # ---------------------------------------------------------
    def load_simulation_aligned(self):
        self._set_central_widget(SimulationAlignedWidget())

    # ---------------------------------------------------------
    # CARICA SIMULAZIONE COAXIAL CONCENTRIC
    # ---------------------------------------------------------
    def load_simulation_concentric(self):
        self._set_central_widget(SimulationConcentricWidget())

    # ---------------------------------------------------------
    # CARICA DESIGN COAXIAL ALIGNED
    # ---------------------------------------------------------
    def load_design_aligned(self):
        self._set_central_widget(DesignAlignedWidget())

    # ---------------------------------------------------------
    # CARICA DESIGN COAXIAL CONCENTRIC
    # ---------------------------------------------------------
    def load_design_concentric(self):
        self._set_central_widget(DesignConcentricWidget())


# ---------------------------------------------------------
# AVVIO APP
# ---------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
