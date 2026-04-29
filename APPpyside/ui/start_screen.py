from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

class StartScreen(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Simulation and Design of LVDT")
        title.setStyleSheet("font-size: 26px; font-weight: bold; margin-bottom: 40px;")

        self.btn_sim = QPushButton("Simulation of an LVDT")
        self.btn_design = QPushButton("Design an LVDT")

        self.btn_sim.setStyleSheet("font-size: 18px; padding: 12px 20px;")
        self.btn_design.setStyleSheet("font-size: 18px; padding: 12px 20px;")

        layout.addWidget(title)
        layout.addWidget(self.btn_sim)
        layout.addWidget(self.btn_design)

        self.setLayout(layout)
