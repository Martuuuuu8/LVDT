# design_widget.py
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, Signal


class DesignWidget(QWidget):

    open_design_aligned = Signal()
    open_design_concentric = Signal()

    def __init__(self):
        super().__init__()

        self.assets_path = os.path.join(os.path.dirname(__file__), "..", "assets")

        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)
        self.setLayout(self.layout)

        self.show_type_selection()

    def show_type_selection(self):
        self.clear_layout()

        title = QLabel("DESIGN - Select LVDT Type")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 26px; font-weight: bold; margin-bottom: 30px;")

        # --- ALIGNED ---
        img_all = QLabel()
        img_all_path = os.path.join(self.assets_path, "lvdt_aligned.png")
        img_all.setPixmap(QPixmap(img_all_path).scaled(350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        img_all.setAlignment(Qt.AlignCenter)

        btn_all = QPushButton("Coaxial aligned")
        btn_all.setStyleSheet("font-size: 18px; padding: 12px;")
        btn_all.clicked.connect(self.open_design_aligned.emit)

        box_all = QVBoxLayout()
        box_all.setAlignment(Qt.AlignCenter)
        box_all.addWidget(img_all)
        box_all.addWidget(btn_all)

        # --- CONCENTRIC ---
        img_conc = QLabel()
        img_conc_path = os.path.join(self.assets_path, "lvdt_concentric.png")
        img_conc.setPixmap(QPixmap(img_conc_path).scaled(350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        img_conc.setAlignment(Qt.AlignCenter)

        btn_conc = QPushButton("Coaxial concentric")
        btn_conc.setStyleSheet("font-size: 18px; padding: 12px;")
        btn_conc.clicked.connect(self.open_design_concentric.emit)

        box_conc = QVBoxLayout()
        box_conc.setAlignment(Qt.AlignCenter)
        box_conc.addWidget(img_conc)
        box_conc.addWidget(btn_conc)

        # --- LAYOUT ORIZZONTALE ---
        h = QHBoxLayout()
        h.setAlignment(Qt.AlignCenter)
        h.setSpacing(80)
        h.addLayout(box_all)
        h.addLayout(box_conc)

        self.layout.addWidget(title)
        self.layout.addLayout(h)
        self.layout.addStretch()

    def clear_layout(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
