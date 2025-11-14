from PyQt5.QtGui import QValidator, QIntValidator, QIcon
from PyQt5.QtWidgets import QApplication, QLineEdit, QToolBar, QStatusBar, QLabel
from PyQt5 import uic
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = uic.loadUi(r"F:\海底探测\project1\单波束界面.ui")

    ui.show()

    sys.exit(app.exec())
