import numpy as numpy
import matplotlib.pyplot as plt
import cv2
import time
import sys
from PyQt5 import QtWidgets
from myui import Ui_MainWindow

class AppWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # connection btw event & function
        self.ui.pushButton11.clicked.connect(self.pushButton11_Click)
        self.show()
    def pushButton11_Click(self):
        img = cv2.imread('images/dog.bmp', cv2.IMREAD_COLOR)
        rows, cols, ch = img.shape
        print('Height = %d \nWidth = %d' % (rows, cols))
        cv2.imshow('My image', img); cv2.waitKey(5000); cv2.destroyAllWindows()

app = QtWidgets.QApplication(sys.argv)
window = AppWindow()
window.show()
sys.exit(app.exec_())