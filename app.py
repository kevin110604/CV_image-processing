import numpy as numpy
import matplotlib.pyplot as plt
import cv2
import sys
from PyQt5 import QtWidgets
from myui import Ui_MainWindow
from PyQt5.QtGui import QImage, QPixmap

class AppWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # connection btw event & function
        self.ui.pushButton11.clicked.connect(self.pushButton11_Click)
        self.show()
    def pushButton11_Click(self):
        self.exPopup = AppPopup()
        self.exPopup.setGeometry(200, 200, 600, 600)
        self.exPopup.showImg()
        self.exPopup.show()

class AppPopup(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.show()
    def initUI(self):
        self.label = QtWidgets.QLabel('', self)
    def showImg(self):
        self.img = cv2.imread('images/dog.bmp', cv2.IMREAD_COLOR)
        height, width, channel = self.img.shape
        print('Height = %d \nWidth = %d' % (height, width))
        # Change opencv's image to Qimage
        bytesPerLine = channel * width
        self.qImg = QImage(self.img.data, width, height, bytesPerLine,
                           QImage.Format_RGB888).rgbSwapped()
        # Show Qimage
        self.label.setGeometry(0, 0, width, height)
        self.label.setPixmap(QPixmap.fromImage(self.qImg))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())