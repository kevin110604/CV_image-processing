import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from myui import Ui_MainWindow    # my own ui

class AppWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # connection btw event & function
        self.ui.pushButton11.clicked.connect(self.pushButton11_Click)
        self.ui.pushButton12.clicked.connect(self.pushButton12_Click)
        self.ui.pushButton13.clicked.connect(self.pushButton13_Click)
        self.ui.pushButton14.clicked.connect(self.pushButton14_Click)
        self.show()
    def pushButton11_Click(self):
        self.Popup = AppPopup()
        self.Popup.setGeometry(200, 200, 600, 600)
        self.Popup.showImg()
        self.Popup.show()
    def pushButton12_Click(self):
        self.popup = AppPopup()
        self.popup.setGeometry(200, 200, 600, 600)
        self.popup.colorcvt()
        self.popup.show()
    def pushButton13_Click(self):
        self.Popup = AppPopup()
        self.Popup.setGeometry(200, 200, 600, 600)
        self.Popup.flip()
        self.Popup.show()
    def pushButton14_Click(self):
        self.Popup = AppPopup(1)
        self.Popup.setGeometry(200, 200, 600, 600)
        self.Popup.blend()
        self.Popup.show()

class AppPopup(QtWidgets.QWidget):
    def __init__(self, choice=0):
        super().__init__()
        self.initUI(choice)
        self.show()
    def initUI(self, choice):
        self.label = QtWidgets.QLabel('', self)
        if choice == 1:
            self.sl = QtWidgets.QSlider(self)
            self.sl.setGeometry(QtCore.QRect(0, 0, 160, 22))
            self.sl.setMaximum(100)
            self.sl.setProperty('value', 0)
            self.sl.setOrientation(QtCore.Qt.Horizontal)
            self.sl.valueChanged.connect(self.blend) 
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
    def colorcvt(self): 
        self.img = cv2.imread('images/color.png', cv2.IMREAD_COLOR)
        b, g, r = cv2.split(self.img)
        self.img = cv2.merge((g, r, b))
        #self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2BRG)
        height, width, channel = self.img.shape
        # Change opencv's image to Qimage
        bytesPerLine = channel * width
        self.qImg = QImage(self.img.data, width, height, bytesPerLine,
                           QImage.Format_RGB888).rgbSwapped()
        # Show Qimage
        self.label.setGeometry(0, 0, width, height)
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
    def flip(self): 
        self.img = cv2.imread('images/dog.bmp', cv2.IMREAD_COLOR)
        self.img = cv2.flip(self.img, 1)
        # Change opencv's image to Qimage
        height, width, channel = self.img.shape
        bytesPerLine = channel * width
        self.qImg = QImage(self.img.data, width, height, bytesPerLine,
                           QImage.Format_RGB888).rgbSwapped()
        # Show Qimage
        self.label.setGeometry(0, 0, width, height)
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
    def blend(self):
        beta = self.sl.value() * 0.01
        alpha = 1 - beta
        self.img1 = cv2.imread('images/dog.bmp', cv2.IMREAD_COLOR)
        self.img2 = cv2.imread('images/dog.bmp', cv2.IMREAD_COLOR)
        self.img2 = cv2.flip(self.img2, 1)
        self.img = cv2.addWeighted(self.img1, alpha, self.img2, beta, 0)
        # Change opencv's image to Qimage
        height, width, channel = self.img.shape
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
