import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from scipy import signal
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
        self.ui.pushButton21.clicked.connect(self.pushButton21_Click)
        self.ui.pushButton22.clicked.connect(self.pushButton22_Click)
        self.ui.pushButton31.clicked.connect(self.pushButton31_Click)
        self.ui.pushButton32.clicked.connect(self.pushButton32_Click)
        self.ui.pushButton41.clicked.connect(self.pushButton41_Click)
        self.ui.pushButton42.clicked.connect(self.pushButton42_Click)
        self.ui.pushButton43.clicked.connect(self.pushButton43_Click)
        self.ui.pushButton44.clicked.connect(self.pushButton44_Click)
        self.show()
    def pushButton11_Click(self):
        self.popup = AppPopup()
        self.popup.setGeometry(200, 200, 600, 600)
        self.popup.print_h_w()
        self.popup.showImg()
        self.popup.show()
    def pushButton12_Click(self):
        self.popup = AppPopup()
        self.popup.setGeometry(200, 200, 600, 600)
        self.popup.colorcvt()
        self.popup.showImg()
        self.popup.show()
    def pushButton13_Click(self):
        self.popup = AppPopup()
        self.popup.setGeometry(200, 200, 600, 600)
        self.popup.flip()
        self.popup.showImg()
        self.popup.show()
    def pushButton14_Click(self):
        self.popup = AppPopup(1)
        self.popup.setGeometry(200, 200, 600, 600)
        self.popup.blend()
        self.popup.show()
    def pushButton21_Click(self):
        self.popup = AppPopup()
        self.popup.setGeometry(200, 200, 600, 600)
        self.popup.globalthreshold()
        # self.popup.showImg()
        self.popup.show()
    def pushButton22_Click(self):
        self.popup = AppPopup()
        self.popup.setGeometry(200, 200, 600, 600)
        self.popup.localthreshold()
        # self.popup.showImg()
        self.popup.show()
    def pushButton31_Click(self):
        self.popup = AppPopup()
        self.popup.setGeometry(200, 200, 600, 600)
        angle = int(self.ui.lineEdit311.text())
        scale = float(self.ui.lineEdit312.text())
        Tx = int(self.ui.lineEdit313.text())
        Ty = int(self.ui.lineEdit314.text())
        self.popup.transformation(angle, scale, Tx, Ty)
        self.popup.showImg()
        self.popup.show()
    def pushButton32_Click(self):
        self.popup = AppPopup()
        self.popup.setGeometry(200, 200, 600, 600)
        self.popup.localthreshold()
        # self.popup.showImg()
        self.popup.show()
    def pushButton41_Click(self):
        self.popup = AppPopup()
        self.popup.setGeometry(200, 200, 600, 600)
        self.popup.gaussiansmooth()
        self.popup.showgrayImg()
        self.popup.show()
    def pushButton42_Click(self):
        self.popup = AppPopup()
        self.popup.setGeometry(200, 200, 600, 600)
        self.popup.sobelx()
        self.popup.showgrayImg()
        self.popup.show()
    def pushButton43_Click(self):
        self.popup = AppPopup()
        self.popup.setGeometry(200, 200, 600, 600)
        self.popup.sobely()
        self.popup.showgrayImg()
        self.popup.show()
    def pushButton44_Click(self):
        self.popup = AppPopup()
        self.popup.setGeometry(200, 200, 600, 600)
        self.popup.magnitude()
        self.popup.showgrayImg()
        self.popup.show()

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
        # Change opencv's image to Qimage
        height, width, channel = self.img.shape
        bytesPerLine = channel * width
        self.qImg = QImage(self.img.data, width, height, bytesPerLine,
                           QImage.Format_RGB888).rgbSwapped()
        # Show Qimage
        self.label.setGeometry(0, 0, width, height)
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
    def showgrayImg(self):
        # Change opencv's image to Qimage
        height, width = self.img.shape
        self.qImg = QImage(self.img.data, width, height, width, QImage.Format_Grayscale8).rgbSwapped()
        # Show Qimage
        self.label.setGeometry(0, 0, width, height)
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
    def print_h_w(self):
        self.img = cv2.imread('images/dog.bmp', cv2.IMREAD_COLOR)
        height, width, channel = self.img.shape
        print('Height = %d \nWidth = %d' % (height, width))
    def colorcvt(self): 
        self.img = cv2.imread('images/color.png', cv2.IMREAD_COLOR)
        b, g, r = cv2.split(self.img)
        self.img = cv2.merge((g, r, b))
    def flip(self): 
        self.img = cv2.imread('images/dog.bmp', cv2.IMREAD_COLOR)
        self.img = cv2.flip(self.img, 1)
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
    def globalthreshold(self):
        self.img = cv2.imread('images/QR.png', cv2.IMREAD_GRAYSCALE)
        ret, self.img = cv2.threshold(self.img, 80, 255, cv2.THRESH_BINARY)
        # Change opencv's image to Qimage
        height, width = self.img.shape
        bytesPerLine = width
        self.qImg = QImage(self.img.data, width, height, bytesPerLine,
                           QImage.Format_Grayscale8).rgbSwapped()
        # Show Qimage
        self.label.setGeometry(0, 0, width, height)
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
    def localthreshold(self):
        self.img = cv2.imread('images/QR.png', cv2.IMREAD_GRAYSCALE)
        self.img = cv2.adaptiveThreshold(self.img, 255, 
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 19, -1)
        # Change opencv's image to Qimage
        height, width = self.img.shape
        bytesPerLine = width
        self.qImg = QImage(self.img.data, width, height, bytesPerLine,
                           QImage.Format_Grayscale8).rgbSwapped()
        # Show Qimage
        self.label.setGeometry(0, 0, width, height)
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
    def transformation(self, angle, scale, Tx, Ty):
        self.img = cv2.imread('images/OriginalTransform.png', cv2.IMREAD_COLOR)
        h, w = self.img.shape[:2]
        # Translation
        M = np.float32([[1, 0, Tx], [0, 1, Ty]])
        self.img = cv2.warpAffine(self.img, M, (w, h))
        # Rotation
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        self.img = cv2.warpAffine(self.img, M, (w, h))
        # Scaling
        rh, rw = int(scale * h), int(scale * w)
        self.img = cv2.resize(self.img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    def gaussiansmooth(self):
        self.img = cv2.imread('images/School.jpg', cv2.IMREAD_COLOR)
        # R G B
        self.img = 0.299 * self.img[:, :, 2] + 0.587 * self.img[:, :, 1] + 0.114 * self.img[:, :, 0]
        # 3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x ** 2 + y ** 2))
        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        # Gaussian smooth
        self.img = signal.convolve2d(self.img, gaussian_kernel, mode='same', boundary='fill')
        self.img = self.img.astype(np.uint8)
    def sobelx(self):
        self.img = cv2.imread('images/School.jpg', cv2.IMREAD_COLOR)
        # R G B
        self.img = 0.299 * self.img[:, :, 2] + 0.587 * self.img[:, :, 1] + 0.114 * self.img[:, :, 0]
        # self.img = cv2.Sobel(self.img, cv2.CV_64F, 1, 0)
        # self.img = np.uint8(np.absolute(self.img))
        # 3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x ** 2 + y ** 2))
        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        # Gaussian smooth
        self.img = signal.convolve2d(self.img, gaussian_kernel, mode='same', boundary='fill')
        # Sobel Operator
        X = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
        # Sobel x
        self.img = signal.convolve2d(self.img, X, mode='same', boundary='fill')
        self.img = np.abs(self.img)
        self.img = self.img.astype(np.uint8)
    def sobely(self):
        self.img = cv2.imread('images/School.jpg', cv2.IMREAD_COLOR)
        # R G B
        self.img = 0.299 * self.img[:, :, 2] + 0.587 * self.img[:, :, 1] + 0.114 * self.img[:, :, 0]
        # 3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x ** 2 + y ** 2))
        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        # Gaussian smooth
        self.img = signal.convolve2d(self.img, gaussian_kernel, mode='same', boundary='fill')
        # Sobel Operator
        Y = np.array([[ 1,  2,  1],
                      [ 0,  0,  0],
                      [-1, -2, -1]])
        # Sobel y
        self.img = signal.convolve2d(self.img, Y, mode='same', boundary='fill')
        self.img = np.abs(self.img)
        self.img = self.img.astype(np.uint8)
    def magnitude(self):
        self.imgo = cv2.imread('images/School.jpg', cv2.IMREAD_COLOR)
        # R G B
        self.imgo = 0.299 * self.imgo[:, :, 2] + 0.587 * self.imgo[:, :, 1] + 0.114 * self.imgo[:, :, 0]
        # 3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x ** 2 + y ** 2))
        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        # Gaussian smooth
        self.imgo = signal.convolve2d(self.imgo, gaussian_kernel, mode='same', boundary='fill')
        # Sobel x
        X = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
        self.imgx = signal.convolve2d(self.imgo, X, mode='same', boundary='fill')
        # Sobel y
        Y = np.array([[ 1,  2,  1],
                      [ 0,  0,  0],
                      [-1, -2, -1]])
        self.imgy = signal.convolve2d(self.imgo, Y, mode='same', boundary='fill')
        self.img = np.sqrt(self.imgx ** 2 + self.imgy ** 2)
        self.img -= self.img.min()
        self.img *= 255.0 / self.img.max()
        self.img = self.img.astype(np.uint8)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())
