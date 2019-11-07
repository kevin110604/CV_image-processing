import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import struct
import random
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
        # Connection btw event & function
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
        self.ui.pushButton51.clicked.connect(self.pushButton51_Click)
        self.ui.pushButton52.clicked.connect(self.pushButton52_Click)
        self.ui.pushButton53.clicked.connect(self.pushButton53_Click)
        self.ui.pushButton54.clicked.connect(self.pushButton54_Click)
        self.ui.pushButton55.clicked.connect(self.pushButton55_Click)
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
    def pushButton51_Click(self):
        self.popup = AppPopup(2)
        self.popup.show_train_img()
    def pushButton52_Click(self):
        print('hyperparameters:\n', 'batch size: 32\n', 'learning rate: 0.001\n', 'optimizer: SGD\n')
    def pushButton53_Click(self):
        epoch_one_loss = [2.309455156326294, 2.3062610626220703, 2.311551570892334, 2.315929412841797, 2.2834324836730957, 2.2850048542022705, 2.3059277534484863, 2.2876811027526855, 2.2965619564056396, 2.2940359115600586, 2.295682191848755, 2.303783893585205, 2.290196418762207, 2.30722713470459, 2.2812552452087402, 2.2965731620788574, 2.3022544384002686, 2.289139747619629, 2.304690361022949, 2.2859506607055664, 2.2633657455444336, 2.286907434463501, 2.2947187423706055, 2.275141477584839, 2.301710844039917, 2.2854249477386475, 2.2925031185150146, 2.2961442470550537, 2.2831716537475586, 2.292861223220825, 2.2766053676605225, 2.272902727127075, 2.28235125541687, 2.284325361251831, 2.281475067138672, 2.2834293842315674, 2.270029306411743, 2.286776304244995, 2.2946484088897705, 2.2710461616516113, 2.272761106491089, 2.264686346054077, 2.2894394397735596, 2.2731611728668213, 2.2476541996002197, 2.253422975540161, 2.26839280128479, 2.2898435592651367, 2.278592348098755, 2.256791830062866, 2.2679340839385986, 2.263166904449463, 2.2556607723236084, 2.2438607215881348, 2.2598164081573486, 2.271494150161743, 2.285444498062134, 2.246676445007324, 2.279439926147461, 2.273203134536743, 2.261751890182495, 2.2644765377044678, 2.2566354274749756, 2.2542269229888916, 2.279810905456543, 2.2486491203308105, 2.2207255363464355, 2.242825508117676, 2.237362861633301, 2.2534162998199463, 2.213463068008423, 2.2382867336273193, 2.2582690715789795, 2.240208148956299, 2.2261269092559814, 2.2483744621276855, 2.2314064502716064, 2.246098756790161, 2.22929048538208, 2.2364859580993652, 2.2492496967315674, 2.213142156600952, 2.2227938175201416, 2.2606594562530518, 2.1991512775421143, 2.2027816772460938, 2.1827316284179688, 2.18540358543396, 2.208200216293335, 2.1990227699279785, 2.206076145172119, 2.175732135772705, 2.17747163772583, 2.1957101821899414, 2.193697214126587, 2.14681077003479, 2.144404411315918, 2.1560916900634766, 2.140204429626465, 2.1605851650238037, 2.201467990875244, 2.189659595489502, 2.1366546154022217, 2.153390884399414, 2.151458501815796, 2.1380319595336914, 2.13814377784729, 2.142993211746216, 2.0918402671813965, 2.0699572563171387, 2.0815043449401855, 2.0996556282043457, 2.0166423320770264, 2.05517578125, 2.0604772567749023, 1.974403738975525, 1.943175196647644, 2.0825843811035156, 1.9514628648757935, 1.949338674545288, 1.981993556022644, 1.9919061660766602, 1.9655207395553589, 1.8776007890701294, 1.8532859086990356, 1.8786648511886597, 1.762534260749817, 1.7784790992736816, 1.6647239923477173, 1.676282286643982, 1.6260868310928345, 1.7585138082504272, 1.6866481304168701, 1.560402512550354, 1.7500180006027222, 1.5180578231811523, 1.5897576808929443, 1.4741287231445312, 1.561781644821167, 1.5610910654067993, 1.506373643875122, 1.4053910970687866, 1.4619410037994385, 1.4621139764785767, 1.6361875534057617, 1.3169102668762207, 1.2338606119155884, 1.3469363451004028, 1.296148419380188, 1.20128333568573, 1.1116793155670166, 1.2783265113830566, 1.1320661306381226, 0.8281052708625793, 1.119053840637207, 0.8666683435440063, 1.0954885482788086, 0.8763176798820496, 0.8308529853820801, 1.180712342262268, 0.9195947647094727, 0.7004604339599609, 1.0001665353775024, 1.2512567043304443, 0.8469095230102539, 0.7034301161766052, 0.7451775670051575, 0.7977430820465088, 0.6876488327980042, 0.8754898905754089, 0.8450217247009277, 0.7580893039703369, 0.878572940826416, 0.6266390085220337, 0.5611291527748108, 0.6620399355888367, 0.6565808057785034, 0.42689982056617737, 0.8784221410751343, 0.5873407125473022, 0.8570502400398254, 0.5374785661697388, 0.6382054686546326, 0.832611083984375, 0.5627081990242004, 0.7495489716529846, 0.604522705078125, 0.7980763912200928]
        plt.plot(np.arange(1, len(epoch_one_loss)+1), epoch_one_loss)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.title('Epoch[1/50]')
        plt.show()
    def pushButton54_Click(self):
        training_loss = [0.27801019916534425, 0.1734770780324936, 0.12968837683051826, 0.11358276097178459, 0.08707914708256721, 0.07974489688128233, 0.06884555751830339, 0.06906814941316843, 0.0585213128298521, 0.05707976774200797, 0.051379289422556755, 0.04646936908680946, 0.04473398478664458, 0.04289475942216814, 0.043769344283267855, 0.05045518733784556, 0.038426468900218606, 0.03774076570998877, 0.03762611748250201, 0.03744259775523096, 0.0373833829022944, 0.03380541289802641, 0.032416088775265965, 0.03376530309477821, 0.032799575563333926, 0.032588873401004824, 0.03072367932954803, 0.0316964941451326, 0.030654475525394083, 0.030864385814219714, 0.02960870247525163, 0.0316222756926436, 0.03202091944692657, 0.029201694820681585, 0.032917821321217344, 0.03093822462717071, 0.027923995987838134, 0.029558965652040206, 0.027755297914985568, 0.02804889254262671, 0.02715451499344781, 0.026969457214372234, 0.02525769136203453, 0.026984780780551956, 0.02628664475143887, 0.029654834644298536, 0.028099226865405218, 0.027231024555303157, 0.027102080394513905, 0.026225816565332936]
        training_accuracy =  [92.16, 94.85, 96.29, 96.56, 97.3, 97.64, 98.03, 97.97, 98.23, 98.13, 98.47, 98.59, 98.6, 98.66, 98.61, 98.46, 98.74, 98.79, 98.77, 98.75, 98.85, 98.87, 98.9, 98.87, 98.88, 98.91, 98.99, 98.93, 98.96, 98.91, 99.02, 98.95, 98.99, 99.05, 98.92, 99.0, 98.99, 99.03, 99.02, 99.07, 99.1, 99.05, 99.1, 99.04, 99.09, 99.09, 99.05, 99.02, 99.05, 99.08]
        plt.plot(np.arange(1, 51), training_loss, label='training')
        plt.xlabel('epoch')
        plt.title('Loss')
        plt.legend()
        plt.show()
        plt.plot(np.arange(1, 51), training_accuracy, label='training')
        plt.xlabel('epoch')
        plt.title('Accurarcy')
        plt.legend()
        plt.show()
    def pushButton55_Click(self):
        self.popup = AppPopup()
        index = int(self.ui.lineEdit551.text())
        self.popup.inference(index)
        self.popup.showgrayImg()

class AppPopup(QtWidgets.QWidget):
    def __init__(self, choice=0):
        super().__init__()
        self.initUI(choice)
        self.show()
    def initUI(self, choice):
        if choice == 0:
            self.label = QtWidgets.QLabel('', self)
        elif choice == 1:
            self.label = QtWidgets.QLabel('', self)
            self.sl = QtWidgets.QSlider(self)
            self.sl.setGeometry(QtCore.QRect(0, 0, 160, 22))
            self.sl.setMaximum(100)
            self.sl.setProperty('value', 0)
            self.sl.setOrientation(QtCore.Qt.Horizontal)
            self.sl.valueChanged.connect(self.blend) 
        elif choice == 2:
            self.label_pic = []
            self.label = []
            for i in range(10):
                self.label_pic.append(QtWidgets.QLabel('', self))
                self.label.append(QtWidgets.QLabel('', self))
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
    def show_train_img(self):
        imgs = read_idx('MNIST/train-images-idx3-ubyte')
        labels = read_idx('MNIST/train-labels-idx1-ubyte')
        for i in range(0, 10):
            index = random.randint(0, 60000)
            self.img = imgs[index]
            # Change opencv's image to Qimage
            height, width = self.img.shape
            self.qImg = QImage(self.img.data, width, height, width, QImage.Format_Grayscale8).rgbSwapped()
            # Show Qimage
            self.label_pic[i].setGeometry(i*30, 0, width, height)
            self.label_pic[i].setPixmap(QPixmap.fromImage(self.qImg))
            # Set label
            self.label[i].setGeometry(i*30, 20, width, height)
            self.label[i].setText(str(labels[index]))
    def inference(self, index):
        imgs = read_idx('MNIST/t10k-images-idx3-ubyte')
        self.img = imgs[index]

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())
