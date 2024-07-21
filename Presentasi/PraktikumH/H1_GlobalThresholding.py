import cv2
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.uic import loadUi
import numpy as np
from matplotlib import pyplot as plt


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('../showimage.ui', self)
        self.image = None
        self.loadButton.clicked.connect(self.loadClicked)
        self.grayButton.clicked.connect(self.grayClicked)
        self.actionBrightness.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Stretching.triggered.connect(self.contrastStretching)
        self.actionNegative_Image.triggered.connect(self.negativeImage)
        self.actionBiner_Image.triggered.connect(self.binaryImage)
        self.actionHistogram_Grayscale.triggered.connect(self.histogram)
        self.actionHistogram_RGB.triggered.connect(self.RGBHistogram)
        self.actionHistogram_Equalization.triggered.connect(self.EqualHistogramClicked)
        self.actionTranslasi.triggered.connect(self.translasi)

        # Connect rotation actions
        self.actionRotasi_Minus_45.triggered.connect(lambda: self.rotasi(-45))
        self.actionRotasi_45.triggered.connect(lambda: self.rotasi(45))
        self.actionRotasi_Minus_90.triggered.connect(lambda: self.rotasi(-90))
        self.actionRotasi_90.triggered.connect(lambda: self.rotasi(90))
        self.actionRotasi_180.triggered.connect(lambda: self.rotasi(180))

        self.actionZoom_In.triggered.connect(self.zoomIn)
        self.actionZoom_Out.triggered.connect(self.zoomOut)
        self.actionSkewed_Image.triggered.connect(self.skewedImage)
        self.actionCrop.triggered.connect(self.cropImage)

        self.actionSobel.triggered.connect(self.sobelClicked)
        self.actionCanny.triggered.connect(self.cannyClicked)

        self.actionDilasi.triggered.connect(self.dilation)
        self.actionErosi.triggered.connect(self.erosion)
        self.actionOpening.triggered.connect(self.opening)
        self.actionClosing.triggered.connect(self.closing)

        # Connect thresholding actions
        self.actionBinary.triggered.connect(lambda: self.thresholding(cv2.THRESH_BINARY))
        self.actionBinary_Invers.triggered.connect(lambda: self.thresholding(cv2.THRESH_BINARY_INV))
        self.actionTrunc.triggered.connect(lambda: self.thresholding(cv2.THRESH_TRUNC))
        self.actionTo_Zero.triggered.connect(lambda: self.thresholding(cv2.THRESH_TOZERO))
        self.actionTo_Zero_Invers.triggered.connect(lambda: self.thresholding(cv2.THRESH_TOZERO_INV))

        self.contrast_value = 1.6

    @pyqtSlot()
    def loadClicked(self):
        self.loadImage('../koala.jpeg')

    @pyqtSlot()
    def grayClicked(self):
        try:
            if self.image is not None:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                self.image = gray
                self.displayImage(windows=2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during grayscale conversion: {str(e)}")

    @pyqtSlot()
    def brightness(self):
        try:
            if self.image is not None:
                brightness = 80
                self.image = np.clip(self.image.astype(int) + brightness, 0, 255).astype(np.uint8)
                self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def contrast(self):
        try:
            if self.image is not None:
                self.image = np.clip(self.image.astype(float) * self.contrast_value, 0, 255).astype(np.uint8)
                self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def contrastStretching(self):
        try:
            if self.image is not None:
                min_val = np.min(self.image)
                max_val = np.max(self.image)
                stretched_image = 255 * ((self.image - min_val) / (max_val - min_val))
                self.image = stretched_image.astype(np.uint8)
                self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def negativeImage(self):
        try:
            if self.image is not None:
                self.image = 255 - self.image
                self.displayImage(1)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during negative transformation: {str(e)}")

    @pyqtSlot()
    def binaryImage(self):
        try:
            if self.image is not None:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
                self.image = binary_image
                self.displayImage(1)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during binary transformation: {str(e)}")

    @pyqtSlot()
    def histogram(self):
        try:
            if self.image is not None:
                if len(self.image.shape) == 3:
                    gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                else:
                    gray_image = self.image

                self.image = gray_image
                self.displayImage(2)

                plt.hist(gray_image.ravel(), 255, [0, 255])
                plt.title('Histogram of Grayscale Image')
                plt.xlabel('Pixel Values')
                plt.ylabel('Frequency')
                plt.show()
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during histogram plotting: {str(e)}")

    @pyqtSlot()
    def RGBHistogram(self):
        try:
            if self.image is not None:
                color = ('b', 'g', 'r')
                for i, col in enumerate(color):
                    histo = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                    plt.plot(histo, color=col)
                plt.xlim([0, 256])
                plt.title('Histogram of RGB Image')
                plt.xlabel('Pixel Values')
                plt.ylabel('Frequency')
                plt.show()
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during RGB histogram plotting: {str(e)}")

    @pyqtSlot()
    def EqualHistogramClicked(self):
        try:
            if self.image is not None:
                hist, bins = np.histogram(self.image.flatten(), 256, [0, 256])
                cdf = hist.cumsum()

                cdf_normalized = cdf * hist.max() / cdf.max()
                cdf_m = np.ma.masked_equal(cdf, 0)
                cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
                cdf = np.ma.filled(cdf_m, 0).astype('uint8')
                self.image = cdf[self.image]
                self.displayImage(2)

                plt.plot(cdf_normalized, color='b')
                plt.hist(self.image.flatten(), 256, [0, 256], color='r')
                plt.xlim([0, 256])
                plt.legend(('cdf', 'histogram'), loc='upper left')
                plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during histogram equalization: {str(e)}")

    @pyqtSlot()
    def translasi(self):
        try:
            if self.image is not None:
                h, w = self.image.shape[:2]
                quarter_h, quarter_w = h / 4, w / 4
                T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
                img_translated = cv2.warpAffine(self.image, T, (w, h))
                self.image = img_translated
                self.displayImage(2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def rotasi(self, degree):
        try:
            if self.image is not None:
                h, w = self.image.shape[:2]
                rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, .5)
                img_rotation = cv2.warpAffine(self.image, rotationMatrix, (w, h))
                self.image = img_rotation
                self.displayImage(2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def zoomIn(self):
        try:
            if self.image is not None:
                self.image = cv2.resize(self.image, None, fx=1.2, fy=1.2)
                self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def zoomOut(self):
        try:
            if self.image is not None:
                self.image = cv2.resize(self.image, None, fx=0.8, fy=0.8)
                self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def skewedImage(self):
        try:
            if self.image is not None:
                rows, cols, ch = self.image.shape
                pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
                pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(self.image, M, (cols, rows))
                self.image = dst
                self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def cropImage(self):
        try:
            if self.image is not None:
                self.image = self.image[10:200, 10:200]
                self.displayImage(2)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def displayImage(self, windows=1):
        try:
            qformat = QImage.Format_Indexed8
            if len(self.image.shape) == 3:
                if self.image.shape[2] == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888
            img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
            img = img.rgbSwapped()
            if windows == 1:
                self.imgLabel.setPixmap(QPixmap.fromImage(img))
                self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                self.imgLabel.setScaledContents(True)
            if windows == 2:
                self.hasilLabel.setPixmap(QPixmap.fromImage(img))
                self.hasilLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                self.hasilLabel.setScaledContents(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def loadImage(self, fname):
        try:
            self.image = cv2.imread(fname)
            self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def applyMorphologicalOperation(self, operation):
        try:
            if self.image is not None:
                kernel = np.ones((5, 5), np.uint8)
                if operation == 'dilation':
                    self.image = cv2.dilate(self.image, kernel, iterations=1)
                elif operation == 'erosion':
                    self.image = cv2.erode(self.image, kernel, iterations=1)
                elif operation == 'opening':
                    self.image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
                elif operation == 'closing':
                    self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
                self.displayImage(2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during {operation}: {str(e)}")

    @pyqtSlot()
    def dilation(self):
        self.applyMorphologicalOperation('dilation')

    @pyqtSlot()
    def erosion(self):
        self.applyMorphologicalOperation('erosion')

    @pyqtSlot()
    def opening(self):
        self.applyMorphologicalOperation('opening')

    @pyqtSlot()
    def closing(self):
        self.applyMorphologicalOperation('closing')

    def applyEdgeDetection(self, method):
        try:
            if self.image is not None:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                if method == 'sobel':
                    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
                    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
                    abs_grad_x = cv2.convertScaleAbs(grad_x)
                    abs_grad_y = cv2.convertScaleAbs(grad_y)
                    sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                    self.image = sobel
                elif method == 'canny':
                    self.image = cv2.Canny(gray, 100, 200)
                self.displayImage(2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during {method} edge detection: {str(e)}")

    @pyqtSlot()
    def sobelClicked(self):
        self.applyEdgeDetection('sobel')

    @pyqtSlot()
    def cannyClicked(self):
        self.applyEdgeDetection('canny')

    @pyqtSlot()
    def thresholding(self, method):
        try:
            if self.image is not None:
                if len(self.image.shape) == 3:
                    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.image

                _, thresholded = cv2.threshold(gray, 127, 255, method)
                self.image = thresholded
                self.displayImage(2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during thresholding: {str(e)}")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ShowImage()
    window.setWindowTitle('PyQt OpenCV Image Processing')
    window.show()
    sys.exit(app.exec_())
