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
        self.actionHistogram_Grayscale.triggered.connect(self.histogram)  # Connect histogram action
        self.actionHistogram_RGB.triggered.connect(self.RGBHistogram)  # Connect RGB histogram action
        self.actionHistogram_Equalization.triggered.connect(self.EqualHistogramClicked)  # Connect equal histogram action
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

        # Connect Sobel edge detection action
        self.actionSobel.triggered.connect(self.sobelClicked)

        # Connect Canny edge detection action
        self.actionCanny.triggered.connect(self.cannyClicked)

        self.contrast_value = 1.6

    @pyqtSlot()
    def loadClicked(self):
        self.loadImage('../koala.jpeg')

    @pyqtSlot()
    def grayClicked(self):
        try:
            if self.image is not None:
                H, W = self.image.shape[:2]
                gray = np.zeros((H, W), np.uint8)
                for i in range(H):
                    for j in range(W):
                        gray[i, j] = np.clip(
                            0.299 * self.image[i, j, 0] + 0.587 * self.image[i, j, 1] + 0.114 * self.image[i, j, 2], 0,
                            255)
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
                # Apply contrast enhancement
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

                # Plot histogram
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
                self.displayImage(2)  # Display the translated image in the second label
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def rotasi(self, degree):
        try:
            if self.image is not None:
                h, w = self.image.shape[:2]
                rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 0.7)
                cos = np.abs(rotationMatrix[0, 0])
                sin = np.abs(rotationMatrix[0, 1])
                nW = int((h * sin) + (w * cos))
                nH = int((h * cos) + (w * sin))
                rotationMatrix[0, 2] += (nW / 2) - w / 2
                rotationMatrix[1, 2] += (nH / 2) - h / 2
                rotated_image = cv2.warpAffine(self.image, rotationMatrix, (w, h))
                self.image = rotated_image
                self.displayImage(2)  # Display the rotated image in the second label
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def zoomIn(self):
        try:
            if self.image is not None:
                self.image = cv2.resize(self.image, None, fx=1.2, fy=1.2)
                self.displayImage(2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def zoomOut(self):
        try:
            if self.image is not None:
                self.image = cv2.resize(self.image, None, fx=0.8, fy=0.8)
                self.displayImage(2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
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
                self.displayImage(2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def cropImage(self):
        try:
            if self.image is not None:
                self.image = self.image[10:500, 500:1000]
                self.displayImage(2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8
        if len(self.image.shape) == 3:  # rows[0], cols[1], channels[2]
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

    def loadImage(self, fname):
        self.image = cv2.imread(fname)
        self.displayImage(1)

    @pyqtSlot()
    def sobelClicked(self):
        try:
            if self.image is not None:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
                sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
                sobelX = np.uint8(np.absolute(sobelX))
                sobelY = np.uint8(np.absolute(sobelY))
                sobelCombined = cv2.bitwise_or(sobelX, sobelY)
                self.image = sobelCombined
                self.displayImage(2)  # Display the Sobel edge detected image in the second label

                plt.imshow(self.image, cmap='gray', interpolation='bicubic')
                plt.title('Sobel Edge Detection')
                plt.axis('off')
                plt.show()
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during Sobel edge detection: {str(e)}")

    @pyqtSlot()
    def cannyClicked(self):
        try:
            if self.image is not None:
                # Step 1: Noise reduction using Gaussian Blur
                blurred_image = cv2.GaussianBlur(self.image, (5, 5), 1.4)

                # Step 2: Gradient calculation using Sobel operator
                gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
                Gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
                Gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
                gradient_direction = np.arctan2(Gy, Gx)

                # Step 3: Non-maximum suppression
                H, W = gradient_magnitude.shape
                suppressed_image = np.zeros((H, W), dtype=np.uint8)
                angle = gradient_direction * 180. / np.pi
                angle[angle < 0] += 180

                for i in range(1, H - 1):
                    for j in range(1, W - 1):
                        try:
                            q = 255
                            r = 255
                            # angle 0
                            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                                q = gradient_magnitude[i, j + 1]
                                r = gradient_magnitude[i, j - 1]
                            # angle 45
                            elif (22.5 <= angle[i, j] < 67.5):
                                q = gradient_magnitude[i + 1, j - 1]
                                r = gradient_magnitude[i - 1, j + 1]
                            # angle 90
                            elif (67.5 <= angle[i, j] < 112.5):
                                q = gradient_magnitude[i + 1, j]
                                r = gradient_magnitude[i - 1, j]
                            # angle 135
                            elif (112.5 <= angle[i, j] < 157.5):
                                q = gradient_magnitude[i - 1, j - 1]
                                r = gradient_magnitude[i + 1, j + 1]

                            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                                suppressed_image[i, j] = gradient_magnitude[i, j]
                            else:
                                suppressed_image[i, j] = 0

                        except IndexError as e:
                            pass

                # Step 4: Hysteresis Thresholding
                high_threshold = 150
                low_threshold = 100
                strong = 255
                weak = 100

                result = np.zeros_like(suppressed_image, dtype=np.uint8)

                strong_i, strong_j = np.where(suppressed_image >= high_threshold)
                weak_i, weak_j = np.where((suppressed_image <= high_threshold) & (suppressed_image >= low_threshold))

                result[strong_i, strong_j] = strong
                result[weak_i, weak_j] = weak

                H, W = suppressed_image.shape
                for i in range(1, H - 1):
                    for j in range(1, W - 1):
                        if result[i, j] == weak:
                            if ((result[i + 1, j - 1:j + 2] == strong).any() or
                                    (result[i - 1, j - 1:j + 2] == strong).any() or
                                    (result[i, [j - 1, j + 1]] == strong).any()):
                                result[i, j] = strong
                            else:
                                result[i, j] = 0

                self.image = result
                self.displayImage(2)  # Display the Canny edge detected image

                plt.imshow(self.image, cmap='gray', interpolation='bicubic')
                plt.title('Canny Edge Detection')
                plt.axis('off')
                plt.show()
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during Canny edge detection: {str(e)}")


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('PyQt Image Processing')
window.show()
sys.exit(app.exec_())
