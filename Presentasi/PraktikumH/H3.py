import cv2
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.uic import loadUi
import numpy as np


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('showimage.ui', self)
        self.image = None
        self.loadButton.clicked.connect(self.loadClicked)
        self.grayButton.clicked.connect(self.grayClicked)
        self.segmentButton.clicked.connect(self.segmentImage)

    @pyqtSlot()
    def loadClicked(self):
        self.loadImage('shapes.png')

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
    def segmentImage(self):
        try:
            if self.image is not None:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                for contour in contours:
                    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                    cv2.drawContours(self.image, [approx], 0, (0, 255, 0), 5)

                    # Find the center of the contour
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                    else:
                        cx, cy = 0, 0

                    # Check if the contour is a rectangle or square
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(approx)
                        aspectRatio = float(w) / h
                        if 0.95 <= aspectRatio <= 1.05:
                            shape = "Square"
                        else:
                            shape = "Rectangle"
                    elif len(approx) == 3:
                        shape = "Triangle"
                    elif len(approx) == 5:
                        shape = "Pentagon"
                    else:
                        shape = "Circle"

                    cv2.putText(self.image, shape, (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                self.displayImage(windows=2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during segmentation: {str(e)}")

    def loadImage(self, filename):
        self.image = cv2.imread(filename)
        self.displayImage()

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
            self.processedImgLabel.setPixmap(QPixmap.fromImage(img))
            self.processedImgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.processedImgLabel.setScaledContents(True)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ShowImage()
    window.setWindowTitle('PyQt OpenCV Image Processing')
    window.show()
    sys.exit(app.exec_())
