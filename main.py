#!/usr/bin/python2
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import *

import numpy as np
import matplotlib.image as mpimg
import util

class Example(QMainWindow):

    def __init__(self):
        super(Example, self).__init__()
        self.initUI()

    def initUI(self):

        """
        Main Window global parameters
        """
        self.imgOriginal = np.array([])
        self.mainWidth = 1280
        self.mainHeight = 640
        self.main = QLabel()
        self.imgNpBefore = np.array([])
        self.imgNpAfter = np.array([])

        grid = QGridLayout()
        self.main.setLayout(grid)

        self.mainBefore = QLabel('Before')
        self.mainBefore.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.mainBefore.setWordWrap(True)
        self.mainBefore.setFont(QFont('Monospace', 10))

        self.mainAfter = QLabel('After')
        self.mainAfter.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.mainAfter.setWordWrap(True)
        self.mainAfter.setFont(QFont('Monospace', 10))

        grid.addWidget(self.mainBefore, 0, 0)
        grid.addWidget(self.mainAfter, 0, 1)

        """
        Menu Bar
        """

        # FILE MENU
        openFile = QAction('Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.showDialog)

        exitAction = QAction('Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)

        moveLeft = QAction('Move Left', self)
        moveLeft.setShortcut('Ctrl+L')
        moveLeft.triggered.connect(lambda: self.updateBefore(self.imgNpAfter))


        # PROCESS MENU
        equalizeMenu = QAction('Equalize', self)
        equalizeMenu.triggered.connect(lambda: self.updateImgAfter(util.equalize(self.imgNpBefore)))

        histogramMenu = QAction('Histogram', self)
        #histogramMenu.triggered.connect(lambda: self.updateImgAfter(

        grayscaleMenu = QAction('Grayscale', self)
        grayscaleMenu.triggered.connect(lambda: self.updateImgAfter(util.getgrayscale(self.imgNpBefore)))

        binarizeMenu = QAction('Binarize', self)
        binarizeMenu.triggered.connect(lambda: self.updateImgAfter(util.binarize(self.imgNpBefore)))

        gaussianMenu = QAction('Smooth', self)
        gaussianMenu.triggered.connect(lambda: self.updateImgAfter(util.convolve(util.gaussian_filt(), util.getgrayscale(self.imgNpBefore))))        

        resizeMenu = QAction('Resize', self)
        resizeMenu.triggered.connect(lambda: self.updateImgAfter(util.downsample(self.imgNpBefore)))

        segmentMenu = QAction('Segment', self)
        segmentMenu.triggered.connect(lambda: self.updateTxtAfter(str(len(util.preprocess(util.segment(util.zhangsuen(util.binarize(self.imgNpBefore))))))))

        # FEATURE MENU
        chaincodeMenu = QAction('Chain code', self)
        chaincodeMenu.triggered.connect(lambda: self.updateTxtAfter(str([util.getdirection(chain[n][0], chain[n][1]) for chain in util.segment(util.thin(self.imgNpBefore), cc=True) for n in xrange(len(chain))])))

        turncodeMenu = QAction('Turn code', self)
        turncodeMenu.triggered.connect(lambda: self.updateTxtAfter(str([util.getturncode(cc) for cc in util.segment(util.thin(self.imgNpBefore, bg='light'), cc=False)])))

        skeletonMenu = QAction('Zhang-Suen thinning', self)
        skeletonMenu.triggered.connect(lambda:self.updateImgAfter(util.zhangsuen(util.binarize(self.imgNpBefore, bg='light'))))

        skinMenu = QAction('Boundary detection', self)
        skinMenu.triggered.connect(lambda:self.updateImgAfter(util.thin(self.imgNpBefore, bg='light')))

        freemanMenu = QAction('Contour profile', self)


        # RECOGNITION MENU
        freemantrainfontMenu = QAction('Train Contour Font', self)
        freemantrainfontMenu.triggered.connect(lambda: util.train(self.imgNpBefore, feats='zs', setname='font1')) 

        freemantrainplatMenu = QAction('Train ZS Plate (GNB)', self)
        freemantrainplatMenu.triggered.connect(lambda: util.gnb_train())

        cctctrainfontMenu = QAction('Train CC + TC Font', self)

        cctctrainplatMenu = QAction('Train CC + TC Plate', self)

        freemantestfontMenu = QAction('Predict Contour Font', self)
        freemantestfontMenu.triggered.connect(lambda: self.updateTxtAfter(util.test(self.imgNpBefore, feats='zs', setname='font1')))
        
        freemantestplatMenu = QAction('Predict Contour Plate', self)
        cctctestfontMenu = QAction('Predict CC + TC Font', self)

        cctctestplatMenu = QAction('Predict CC + TC Plate', self)

        # MENU BAR
        menubar = self.menuBar()

        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)
        fileMenu.addAction(exitAction)
        fileMenu.addAction(moveLeft)

        processMenu = menubar.addMenu('&Preprocess')
        #processMenu.addAction(histogramMenu)
        processMenu.addAction(equalizeMenu)
        processMenu.addAction(grayscaleMenu)
        processMenu.addAction(binarizeMenu)
        processMenu.addAction(gaussianMenu)
        processMenu.addAction(resizeMenu)
        processMenu.addAction(segmentMenu)

        featureMenu = menubar.addMenu('&Features')
        featureMenu.addAction(chaincodeMenu)
        featureMenu.addAction(turncodeMenu)
        featureMenu.addAction(skeletonMenu)
        featureMenu.addAction(skinMenu)
        featureMenu.addAction(freemanMenu)

        recogMenu = menubar.addMenu('&Recognition')
        recogMenu.addAction(freemantrainfontMenu)
        recogMenu.addAction(freemantrainplatMenu)
        recogMenu.addAction(cctctrainfontMenu)
        recogMenu.addAction(cctctrainplatMenu)
        recogMenu.addAction(freemantestfontMenu)
        recogMenu.addAction(freemantestplatMenu)
        recogMenu.addAction(cctctestfontMenu)
        recogMenu.addAction(cctctestplatMenu)
        #recogMenu.addAction(

        """
        Toolbar, Status Bar, Tooltip
        """
        self.statusBar().showMessage('Ready')

        QToolTip.setFont(QFont('SansSerif', 10))
        #self.setToolTip('This is a <b>QWidget</b> widget')

        """
        Displaying
        """

        self.setGeometry(12, 30, self.mainWidth, self.mainHeight+80)
        self.setWindowTitle('Pyxel')
        self.setWindowIcon(QIcon('res/web.png'))

        self.setCentralWidget(self.main)

        self.main.setGeometry(QRect(0, 80, self.mainWidth, self.mainHeight))
        #self.mainAfter.setGeometry(QRect(self.mainWidth/2, 80, self.mainWidth/2, self.mainHeight))

        self.show()

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home/cilsat/Dropbox/kuliah/sem1/pp')

        imgOriginal = np.require(mpimg.imread(fname[0]), np.uint8, 'C')
        self.updateBefore(imgOriginal)

    def updateBefore(self, img):

        print img.shape

        # the image format depends on the size/shape of the input image array
        if len(img.shape) == 3:
            height, width, bytesPerPixel = img.shape
            if bytesPerPixel == 3:
                imgBefore = QImage(img, width, height, bytesPerPixel*width, QImage.Format_RGB888)
            elif bytesPerPixel == 4:
                imgBefore = QImage(img, width, height, bytesPerPixel*width, QImage.Format_RGBA8888_Premultiplied)

        elif len(img.shape) == 2:
            height, width = img.shape
            if img.dtype == np.bool:
                img = img.astype(np.uint8)*128
            if img.dtype == np.uint8:
                imgBefore = QImage(img, width, height, width, QImage.Format_Indexed8)

        myPixmap = QPixmap.fromImage(imgBefore)
        myPixmap = myPixmap.scaled(self.mainBefore.size(), Qt.KeepAspectRatio)
        self.mainBefore.setPixmap(myPixmap)
        self.imgNpBefore = img

    def updateImgAfter(self, img):

        # the image format depends on the size/shape of the input image array
        if len(img.shape) == 3:
            height, width, bytesPerPixel = img.shape
            if bytesPerPixel == 3:
                imgAfter = QImage(img, width, height, bytesPerPixel*width, QImage.Format_RGB888)
            elif bytesPerPixel == 4:
                imgAfter = QImage(img, width, height, bytesPerPixel*width, QImage.Format_RGBA8888_Premultiplied)

        elif len(img.shape) == 2:
            height, width = img.shape
            if img.dtype == np.bool:
                img = img.astype(np.uint8)*128
            if img.dtype == np.uint8:
                imgAfter = QImage(img, width, height, width, QImage.Format_Indexed8)
        myPixmap = QPixmap.fromImage(imgAfter)
        myPixmap = myPixmap.scaled(self.mainAfter.size(), Qt.KeepAspectRatio)
        self.mainAfter.setPixmap(myPixmap)
        self.imgNpAfter = img


    def updateTxtAfter(self, txt):
        self.mainAfter.setText(txt)

    def equalize(self):
        
        eq = util.equalize(self.imgNpBefore)
        self.updateImgAfter(eq)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()

    sys.exit(app.exec_())
