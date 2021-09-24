# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'create_dataset.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(884, 792)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lJudul = QtWidgets.QLabel(self.centralwidget)
        self.lJudul.setGeometry(QtCore.QRect(10, 20, 651, 81))
        self.lJudul.setAlignment(QtCore.Qt.AlignCenter)
        self.lJudul.setWordWrap(True)
        self.lJudul.setObjectName("lJudul")
        self.leClass = QtWidgets.QLineEdit(self.centralwidget)
        self.leClass.setGeometry(QtCore.QRect(20, 670, 341, 41))
        self.leClass.setStyleSheet("border: 5px solid #FF823A;\n"
"font-size: 120%;\n"
"border-radius: 15px;\n"
"padding:5px 5px;\n"
"background: transparent;")
        self.leClass.setObjectName("leClass")
        self.lImage = QtWidgets.QLabel(self.centralwidget)
        self.lImage.setGeometry(QtCore.QRect(20, 170, 641, 480))
        self.lImage.setStyleSheet("border: 5px solid #FF823A;\n"
"font-size: 40px;\n"
"border-radius: 15px;")
        self.lImage.setAlignment(QtCore.Qt.AlignCenter)
        self.lImage.setObjectName("lImage")
        self.btnSimpan = QtWidgets.QPushButton(self.centralwidget)
        self.btnSimpan.setGeometry(QtCore.QRect(370, 670, 141, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.btnSimpan.setFont(font)
        self.btnSimpan.setStyleSheet("background-color:#38FF3D;\n"
"text-weight: bold;\n"
"color: white;\n"
"border-radius:15px;\n"
"font-size: 150%;")
        self.btnSimpan.setObjectName("btnSimpan")
        self.btnScreenCreateDataset = QtWidgets.QPushButton(self.centralwidget)
        self.btnScreenCreateDataset.setGeometry(QtCore.QRect(30, 110, 201, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.btnScreenCreateDataset.setFont(font)
        self.btnScreenCreateDataset.setStyleSheet("background-color:#38FF3D;\n"
"text-weight: bold;\n"
"border-radius:15px;\n"
"color:#282D39;\n"
"font-size: 150%;")
        self.btnScreenCreateDataset.setObjectName("btnScreenCreateDataset")
        self.btnScreenTraining = QtWidgets.QPushButton(self.centralwidget)
        self.btnScreenTraining.setGeometry(QtCore.QRect(240, 110, 201, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.btnScreenTraining.setFont(font)
        self.btnScreenTraining.setStyleSheet("background-color:#38FF3D;\n"
"text-weight: bold;\n"
"border-radius:15px;\n"
"color:#282D39;\n"
"font-size: 150%;")
        self.btnScreenTraining.setObjectName("btnScreenTraining")
        self.btnScreenTesting = QtWidgets.QPushButton(self.centralwidget)
        self.btnScreenTesting.setGeometry(QtCore.QRect(450, 110, 201, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.btnScreenTesting.setFont(font)
        self.btnScreenTesting.setStyleSheet("background-color:#38FF3D;\n"
"text-weight: bold;\n"
"border-radius:15px;\n"
"color:#282D39;\n"
"font-size: 150%;")
        self.btnScreenTesting.setObjectName("btnScreenTesting")
        self.btnKeluar = QtWidgets.QPushButton(self.centralwidget)
        self.btnKeluar.setGeometry(QtCore.QRect(820, 0, 61, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.btnKeluar.setFont(font)
        self.btnKeluar.setStyleSheet("background-color:#FF3A3A;\n"
"text-weight: bold;\n"
"color: white;")
        self.btnKeluar.setObjectName("btnKeluar")
        self.lInfoClass = QtWidgets.QLabel(self.centralwidget)
        self.lInfoClass.setGeometry(QtCore.QRect(20, 730, 851, 61))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.lInfoClass.setFont(font)
        self.lInfoClass.setText("")
        self.lInfoClass.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.lInfoClass.setObjectName("lInfoClass")
        self.btnResetDataset = QtWidgets.QPushButton(self.centralwidget)
        self.btnResetDataset.setGeometry(QtCore.QRect(520, 670, 141, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.btnResetDataset.setFont(font)
        self.btnResetDataset.setStyleSheet("background-color:#FF3A3A;\n"
"text-weight: bold;\n"
"color: white;\n"
"border-radius:15px;\n"
"font-size: 150%;")
        self.btnResetDataset.setObjectName("btnResetDataset")
        self.lImage2 = QtWidgets.QLabel(self.centralwidget)
        self.lImage2.setGeometry(QtCore.QRect(670, 170, 200, 200))
        font = QtGui.QFont()
        font.setPointSize(-1)
        self.lImage2.setFont(font)
        self.lImage2.setStyleSheet("border: 5px solid #FF823A;\n"
"font-size: 12px;\n"
"border-radius: 15px;")
        self.lImage2.setAlignment(QtCore.Qt.AlignCenter)
        self.lImage2.setObjectName("lImage2")
        self.hs_lh = QtWidgets.QSlider(self.centralwidget)
        self.hs_lh.setGeometry(QtCore.QRect(680, 400, 160, 22))
        self.hs_lh.setMaximum(255)
        self.hs_lh.setOrientation(QtCore.Qt.Horizontal)
        self.hs_lh.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.hs_lh.setObjectName("hs_lh")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(680, 380, 47, 13))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(680, 430, 47, 13))
        self.label_2.setObjectName("label_2")
        self.hs_ls = QtWidgets.QSlider(self.centralwidget)
        self.hs_ls.setGeometry(QtCore.QRect(680, 450, 160, 22))
        self.hs_ls.setMaximum(255)
        self.hs_ls.setOrientation(QtCore.Qt.Horizontal)
        self.hs_ls.setObjectName("hs_ls")
        self.hs_lv = QtWidgets.QSlider(self.centralwidget)
        self.hs_lv.setGeometry(QtCore.QRect(680, 500, 160, 22))
        self.hs_lv.setMaximum(255)
        self.hs_lv.setOrientation(QtCore.Qt.Horizontal)
        self.hs_lv.setObjectName("hs_lv")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(680, 480, 47, 13))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(680, 630, 47, 13))
        self.label_4.setObjectName("label_4")
        self.hs_hv = QtWidgets.QSlider(self.centralwidget)
        self.hs_hv.setGeometry(QtCore.QRect(680, 650, 160, 22))
        self.hs_hv.setMaximum(255)
        self.hs_hv.setOrientation(QtCore.Qt.Horizontal)
        self.hs_hv.setObjectName("hs_hv")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(680, 530, 47, 13))
        self.label_5.setObjectName("label_5")
        self.hs_hs = QtWidgets.QSlider(self.centralwidget)
        self.hs_hs.setGeometry(QtCore.QRect(680, 600, 160, 22))
        self.hs_hs.setMaximum(255)
        self.hs_hs.setOrientation(QtCore.Qt.Horizontal)
        self.hs_hs.setObjectName("hs_hs")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(680, 580, 47, 13))
        self.label_6.setObjectName("label_6")
        self.hs_hh = QtWidgets.QSlider(self.centralwidget)
        self.hs_hh.setGeometry(QtCore.QRect(680, 550, 160, 22))
        self.hs_hh.setMaximum(255)
        self.hs_hh.setOrientation(QtCore.Qt.Horizontal)
        self.hs_hh.setObjectName("hs_hh")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.lJudul.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:600;\">KONVERSI BAHASA ISYARAT (SIBI) KE TEKS BAHASA INDONESIA DENGAN MENGGUNAKAN CONVOLUTIONAL NEURAL NETWORK (CNN)</span></p></body></html>"))
        self.leClass.setPlaceholderText(_translate("MainWindow", "Masukkan Nama Class"))
        self.lImage.setText(_translate("MainWindow", "Video Stream"))
        self.btnSimpan.setText(_translate("MainWindow", "Simpan Langsung"))
        self.btnScreenCreateDataset.setText(_translate("MainWindow", "Buat Dataset"))
        self.btnScreenTraining.setText(_translate("MainWindow", "Training"))
        self.btnScreenTesting.setText(_translate("MainWindow", "Testing"))
        self.btnKeluar.setText(_translate("MainWindow", "Keluar"))
        self.btnResetDataset.setText(_translate("MainWindow", "Reset Dataset"))
        self.lImage2.setText(_translate("MainWindow", "Video Stream"))
        self.label.setText(_translate("MainWindow", "Low H"))
        self.label_2.setText(_translate("MainWindow", "Low S"))
        self.label_3.setText(_translate("MainWindow", "Low V"))
        self.label_4.setText(_translate("MainWindow", "High V"))
        self.label_5.setText(_translate("MainWindow", "High H"))
        self.label_6.setText(_translate("MainWindow", "High S"))
