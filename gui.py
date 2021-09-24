from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtWidgets import QMessageBox
import sys
import keyboard
import cv2
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
import time
import numpy as np
import shutil
import os
import copy
from khisoft.layers import Model, Conv2d, Maxpooling2D, Flatten, Dense, Relu
from khisoft.preprocessing import ImageDataGenerator
from khisoft.callback import ProgressBar


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        self.threshold = 60
        self.model = Model([
            Conv2d(padding=1, stride=1),
            Relu(),
            Maxpooling2D(ukuran_filter=2, stride=1),
            Flatten(),
            Dense([10]),
        ])
        self.epochs = 30000
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.createDataset()

    def pushScreen(self, screenName):
        self.jalan = False
        try:
            self.cam.release()
            cv2.destroyAllWindows()
            keyboard.press_and_release('esc')
        except Exception as e:
            print(e)
            pass
        uic.loadUi('UI/'+screenName+'.ui', self)
        self.setWindowIcon(QtGui.QIcon('images/khisoft.png'))
        self.setWindowTitle('Khisoft Sign Language Translator')
        self.btnScreenCreateDataset.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnScreenTraining.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnScreenTesting.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnKeluar.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.btnScreenCreateDataset.clicked.connect(self.createDataset)
        self.btnScreenTraining.clicked.connect(self.training)
        self.btnScreenTesting.clicked.connect(self.testing)

        self.btnKeluar.clicked.connect(self.keluar)

    def saveFrame(self):
        if self.leClass.text() == "":
            QMessageBox.warning(self, "Perhatian", "Inputan kolom class harus di isi")
        else:
            directory = "raw_datasets"
            if not os.path.exists(directory):
                os.makedirs(directory)

            directory = "real"
            if not os.path.exists(directory):
                os.makedirs(directory)

            gambar = self.gambar_crop
            kelas = str(self.leClass.text()).lower()
            dir = 'raw_datasets/' + kelas + '/'
            if not os.path.exists(dir):
                os.mkdir(dir)

            dir2 = 'real/' + kelas + '/'
            if not os.path.exists(dir2):
                os.mkdir(dir2)

            list = os.listdir(dir)
            if len(list) == 0:
                file_name = '1'
            else:
                angka = sorted([int(i.replace('.png', '')) for i in list])[-1]
                file_name = str(angka + 1)

            if int(file_name) <= 100:
                cv2.imwrite(dir + file_name + '.png', gambar)
                self.lInfoClass.setText("Class " + kelas + " mempunyai " + str(len(list)+1) + " data")

                cv2.imwrite(dir2 + file_name + '_crop_biner.png', self.gambar_crop)
                cv2.imwrite(dir2 + file_name + '_crop_asli.png', self.gambar_crop_asli)
                cv2.imwrite(dir2 + file_name + '_crop_grayscale.png', self.gambar_crop_grayscale)
                cv2.imwrite(dir2 + file_name + '_full.png', self.gambar_full)
            else:
                self.lInfoClass.setText("Sudah mencapai 100")

    def saveFrameTesting(self):
        if self.leClass.text() == "":
            QMessageBox.warning(self, "Perhatian", "Inputan kolom class harus di isi")
        else:
            directory = "testing"
            if not os.path.exists(directory):
                os.makedirs(directory)

            judul = self.leClass.text().split(',')

            tipe = str(judul[1]).lower()
            dir = directory + '/' + tipe + '/'
            if not os.path.exists(dir):
                os.mkdir(dir)

            kelas = str(judul[0]).lower()
            dir = directory + '/' + tipe + '/' + kelas + '/'
            if not os.path.exists(dir):
                os.mkdir(dir)

            cv2.imwrite(dir + 'gambar_crop_biner.png', self.gambar_crop)
            cv2.imwrite(dir + 'gambar_crop_asli.png', self.gambar_crop_asli)
            cv2.imwrite(dir + 'gambar_crop_grayscale.png', self.gambar_crop_grayscale)
            cv2.imwrite(dir + 'gambar_full.png', self.gambar_full)
            self.lInfoClass.setText("Sudah di save")

    def keluar(self):
        ask = QMessageBox.question(self, 'Keluar dari Aplikasi', "Apakah Anda yakin ingin Keluar dari aplikasi?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ask == QMessageBox.Yes:
            self.cam.release()
            cv2.destroyAllWindows()
            keyboard.press_and_release('alt+F4')

    def training(self):
        self.pushScreen('training')
        self.show()
        self.btnTraining.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnTraining.clicked.connect(self.prosesTraining)


        dir = 'raw_datasets/'
        list = os.listdir(dir)
        teks = ''
        for i in list:
            jumlah = str(len(os.listdir(dir+i)))
            teks = teks + "Class {} Memiliki {} Data".format(i, jumlah) + '\n'

        self.teClassSummary.append(teks)
    def prosesTraining(self):
        self.lInfoProgress.setText("Memulai Proses Training. Mohon tunggu sebentar..")

        if os.path.exists('data'):
            shutil.rmtree("data/", ignore_errors=False, onerror=None)

        if os.path.exists('model'):
            shutil.rmtree("model/", ignore_errors=False, onerror=None)

        os.mkdir('data/')
        os.mkdir('model/')
        os.mkdir('data/train/')
        os.mkdir('data/test/')

        self.lInfoProgress.setText("Membagi dataset kedalam 80% training dan 20% testing...")
        for i in os.listdir("raw_datasets"):
            jumlah_file = len(
                [name for name in os.listdir("raw_datasets/" + i) if os.path.isfile(os.path.join("raw_datasets/" + i, name))])
            persen = round(jumlah_file * 0.2)
            print("Kelas : " + str(i) + ", jumlah : " + str(jumlah_file) + ", (train, test) : ({},{})".format(
                jumlah_file - persen, persen))

            if not os.path.exists("data/train/" + str(i)):
                os.mkdir("data/train/" + str(i))
            if not os.path.exists("data/test/" + str(i)):
                os.mkdir("data/test/" + str(i))

            for index, berkas in enumerate(os.listdir("raw_datasets/" + i)):
                if index > persen:
                    shutil.copyfile("raw_datasets/{}/".format(i) + str(berkas),
                                    "data/train/" + i + "/" + str(berkas))
                else:
                    shutil.copyfile("raw_datasets/{}/".format(i) + str(berkas),
                                    "data/test/" + i + "/" + str(berkas))
        self.lInfoProgress.setText("Proses pembagian dataset selesai...")

        self.lInfoProgress.setText("Memulai Proses Konvolusi...")
        dataset = ImageDataGenerator("data")
        X_train, y_train, X_test, y_test = dataset.load_dataset(rescale=255)
        y_train_oneHot, y_test_oneHot = dataset.load_class_oneHot()

        self.lInfoProgress.setText("Memulai Proses Training. Mohon tunggu sebentar..")
        model = self.model
        teks = model.summary(input_shape=(28, 28))
        self.pbLoading.setMinimum(1)
        self.pbLoading.setMaximum(100)
        self.pbLoading.setValue(50)

        # pbCallback = ProgressBar.set(progressBar='self.pbLoading')

        model.fit(epochs=self.epochs, X_input=X_train, y_input=y_train_oneHot, X_validation=X_test, y_validation=y_test_oneHot,
                  callback=[''])
        # predict = model.pred(X_input=np.array([X_train[0]]))
        # predict_arg_max = predict.argmax(axis=0)
        model.plot()
        self.teClassSummary.append(teks)
        np.save("model/class", dataset.load_class())
        # print(dataset.load_class())

        self.lInfoProgress.setText("Proses Training Selesai.")

        # plt.imshow(X_train[0], cmap='gray')
        # plt.show()

    def resetDataset(self):
        if os.path.exists('data'):
            shutil.rmtree("data/", ignore_errors=True, onerror=None)

        if os.path.exists('model'):
            shutil.rmtree("model/", ignore_errors=True, onerror=None)

        if os.path.exists('raw_datasets'):
            shutil.rmtree("raw_datasets", ignore_errors=True, onerror=None)

        if os.path.exists('real'):
            shutil.rmtree("real", ignore_errors=True, onerror=None)

        msg = QtWidgets.QMessageBox()
        msg.setText("Dataset berhasil direset!")
        msg.setWindowTitle("Informasi")
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

        self.lInfoClass.setText("")

    def hs_change(self):
        self.l_h = self.hs_lh.value()
        self.l_s = self.hs_ls.value()
        self.l_v = self.hs_lv.value()
        self.u_h = self.hs_hh.value()
        self.u_s = self.hs_hs.value()
        self.u_v = self.hs_hv.value()

        arr = [self.l_h, self.l_s, self.l_v, self.u_h, self.u_s, self.u_v]

        np.save("filter_setting", arr)

        self.lInfoClass.setText("Low Hue {}, Low Salturation {}, Low Value {}, High Hue {}, High Salturation {}, High Value {}".format(self.l_h, self.l_s, self.l_v, self.u_h, self.u_s, self.u_v))

    def createDataset(self):
        self.pushScreen('create_dataset')
        self.btnSimpan.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnSimpan.clicked.connect(self.saveFrame)
        self.btnResetDataset.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnResetDataset.clicked.connect(self.resetDataset)

        if os.path.exists('filter_setting.npy'):
            setting = np.load('filter_setting.npy', allow_pickle=True)
            self.hs_lh.setValue(setting[0])
            self.hs_ls.setValue(setting[1])
            self.hs_lv.setValue(setting[2])
            self.hs_hh.setValue(setting[3])
            self.hs_hs.setValue(setting[4])
            self.hs_hv.setValue(setting[5])
        else:
            self.hs_lh.setValue(0)
            self.hs_ls.setValue(51)
            self.hs_lv.setValue(0)
            self.hs_hh.setValue(255)
            self.hs_hs.setValue(255)
            self.hs_hv.setValue(255)


        self.hs_lh.valueChanged.connect(self.hs_change)
        self.hs_ls.valueChanged.connect(self.hs_change)
        self.hs_lv.valueChanged.connect(self.hs_change)
        self.hs_hh.valueChanged.connect(self.hs_change)
        self.hs_hs.valueChanged.connect(self.hs_change)
        self.hs_hv.valueChanged.connect(self.hs_change)
        self.show()

        # classes = np.load('model/class.npy', allow_pickle=True)


        self.cam = cv2.VideoCapture(0)
        x, y, w, h = 350, 150, 200, 200
        self.jalan = True
        while self.jalan:
            ret, frame = self.cam.read()
            frame = cv2.flip(frame, 1)
            try:

                gc = frame[y:y + h, x:x + w]
                self.gambar_full = copy.deepcopy(frame)
                self.gambar_crop_asli = copy.deepcopy(gc)
                self.gambar_crop_grayscale = copy.deepcopy(gc)
                self.gambar_crop_grayscale = cv2.cvtColor(self.gambar_crop_grayscale, cv2.COLOR_BGR2GRAY)

                img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img2 = cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 0), 2)

                l_h = self.hs_lh.value()
                l_s = self.hs_ls.value()
                l_v = self.hs_lv.value()
                u_h = self.hs_hh.value()
                u_s = self.hs_hs.value()
                u_v = self.hs_hv.value()

                gc = cv2.cvtColor(gc, cv2.COLOR_BGR2HSV)
                lower_blue = np.array([l_h, l_s, l_v])
                upper_blue = np.array([u_h, u_s, u_v])
                mask = cv2.inRange(gc, lower_blue, upper_blue)
                gc = cv2.bitwise_and(gc, gc, mask=mask)
                gc = cv2.cvtColor(gc, cv2.COLOR_HSV2RGB)
                gc = cv2.cvtColor(gc, cv2.COLOR_RGB2GRAY)
                ret, gc = cv2.threshold(gc, self.threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.gambar_crop = copy.deepcopy(gc)

                # model = copy.deepcopy(self.model)
                # predict = model.pred(X_input=np.array([self.gambar_crop]))
                # predict_arg_max = predict.argmax(axis=0)
                # predict_class = classes[predict_arg_max[0]]
                # self.lInfoClass.setText("Terprediksi {} dengan nilai {}%".format(predict_class, round(predict[predict_arg_max[0]][0] * 100)))
                #
                # judul = self.leClass.text().split(',')
                # if(predict_class == judul[0]):
                #     if os.path.exists('testing/{}/{}'.format(judul[1], judul[0])):
                #         pass
                #     else:
                #         self.saveFrameTesting()
                #         time.sleep(0.1)

                height2, width2, channel = img2.shape
                step2 = channel * width2
                qImg1 = QImage(img2.data, width2, height2, step2, QImage.Format_RGB888)
                self.lImage.setPixmap(QPixmap.fromImage(qImg1))

                height, width = self.gambar_crop.shape
                bytesPerLine = 1 * width
                gc = QImage(self.gambar_crop.data, 200, 200, bytesPerLine, QImage.Format_Grayscale8)
                self.lImage2.setPixmap(QPixmap.fromImage(gc))

            except Exception as e:
                print(e)
                keyboard.press_and_release('esc')
                self.jalan = False
                break

            if keyboard.is_pressed('ctrl+s'):
                self.saveFrame()
                time.sleep(0.1)

            if keyboard.is_pressed('ctrl+a'):
                self.saveFrameTesting()
                time.sleep(0.1)


            if cv2.waitKey(1) == 27 or self.jalan == False:
                break

        self.cam.release()
        cv2.destroyAllWindows()

    def testing(self):
        self.pushScreen('testing')

        if os.path.exists('filter_setting.npy'):
            setting = np.load('filter_setting.npy', allow_pickle=True)
            self.hs_lh.setValue(setting[0])
            self.hs_ls.setValue(setting[1])
            self.hs_lv.setValue(setting[2])
            self.hs_hh.setValue(setting[3])
            self.hs_hs.setValue(setting[4])
            self.hs_hv.setValue(setting[5])
        else:
            self.hs_lh.setValue(0)
            self.hs_ls.setValue(51)
            self.hs_lv.setValue(0)
            self.hs_hh.setValue(255)
            self.hs_hs.setValue(255)
            self.hs_hv.setValue(255)

        self.hs_lh.valueChanged.connect(self.hs_change)
        self.hs_ls.valueChanged.connect(self.hs_change)
        self.hs_lv.valueChanged.connect(self.hs_change)
        self.hs_hh.valueChanged.connect(self.hs_change)
        self.hs_hs.valueChanged.connect(self.hs_change)
        self.hs_hv.valueChanged.connect(self.hs_change)
        self.show()


        self.cam = cv2.VideoCapture(0)
        x, y, w, h = 350, 150, 200, 200
        self.jalan = True
        classes = np.load('model/class.npy', allow_pickle=True)
        teks = ""
        self.leResult.setText("")
        arr = []
        while self.jalan:
            ret, frame = self.cam.read()
            frame = cv2.flip(frame, 1)
            try:
                gc = frame[y:y + h, x:x + w]
                img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img2 = cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 0), 2)

                l_h = self.hs_lh.value()
                l_s = self.hs_ls.value()
                l_v = self.hs_lv.value()
                u_h = self.hs_hh.value()
                u_s = self.hs_hs.value()
                u_v = self.hs_hv.value()

                gc = cv2.cvtColor(gc, cv2.COLOR_BGR2HSV)
                lower_blue = np.array([l_h, l_s, l_v])
                upper_blue = np.array([u_h, u_s, u_v])
                mask = cv2.inRange(gc, lower_blue, upper_blue)
                gc = cv2.bitwise_and(gc, gc, mask=mask)
                gc = cv2.cvtColor(gc, cv2.COLOR_HSV2RGB)
                gc = cv2.cvtColor(gc, cv2.COLOR_RGB2GRAY)
                ret, gc = cv2.threshold(gc, self.threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                gambarnya = gc

                height2, width2, channel = img2.shape
                step2 = channel * width2
                qImg1 = QImage(img2.data, width2, height2, step2, QImage.Format_RGB888)
                self.lImageTesting.setPixmap(QPixmap.fromImage(qImg1))

                height, width = gambarnya.shape
                bytesPerLine = 1 * width
                gc = QImage(gambarnya.data, 200, 200, bytesPerLine, QImage.Format_Grayscale8)
                self.lImageTesting2.setPixmap(QPixmap.fromImage(gc))
                model = copy.deepcopy(self.model)
                predict = model.pred(X_input=np.array([gambarnya]))
                # print(predict)
                predict_arg_max = predict.argmax(axis=0)
                predict_class = classes[predict_arg_max[0]]
                # print(predict[predict_arg_max[0]], predict_class)
                self.lInfoClass.setText("Terprediksi {} dengan nilai {}%".format(predict_class, round(predict[predict_arg_max[0]][0] * 100)))
                length_arr = 10
                if (len(set(arr)) == 1):
                    yakin = (len(arr)/length_arr) * 100
                else:
                    yakin = 0
                self.lInfoClassBottom.setText("Yakin {}%".format(round(yakin)))

                if (predict[predict_arg_max[0]] > 0.5):
                    arr.append(predict_class)
                    if (len(arr) >= length_arr):
                        if (len(set(arr)) == 1):
                            panjang_teks = len(predict_class)
                            if panjang_teks > 1:
                                predict_class = predict_class + " "
                            teks = teks + predict_class
                            self.leResult.setText(teks)
                            # print(teks)
                        arr.clear()



            except Exception as e:
                print("error")
                print(e)
                keyboard.press_and_release('esc')
                self.jalan = False
                break

            if cv2.waitKey(1) == 27 or self.jalan == False:
                break

        self.cam.release()
        cv2.destroyAllWindows()


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
