try:
    import cv2
    import numpy as np
    import os
except:
    print("Library tidak ditemukan !")
    print("Pastikan library cv2, os, numpy sudah terinstall")


class ImageDataGenerator:
    def __init__(self, lokasi):
        self.lokasi = lokasi

    def ambil_gambar(lokasi, jenis='', rescale=None):
        labels = os.listdir(os.path.join(lokasi, jenis))
        X = []
        y = []

        for label in labels:
            for file in os.listdir(os.path.join(lokasi, jenis, label)):
                gambar = cv2.imread(os.path.join(lokasi, jenis, label, file), 0)

                if rescale == None:
                    X.append(gambar)
                else:
                    X.append(gambar / rescale)
                y.append(label)

        return np.array(X), np.array(y)


    def load_dataset(self, rescale=1):
        x_train, y_train = ImageDataGenerator.ambil_gambar(lokasi=self.lokasi, jenis="train", rescale=rescale)
        x_test, y_test = ImageDataGenerator.ambil_gambar(lokasi=self.lokasi, jenis="test", rescale=rescale)

        self.y_train = y_train
        self.y_test = y_test

        return x_train, y_train, x_test, y_test

    def load_class(self):
        return self.class_indicies

    def generate_oneHot(labels, kelas):
        arr_oneHot = []
        for k in kelas:
            oneHot = []
            for label in labels:
                if label == k:
                    oneHot.append(1)
                else:
                    oneHot.append(0)
            arr_oneHot.append(oneHot)
        return arr_oneHot
    def load_class_oneHot(self):
        labels = os.listdir(os.path.join(self.lokasi, "train"))
        y_train_oneHot = ImageDataGenerator.generate_oneHot(labels, self.y_train)
        y_test_onHot = ImageDataGenerator.generate_oneHot(labels, self.y_test)

        return y_train_oneHot, y_test_onHot