try:
    import cv2
    import numpy as np
    import os
    import random
    import sys
except:
    print("Library tidak ditemukan !")
    print("Pastikan library cv2, os, numpy, random, sys sudah terinstall")

class ProgressBar:
    def printProgressBar(awal=100, akhir=10, prefix = ''):
        suffix = '(' + str(awal) + '/' + str(akhir) + ')'
        decimals = 1
        length = 100
        fill = '█'

        percent = ("{0:." + str(decimals) + "f}").format(100 * (awal / float(akhir)))
        filled_length = int(length * awal // akhir)
        bar = fill * filled_length + '-' * (length - filled_length)

        sys.stdout.write("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix))
        sys.stdout.flush()

class Model:
    def __init__(self):
        self.layers = []
        self.input_output = []
        self.validation_output = []

    def __init__(self, layers=[]):
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def summary(self, input_shape=(200,200)):
        X, Y = input_shape
        X_input = np.array([[[random.random() for i in range(X)] for j in range(X)]])
        y_input = np.array([[[random.random() for i in range(X)] for j in range(X)]])
        X_validation = np.array([[[random.random() for i in range(X)] for j in range(X)]])
        y_validation = np.array([[[random.random() for i in range(X)] for j in range(X)]])

        print("Berikut adalah summary dari model dengan 1 gambar ukuran " + str(input_shape))
        for layer in self.layers:
            try:
                input_X = input_output
                validation_X = validation_output
            except:
                input_X = X_input
                validation_X = X_validation

            if type(layer).__name__ == "Dense":
                print("Dense with hidden layer = " + str(layer.hidden_layer))
            else:
                input_output, validation_output = layer.train(X_input=input_X, y_input=y_input,
                                                                X_validation=validation_X, y_validation=y_validation, train=False)
                input_output = input_output
                validation_output = validation_output

                print(type(layer).__name__, "output shape = " + str(input_output.shape))

    def compile(self):
        pass

    def fit(self, X_input=[], y_input=[], X_validation=[], y_validation=[], epochs=10, callback=[]):
        self.X_input = X_input
        self.y_input = y_input
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.epochs = epochs


        print()
        print("Memulai Training : ")
        for layer in self.layers:
            try:
                input_X = self.input_output
                validation_X = self.validation_output
            except:
                input_X = self.X_input
                validation_X = self.X_validation

            if type(layer).__name__ == "Dense":
                layer.set_params(epochs=epochs,callback=callback)
                input_output, validation_output = layer.train(X_input=input_X, y_input=self.y_input,
                                                              X_validation=validation_X, y_validation=self.y_validation,
                                                              train=True)
                self.input_output = input_output
                self.validation_output = validation_output

            else:
                input_output, validation_output = layer.train(X_input=input_X, y_input=self.y_input,
                                                              X_validation=validation_X, y_validation=self.y_validation,
                                                              train=True)
                self.input_output = input_output
                self.validation_output = validation_output


        print()
        print("Proses Training Selesai...")

class Conv2d:
    def __init__(self, padding=0, stride=1):
        self.filter1 = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])
        self.filter2 = np.array([[1,   1,  1],
                                   [0,   0,  0],
                                   [-1, -1, -1]])
        self.padding = padding
        self.stride = stride

    def convolution(self, gambar_input, train=True):
        _, X, Y = gambar_input.shape
        X_FILTER, Y_FILTER = self.filter1.shape

        arr_hasil_perkalian_konvolusi = []
        awal = 1
        for gambar in gambar_input:
            # menambah padding pada setiap sisi
            gambar_output = np.zeros((X + (self.padding * 2) * 1, Y + (self.padding * 2) * 1), dtype=gambar.dtype)
            if self.padding == 0:
                gambar_output = gambar
            else:
                gambar_output[self.padding:-self.padding, self.padding:-self.padding] = gambar

            X_baru, Y_baru = gambar_output.shape
            # menghitung ukuran matriks hasil konvolusi
            ukuran_hasil_perkalian_X = int((X_baru - X_FILTER) / self.stride) + 1
            ukuran_hasil_perkalian_Y = int((Y_baru - Y_FILTER) / self.stride) + 1
            hasil_perkalian_konvolusi = np.zeros((ukuran_hasil_perkalian_X, ukuran_hasil_perkalian_Y))

            # Proses perkalian konvolusi
            posisi_Y_hasil = 0
            posisi_Y_sekarang = 0

            while posisi_Y_sekarang + Y_FILTER <= Y_baru:
                posisi_X_hasil = 0
                posisi_X_sekarang = 0

                while posisi_X_sekarang + X_FILTER <= X_baru:
                    hasil_perkalian_filter_1 = self.filter1 * gambar_output[
                                                              posisi_Y_sekarang:posisi_Y_sekarang + Y_FILTER,
                                                              posisi_X_sekarang:posisi_X_sekarang + X_FILTER]
                    hasil_perkalian_filter_2 = self.filter2 * gambar_output[
                                                              posisi_Y_sekarang:posisi_Y_sekarang + Y_FILTER,
                                                              posisi_X_sekarang:posisi_X_sekarang + X_FILTER]
                    hasil_penjumlahan_dari_perkalian_filter = np.sum(hasil_perkalian_filter_1) + np.sum(hasil_perkalian_filter_2)
                    hasil_perkalian_konvolusi[posisi_Y_hasil, posisi_X_hasil] = hasil_penjumlahan_dari_perkalian_filter

                    posisi_X_sekarang += self.stride
                    posisi_X_hasil += 1
                posisi_Y_sekarang += self.stride
                posisi_Y_hasil += 1

            if train:
                ProgressBar.printProgressBar(awal=awal, akhir=len(gambar_input), prefix="Konvolusi")
                awal += 1

            arr_hasil_perkalian_konvolusi.append(hasil_perkalian_konvolusi)
        return arr_hasil_perkalian_konvolusi
    def train(self, X_input, y_input, X_validation, y_validation, train=True):
        hasil_konvolusi_input = Conv2d.convolution(self, gambar_input = X_input, train=train)
        hasil_konvolusi_validation = Conv2d.convolution(self, gambar_input=X_validation, train=train)
        return np.array(hasil_konvolusi_input), np.array(hasil_konvolusi_validation)

class Relu:
    def __init__(self):
        pass

    def train(self, X_input, y_input, X_validation, y_validation, train=True):
        return np.maximum(0, X_input), np.maximum(0, X_validation)

class Maxpooling2D:
    def __init__(self, ukuran_filter=2, stride=1):
        self.ukuran_filter = ukuran_filter
        self.stride = stride

    def convolution(self, gambar_input, train):
        _, X, Y = gambar_input.shape

        arr_hasil_perkalian_konvolusi = []
        awal = 1
        for gambar in gambar_input:
            X_baru, Y_baru = gambar.shape

            # menghitung ukuran matriks hasil konvolusi
            ukuran_hasil_perkalian_X = int((X_baru - self.ukuran_filter) / self.stride) + 1
            ukuran_hasil_perkalian_Y = int((Y_baru - self.ukuran_filter) / self.stride) + 1
            hasil_perkalian_konvolusi = np.zeros((ukuran_hasil_perkalian_X, ukuran_hasil_perkalian_Y))

            # Proses maxpooling konvolusi
            posisi_Y_hasil = 0
            posisi_Y_sekarang = 0

            while posisi_Y_sekarang + self.ukuran_filter <= Y_baru:
                posisi_X_hasil = 0
                posisi_X_sekarang = 0

                while posisi_X_sekarang + self.ukuran_filter <= X_baru:
                    hasil_perkalian_konvolusi[posisi_Y_hasil, posisi_X_hasil] = np.max(gambar[
                                                              posisi_Y_sekarang:posisi_Y_sekarang + self.ukuran_filter,
                                                              posisi_X_sekarang:posisi_X_sekarang + self.ukuran_filter])

                    posisi_X_sekarang += self.stride
                    posisi_X_hasil += 1
                posisi_Y_sekarang += self.stride
                posisi_Y_hasil += 1

            if train:
                ProgressBar.printProgressBar(awal=awal, akhir=len(gambar_input), prefix="Maxpooling")
                awal += 1

            arr_hasil_perkalian_konvolusi.append(hasil_perkalian_konvolusi)
        return arr_hasil_perkalian_konvolusi
    def train(self, X_input, y_input, X_validation, y_validation, train=True):
        hasil_konvolusi_input = Maxpooling2D.convolution(self, gambar_input = X_input, train=train)
        hasil_konvolusi_validation = Maxpooling2D.convolution(self, gambar_input=X_validation, train=train)
        return np.array(hasil_konvolusi_input), np.array(hasil_konvolusi_validation)

class Flatten:
    def __init__(self):
        pass

    def train(self, X_input, y_input, X_validation, y_validation, train=True):
        X_input_flatten = []
        for train in X_input:
            X_input_flatten.append(np.array(train).flatten())

        X_validation_flatten = []
        for validation in X_validation:
            X_validation_flatten.append(np.array(validation).flatten())

        return np.array(X_input_flatten), np.array(X_validation_flatten)

class Dense:
    def __init__(self, hidden_layer=[]):
        self.hidden_layer = hidden_layer
        self.weight = {}
        self.bias = {}

    def set_params(self, epochs=10, callback=[]):
        self.epochs = epochs
        self.callback = callback

    def train(self, X_input, y_input, X_validation, y_validation, train=True):
        print()

        # Menambah layer input sesuai jumlah output dari flatten
        self.hidden_layer.insert(0, X_input.shape[1])
        self.hidden_layer.append(len(y_input[0]))

        # print(self.hidden_layer)
        # Inisialisasi bobot dan bias
        for i_layer in range(1, len(self.hidden_layer)):
            self.weight[i_layer] = np.random.randn(self.hidden_layer[i_layer], self.hidden_layer[i_layer-1]) / np.sqrt(self.hidden_layer[i_layer-1])
            self.bias[i_layer] = np.zeros((self.hidden_layer[i_layer], 1))

        # print(self.weight[4].shape)


        # for iterasi in range(self.epochs):
        for iterasi in range(1):
            result_sementara_perkalian_dan_tambah_neuron = {}
            result_sementara_aktivasi = {}
            result_sementara_weight = {}
            result_weight_baru = {}
            result_bias_baru = {}

            n = np.array(X_input).shape[1]


            #Forward Propagation
            X_input_transpose = np.array(X_input).T
            for i_layer in range(1, len(self.hidden_layer)):

                # Perkalian dot product input dan weight ditambah bias
                result = np.dot(self.weight[int(i_layer)], X_input_transpose) + self.bias[int(i_layer)]
                result_sementara_perkalian_dan_tambah_neuron[int(i_layer)] = result

                if(i_layer == len(self.hidden_layer)-1):
                    # Jika ini layer terakhir fungsi aktifasi sigmoid
                    X_input_transpose = DenseSupport.softmax(result)
                else:
                    #Jika ini bukan layer awal dan bukan layer akhir
                    X_input_transpose = DenseSupport.sigmoid(result)

                result_sementara_aktivasi[int(i_layer)] = X_input_transpose
                result_sementara_weight[int(i_layer)] = self.weight[int(i_layer)]


            # Loss function Categorical Cross Entropy
            loss_input = DenseSupport.loss(X_input_transpose.T, y_input)
            print(loss_input)

            # Menghitung Akurasi
            accuraacy_input = DenseSupport.accuracy(X_input_transpose.T, y_input)
            print(accuraacy_input)

            # Loss derivative
            # loss_derivative = DenseSupport.loss_derivative(X_input_transpose.T, y_input)
            # print(loss_derivative.shape)

            #Backpropagation
            A = result_sementara_aktivasi[len(self.hidden_layer)-1]
            dz = A.T - y_input
            result_weight_baru[len(self.hidden_layer)-1] = np.dot(dz, result_sementara_aktivasi[len(self.hidden_layer)-1]) / n
            result_bias_baru[len(self.hidden_layer) - 1] = np.sum(dz, axis=1, keepdims=True)
            # print(result_weight_baru[len(self.hidden_layer)-1])

            result_sementara_aktivasi[0] = np.array(X_input).T

            # input_derivative = loss_derivative.copy()
            for i_layer in range(len(self.hidden_layer)-1, 0, -1):
                pass
                # print(i_layer)
                # print(np.dot(self.weight[int(i_layer)], input_derivative.T))
                # weight_derivative = np.dot(self.weight[int(i_layer)].T, input_derivative)
                # bias_derivative = np.sum(input_derivative, axis=0, keepdims=True)
                # print(input_derivative.shape)
                # print(weight_derivative.shape)
                # input_derivative = np.dot(input_derivative, weight_derivative.T)

            # print(input_derivative)

        return X_input, X_validation

class DenseSupport:
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    def softmax(X):
        eksponen = np.exp(X - np.max(X))
        return eksponen / np.sum(eksponen, axis=1, keepdims=True)

    def get_cost(X,y):
        prediksi = np.clip(X, 1e-7, 1 - 1e-7)
        correct_confidences = np.sum(X * y, axis=1)
        loss = -np.log(correct_confidences)
        return loss

    def accuracy(X, y):
        loss = DenseSupport.get_cost(X,y)
        prediksi = np.argmax(X, axis=1)
        kelas_sebenarnya = np.argmax(y, axis=1)
        akurasi = np.mean(prediksi == kelas_sebenarnya)
        return akurasi

    def loss(X, y):
        loss = DenseSupport.get_cost(X,y)
        loss = np.mean(loss)
        # loss = -np.mean(y * np.log(X.T))
        return loss

    def loss_derivative(X, y):
        sample = len(X)

        y_T = np.argmax(y, axis=1)
        input_derivative_dari_loss = X.copy()
        input_derivative_dari_loss[range(sample), y_T] -= 1
        input_derivative_dari_loss = input_derivative_dari_loss / sample
        return input_derivative_dari_loss