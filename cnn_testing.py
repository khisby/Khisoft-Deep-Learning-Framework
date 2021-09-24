from khisoft.preprocessing import ImageDataGenerator
from khisoft.layers import Model, Conv2d, Relu, Maxpooling2D, Flatten, Dense
# from khisoft.output_layers import Softmax
# from khisoft.loss import CategoricalCrossEntropy
# from khisoft.optimizer import SGD
import numpy as np
import copy
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dataset = ImageDataGenerator("data")
X_train, y_train, X_test, y_test = dataset.load_dataset(rescale=255)
y_train_oneHot, y_test_oneHot = dataset.load_class_oneHot()

model = Model([
    Conv2d(padding=1,stride=2),
    Relu(),
    Maxpooling2D(ukuran_filter=2, stride=2),
    Flatten(),
    Dense([11]),
])

# model.summary(input_shape = (28,28))
# # model.compile(loss=CategoricalCrossEntropy(), optimizer=SGD(lr=0.1, momentum=0.9), output_layer=Softmax())
# epochs = 30000
# model.fit(epochs=epochs, X_input=X_train, y_input=y_train_oneHot, X_validation=X_test, y_validation=y_test_oneHot, callback=['plot_loss','plot_accuraacy'])
# model.plot()

classes = np.load('model/class.npy', allow_pickle=True)


# gc = cv2.imread("testing/normal/kamu/gambar_crop_asli.png", cv2.IMREAD_UNCHANGED)
# gc = cv2.cvtColor(gc, cv2.COLOR_BGR2HSV)
# lower_blue = np.array([0, 39, 75])
# upper_blue = np.array([255, 255, 214])
# mask = cv2.inRange(gc, lower_blue, upper_blue)
# gc = cv2.bitwise_and(gc, gc, mask=mask)
# gc = cv2.cvtColor(gc, cv2.COLOR_HSV2RGB)
# gc = cv2.cvtColor(gc, cv2.COLOR_RGB2GRAY)
# ret, gc = cv2.threshold(gc, 60, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# model_baru = copy.deepcopy(model)
# predict = model_baru.pred(X_input=np.array([gc]))
# predict_arg_max = predict.argmax(axis=0)
# predict_class = classes[predict_arg_max[0]]
# prediksi = predict_class
# print("Hasil Prediksi " + prediksi)

list = os.listdir("testing")
for i in list:
    list1 = os.listdir("testing/"+i)
    for j in list1:
        folder = "testing/"+i+'/'+j+'/'
        sebenarnya = j
        tipe = i
        gambar_crop_asli = folder + "gambar_crop_asli.png"
        gambar_crop_grayscale = folder + "gambar_crop_grayscale.png"
        gambar_crop_biner = folder + "gambar_crop_biner.png"
        img = cv2.imread(gambar_crop_biner, cv2.IMREAD_UNCHANGED)

        model_baru = copy.deepcopy(model)
        predict = model_baru.pred(X_input=np.array([np.asarray(img)]))
        predict_arg_max = predict.argmax(axis=0)
        predict_class = classes[predict_arg_max[0]]
        prediksi = predict_class
        print("{},{},{},{},{},{}".format(tipe,sebenarnya,prediksi, gambar_crop_asli,gambar_crop_grayscale, gambar_crop_biner))

