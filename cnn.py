from khisoft.preprocessing import ImageDataGenerator
from khisoft.layers import Model, Conv2d, Relu, Maxpooling2D, Flatten, Dense
# from khisoft.output_layers import Softmax
# from khisoft.loss import CategoricalCrossEntropy
# from khisoft.optimizer import SGD
import matplotlib.pyplot as plt
from mnist.loader import MNIST
import numpy as np
from sklearn.preprocessing import OneHotEncoder

mndata = MNIST('data_files')
mndata.gz = True
images, labels = mndata.load_training()
train_x = np.array(images)
train_y = np.array(labels)

images, labels = mndata.load_testing()
test_x = np.array(images)
test_y = np.array(labels)


train_x = train_x[:100] / 255
test_x = test_x[:100] / 255

enc = OneHotEncoder(sparse=False, categories='auto')
train_y = enc.fit_transform(train_y[:100].reshape(len(train_y[:100]), -1))
test_y = enc.fit_transform(test_y[:100].reshape(len(test_y[:100]), -1))

train_x_jadi = []
for i in train_x:
    train_x_jadi.append(np.asarray(i.reshape(28,28)))

test_x_jadi = []
for i in test_x:
    test_x_jadi.append(np.asarray(i.reshape(28,28)))

train_x_jadi = np.asarray(train_x_jadi)
test_x_jadi = np.asarray(test_x_jadi)
# dataset = ImageDataGenerator("datasets")
# X_train, y_train, X_test, y_test = dataset.load_dataset(rescale=255)
# y_train_oneHot, y_test_oneHot = dataset.load_class_oneHot()
# plt.imshow(X_train[0], cmap='gray')
# plt.show()


model = Model([
    Conv2d(padding=1,stride=2),
    Maxpooling2D(ukuran_filter=2, stride=2),
    Flatten(),
    Dense([11]),
])

model.summary(input_shape = (28,28))
# # model.compile(loss=CategoricalCrossEntropy(), optimizer=SGD(lr=0.1, momentum=0.9), output_layer=Softmax())
model.fit(epochs=200, X_input=train_x_jadi, y_input=train_y, X_validation=test_x_jadi, y_validation=test_y, callback=['plot_loss','plot_accuraacy'])
predict = model.pred(X_input=np.array([train_x_jadi[0]]))
predict_arg_max = predict.argmax(axis=0)
# print("====  CNN ==== ")
# print(predict)
# print(predict_arg_max)