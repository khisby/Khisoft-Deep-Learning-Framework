from khisoft.preprocessing import ImageDataGenerator
from khisoft.layers import Model, Conv2d, Relu, Maxpooling2D, Flatten, Dense
# from khisoft.output_layers import Softmax
# from khisoft.loss import CategoricalCrossEntropy
# from khisoft.optimizer import SGD

import matplotlib.pyplot as plt

dataset = ImageDataGenerator("datasets")
X_train, y_train, X_test, y_test = dataset.load_dataset(rescale=255)
y_train_oneHot, y_test_oneHot = dataset.load_class_oneHot()
#
# plt.imshow(X_train[0], cmap='gray')
# plt.show()

model = Model([
    Conv2d(padding=1,stride=2),
    Relu(),
    Maxpooling2D(ukuran_filter=3, stride=3),
    Maxpooling2D(ukuran_filter=2, stride=1),
    Maxpooling2D(ukuran_filter=3, stride=2),
    Maxpooling2D(ukuran_filter=3, stride=2),
    Maxpooling2D(ukuran_filter=3, stride=3),
    Flatten(),
    Dense([20,30,40]),
])
model.summary(input_shape = (200,200))
# model.compile(loss=CategoricalCrossEntropy(), optimizer=SGD(lr=0.1, momentum=0.9), output_layer=Softmax())
model.fit(epochs=100, X_input=X_train, y_input=y_train_oneHot, X_validation=X_test, y_validation=y_test_oneHot, callback=['plot_loss','plot_accuraacy'])
