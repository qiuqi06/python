import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

np.random.seed(1337)  # from reproducibility

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1) / 255
x_test = x_test.reshape(x_test.shape[0], -1) / 255

y_train = np_utils.to_categorical(y_train, nb_class=10)
y_test = np_utils.to_categorical(y_test, nb_classes=10)

model = Sequential([
	Dense(32, input_dim=784),
	Activation('relu'),
	Dense(10),
	Activation('softmax')
])
rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0)
model.compile(
	optimizer=rmsprop,
	loss='categorical_crossentropy',
	metrics=['accuracy']
)

print('\nTraining----------------------------')
model.fit(x_train, y_train, nb_epoch=2, batch_size=32)

print('\ntesting-----------------')

loss, accuracy = model.evaluate(x_test, y_test)

print('loss', loss)
print('accuracy', accuracy)
