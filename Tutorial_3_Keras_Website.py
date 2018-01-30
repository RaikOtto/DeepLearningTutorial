
### https://www.learnopencv.com/deep-learning-using-keras-the-basics/

from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import boston_housing

(X_train, Y_train), (X_test, Y_test) = boston_housing.load_data()

nFeatures = X_train.shape[1]

model = Sequential()
model.add(Dense(1, input_shape=(nFeatures,), activation='linear'))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])

model.fit(X_train, Y_train, batch_size=4, epochs=1000)

model.summary()

model.evaluate(X_test, Y_test, verbose=True)

Y_pred = model.predict(X_test)

print Y_test[:5]
print Y_pred[:5, 0]

### https://keras.io/

from keras.models import Sequential

model = Sequential()
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

model.compile(loss='categorical_crossentropy',optimizer=s.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
model.fit(x_train, y_train, epochs=5, batch_size=32)

model.train_on_batch(x_batch, y_batch)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
classes = model.predict(x_test, batch_size=128)
