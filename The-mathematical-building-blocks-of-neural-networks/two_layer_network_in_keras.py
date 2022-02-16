from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential

(X_train, y_train), (X_test, y_test) = mnist.load_data()

model = Sequential([
    Dense(units=512, activation='relu'),
    Dense(units=10, activation='softmax')])

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

X_train = X_train.reshape((X_train.shape[0], -1))
X_train = X_train.astype('float32') / 255

X_test = X_test.reshape((X_test.shape[0], -1))
X_test = X_test.astype('float32') / 255

model.fit(x=X_train, y=y_train, epochs=5, batch_size=128)

result = model.evaluate(X_test, y_test)

print(f'result on test data:\n{result}')
