import tensorflow as tf
import numpy as np


class NaiveDense:
    def __init__(self, input_shape, output_shape, activation):

        self.activation = activation

        w_initial_value = tf.random.uniform((input_shape, output_shape), minval=0, maxval=0.1)
        self.W = tf.Variable(w_initial_value)

        b_initial = tf.zeros((output_shape,))
        self.b = tf.Variable(b_initial)

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):
        return [self.W, self.b]


class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs

        for layer in self.layers:
            x = layer(x)

        return x

    @property
    def weights(self):
        weights = []

        for layer in self.layers:
            weights += layer.weights

        return weights


class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)

        self.index = 0
        self.batch_size = batch_size
        self.images = images
        self.labels = labels
        self.num_batches = int(np.ceil(len(self.images) / self.batch_size))

    def next(self):
        images = self.images[self.index:self.index + self.batch_size]
        labels = self.labels[self.index:self.index + self.batch_size]

        self.index += self.batch_size

        return images, labels


def one_training_step(model, optimizer, images_batch, labels_batch):

    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(labels_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_losses)

    gradients = tape.gradient(average_loss, model.weights)

    optimizer.apply_gradients(zip(gradients, model.weights))

    return average_loss


def fit(model, images, labels, epochs, optimizer, batch_size=128):

    for epoch in range(epochs):
        print(f'Epoch {epoch}')

        batch_generator = BatchGenerator(images=images, labels=labels, batch_size=batch_size)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()

            loss = one_training_step(model=model, optimizer=optimizer,
                                     images_batch=images_batch, labels_batch=labels_batch)

            print(f'loss at batch {batch_counter}: {loss:.2f}')


# Build model
naive_model = NaiveSequential([
    NaiveDense(input_shape=28*28, output_shape=512, activation=tf.nn.relu),
    NaiveDense(input_shape=512, output_shape=10, activation=tf.nn.softmax),
])

sgd_optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)

# Load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], -1))
X_train = X_train.astype('float32') / 255

X_test = X_test.reshape((X_test.shape[0], -1))
X_test = X_test.astype('float32') / 255

# Train model
fit(model=naive_model, images=X_train, labels=y_train, epochs=5, optimizer=sgd_optimizer, batch_size=128)

# Evaluation model
predictions = naive_model(X_test)
predictions = predictions.numpy()
predicted_labels = predictions.argmax(axis=1)
matches = predicted_labels == y_test
print(f"accuracy: {matches.mean():.2f}")

