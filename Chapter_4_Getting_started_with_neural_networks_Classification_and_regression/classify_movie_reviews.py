import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def get_imdb_dataset(num_words=10000):
    """
    Get the imdb dataset
    :param num_words: number of top occurrence words
    :return:
    """

    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.\
        load_data(num_words=num_words)

    return train_data, train_labels, test_data, test_labels


def decode_vector_to_text(review):
    """
    Decode the index vector to text.
    :param review: A vector of review
    :return:
    """

    word_to_index = tf.keras.datasets.imdb.get_word_index()
    index_to_word = dict([(value, key) for (key, value) in word_to_index.items()])

    text = " ".join([index_to_word.get(i - 3, '?') for i in review])

    return text


def convert_to_multi_hot_encoding(sequences, dimension):
    """
    Convert the reviews indices to multi-hot encoding
    :param sequences:
    :param dimension:
    :return:
    """

    multi_hot_vector = np.zeros(shape=(len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        for j in sequence:
            multi_hot_vector[i, j] = 1

    return multi_hot_vector.astype('float32')


def build_model(input_shape):
    """
    Build a simple neural network with two dense layers.
    :param input_shape: shape of the input data
    :return:
    """

    X_input = tf.keras.layers.Input(shape=input_shape)
    X = tf.keras.layers.Dense(units=16, activation='relu')(X_input)
    X = tf.keras.layers.Dense(units=16, activation='relu')(X)
    X_output = tf.keras.layers.Dense(units=16, activation='sigmoid')(X)

    model = tf.keras.models.Model(inputs=X_input, outputs=X_output)

    return model


def plot_train_val_loss(train_loss_values, val_loss_values):
    """
    Plot the train and validation loss values.
    :param train_loss_values: loss values of train dataset
    :param val_loss_values: loss values of validation dataset
    :return:
    """

    epochs = range(1, len(train_loss_values) + 1)

    plt.plot(epochs, train_loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.savefig('loss.png')


def plot_train_val_accuracy(train_acc_values, val_acc_values):
    """
    Plot the train and validation accuracy values.
    :param train_acc_values: loss values of train dataset
    :param val_acc_values: loss values of validation dataset
    :return:
    """

    epochs = range(1, len(train_acc_values) + 1)

    plt.plot(epochs, train_acc_values, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.savefig('accuracy.png')


# Get imdb dataset
train_data, train_labels, test_data, test_labels = get_imdb_dataset(num_words=10000)

# Split train_data into train and validation subsets.
val_data = train_data[:10000]
val_labels = train_labels[:10000]
y_val = np.array(val_labels, dtype='float32')

train_data = train_data[10000:]
train_labels = train_labels[10000:]
y_train = np.array(train_labels, dtype='float32')

y_test = np.array(test_labels, dtype='float32')

# Decode review indices to text
# decoded_review = decode_vector_to_text(x_train[0])
# print(decoded_review)

# Convert review indices to multi-hot encoding
x_train = convert_to_multi_hot_encoding(sequences=train_data, dimension=10000)
x_val = convert_to_multi_hot_encoding(sequences=val_data, dimension=10000)
x_test = convert_to_multi_hot_encoding(sequences=test_data, dimension=10000)

# Build and compile the model
model = build_model(input_shape=10000)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Fit model
history = model.fit(x=x_train, y=train_labels, validation_data=(x_val, val_labels),
                    epochs=20, batch_size=512)

# Plot loss and accuracy
plot_train_val_accuracy(train_loss_values=history.history['loss'],
                        val_loss_values=history.history['val_loss'])

plot_train_val_accuracy(train_acc_values=history.history['accuracy'],
                        val_acc_values=history.history['val_accuracy'])

# Test model
test_result = model.evaluate(x_test, y_test)
