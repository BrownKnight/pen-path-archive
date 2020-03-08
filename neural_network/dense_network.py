import glob

from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Masking, Embedding, \
    Reshape, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, UpSampling2D
from tensorflow.keras import models, losses, optimizers

import numpy as np
import matplotlib.pyplot as plt

# MODEL_PATH = "models/model_300_neurons_0.00001_lr_char-01-000-*-*.h5"

MODEL_PATH = "models/dense_autoencoder.h5"
TEST_SPLIT = 0.1


def main():
    x_data = load_y("test.nosync/image_output/char-*-000-*-*.csv")
    normalize_y(x_data)
    target_data = load_y("test.nosync/ground_truth/char-*-000-*-*.txt")
    normalize_y(target_data)

    model = create_model()
    train_model(model, x_data, target_data)

    # test_data = load_y("test.nosync/ground_truth/char-*-000-*-*.txt")
    test_data = load_y("test.nosync/image_output/char-*-000-*-*.csv")
    normalize_y(test_data)
    model = models.load_model(MODEL_PATH)
    print(model.summary())
    predict(model, test_data)


def load_y(path):
    print("Loading target data from %s" % path)
    file_paths = glob.glob(path)
    file_paths.sort()
    num_files = len(file_paths)
    data = np.zeros((num_files, 64, 64, 1))

    for index, file_path in enumerate(file_paths):
        with open(file_path) as file:
            lines = file.readlines()
            lines = [np.asarray(point.split(',')[:2], dtype=int) for point in lines]
            char = np.zeros((64, 64, 1))
            for j, point in enumerate(lines):
                if point[0] == 0 and point[1] == 0:
                    continue
                char[point[1], point[0], 0] = j + 1
            data[index] = char

    print("Loaded %s character target data files" % num_files)

    # np.random.shuffle(data)
    return data


def normalize_y(data):
    data /= 128


def load_x(path):
    print("Loading image data from %s" % path)
    file_paths = glob.glob(path)
    file_paths.sort()
    num_files = len(file_paths)
    data = np.zeros((num_files, 128, 2))

    for index, file_path in enumerate(file_paths):
        with open(file_path) as file:
            lines = file.readlines()
            char = np.asarray([np.asarray(point.split(",")[:2]) for point in lines])
            data[index] = char

    print("Loaded %s character image files" % num_files)

    return data


def normalize_x(data):
    data /= 64


def create_model():
    model = models.Sequential([
        Input((64, 64, 1)),
        Dense(128),
        LeakyReLU(),
        Dense(256),
        LeakyReLU(),
        Dense(256),
        LeakyReLU(),
        Dense(128),
        LeakyReLU(),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.00001), loss=losses.MeanAbsoluteError(),
                  metrics=["accuracy"])

    print(model.summary())

    return model


def train_model(model: models.Sequential, train_x: np.ndarray, train_y: np.ndarray):
    print("Training Model")
    history = model.fit(train_x, train_y, batch_size=32, epochs=150, verbose=1, validation_split=TEST_SPLIT,
                        shuffle=True)

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1])
    plt.legend(loc='lower right')
    plt.show()

    model.save(MODEL_PATH)


def predict(model: models.Sequential, test_data: np.ndarray):
    result = model.predict(test_data)

    result = result[0] * 128
    result = result.reshape((64, 64))
    print(result)

    test = test_data[0] * 128
    test = test.reshape((64, 64))
    # test_image = create_image_from_data(test)

    # write the result array to file

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    colorbar1 = ax1.imshow(test)
    fig.colorbar(colorbar1, ax=ax1)
    colorbar2 = ax2.imshow(result)
    fig.colorbar(colorbar2, ax=ax2)

    plt.show()


if __name__ == "__main__":
    main()
