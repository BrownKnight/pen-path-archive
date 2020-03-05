# from sklearn.model_selection import train_test_split
#
# # Split the data
# x_train, x_valid, y_train, y_valid = train_test_split(data, labels, test_size=0.33, shuffle= True)
import glob

from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Masking, Embedding, \
    Reshape
from tensorflow.keras import models, losses, optimizers

import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = "models/model_300_neurons_0.00001_lr_char-01-000-*-*.h5"
TEST_SPLIT = 0.1


def main():
    x_data = load_x("test.nosync/image_output/char-*-000-*-*.csv")
    target_data = load_y("test.nosync/ground_truth/char-*-000-*-*.txt")
    normalize_x(x_data)
    normalize_y(target_data)
    # print(target_data)

    model = create_model()
    train_model(model, x_data, target_data)

    test_data = load_x("test.nosync/image_output/char-01-000-*-*.csv")
    normalize_x(test_data)
    model = models.load_model(MODEL_PATH)
    print(model.summary())
    predict(model, test_data)


def load_y(path):
    print("Loading target data from %s" % path)
    file_paths = glob.glob(path)
    file_paths.sort()
    num_files = len(file_paths)
    data = np.zeros((num_files, 128, 2))

    for index, file_path in enumerate(file_paths):
        with open(file_path) as file:
            lines = file.readlines()
            char = np.asarray([np.asarray(point.split(",")) for point in lines])
            data[index] = char

    print("Loaded %s character target data files" % num_files)

    return data


def normalize_y(data):
    data /= 64


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
    encoder_inputs = Input(shape=(128, 2))
    masked_encoder_inputs = Masking()(encoder_inputs)
    encoder_lstm = LSTM(100, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(masked_encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(128, 2), )
    masked_decoder_inputs = Masking()(decoder_inputs)
    decoder_lstm = LSTM(100, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_lstm(masked_decoder_inputs, initial_state=encoder_states)

    outputs = TimeDistributed(Dense(2, activation='sigmoid'))(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.00005), loss=losses.MeanAbsoluteError(),
                  metrics=["accuracy"])

    print(model.summary())

    return model


def train_model(model: models.Sequential, train_x: np.ndarray, train_y: np.ndarray):
    print("Training Model")
    history = model.fit([train_x, train_x], train_y, batch_size=256, epochs=400, verbose=1, validation_split=TEST_SPLIT,
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
    result = model.predict([test_data, test_data])

    result = result[0] * 64
    # result = result[0]
    print(result)
    np.savetxt("test.nosync/result.txt", result)
    result_image = create_image_from_data(result)

    test = test_data[0] * 64
    test_image = create_image_from_data(test)

    # write the result array to file

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    colorbar1 = ax1.imshow(test_image)
    fig.colorbar(colorbar1, ax=ax1)
    colorbar2 = ax2.imshow(result_image)
    fig.colorbar(colorbar2, ax=ax2)

    plt.show()


def create_image_from_data(data: np.ndarray):
    image = np.zeros((64, 64), float)
    for i, point in enumerate(data):
        x = int(point[0])
        y = int(point[1])

        if x == 0 and y == 0:
            continue

        image[y, x] = i + 1

    return image


if __name__ == "__main__":
    main()
