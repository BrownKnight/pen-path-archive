import glob

from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.layers import Input, LSTM, TimeDistributed, Dense, Masking, Bidirectional, Concatenate
from tensorflow.keras import models, losses, optimizers, activations

import numpy as np
import matplotlib.pyplot as plt

# MODEL_PATH = "models/model_300_neurons_0.00001_lr_char-01-000-*-*.h5"
MODEL_PATH = "models/bi-lstm-seq2seq-300_epoch.h5"
TEST_SPLIT = 0.1


def capped_relu(x):
    return activations.relu(x, max_value=1.0)


def main():
    x_data = load_x("test.nosync/image_output/char-01-000-*-*.csv")
    normalize_x(x_data)
    y_data = load_y("test.nosync/ground_truth/char-01-000-*-*.txt")
    normalize_y(y_data)

    model = create_model()
    train_model(model, x_data, y_data)

    model = models.load_model(MODEL_PATH, custom_objects={"capped_relu": capped_relu})
    print(model.summary())

    predict(model, x_data, y_data)


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
    data = np.zeros((num_files, 128, 3))

    for index, file_path in enumerate(file_paths):
        with open(file_path) as file:
            lines = file.readlines()
            char = np.asarray([np.asarray(point.split(",")[:3]) for point in lines])
            data[index] = char

    print("Loaded %s character image files" % num_files)

    return data


def normalize_x(data):
    data[:, :, 0] /= 64
    data[:, :, 1] /= 64
    data[:, :, 2] /= 3


def create_model():
    # ENCODER DECODER
    encoder_inputs = Input(shape=(128, 3))
    masked_encoder_inputs = Masking()(encoder_inputs)
    encoder_lstm = Bidirectional(LSTM(256, return_state=True))

    # We discard `encoder_outputs` and only keep the states.
    _, forward_h, forward_c, backward_h, backward_c = encoder_lstm(masked_encoder_inputs)
    state_c = Concatenate()([forward_c, backward_c])
    state_h = Concatenate()([forward_h, backward_h])
    encoder_states = [state_h, state_c]

    # Bottleneck Here

    decoder_inputs = Input(shape=(128, 3))
    masked_decoder_inputs = Masking()(decoder_inputs)
    decoder_lstm = LSTM(512, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_lstm(masked_decoder_inputs, initial_state=encoder_states)

    outputs = TimeDistributed(Dense(2, activation=capped_relu))(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], outputs)

    # # SEQUENTIAL
    # model = models.Sequential([
    #     Input((128, 3)),
    #     Masking(),
    #     Bidirectional(LSTM(512, return_sequences=True)),
    #     Bidirectional(LSTM(512, return_sequences=True)),
    #     TimeDistributed(Dense(2, activation=capped_relu))
    # ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.00002), loss=losses.MeanAbsoluteError(),
                  metrics=["accuracy"])

    print(model.summary())

    return model


def train_model(model: models.Sequential, train_x: np.ndarray, train_y: np.ndarray):
    print("Training Model")
    history = model.fit([train_x, train_x], train_y, batch_size=64, epochs=300, verbose=1, validation_split=TEST_SPLIT,
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


def predict(model: models.Sequential, test_data: np.ndarray, ground_truth: np.ndarray):
    result = model.predict([test_data, test_data])

    data_index = 3

    result = result[data_index] * 64
    print(result)
    np.savetxt("test.nosync/result.txt", result)
    result_image = create_image_from_data(result)

    test = test_data[data_index] * 64
    test_image = create_image_from_data(test)

    ground_truth = ground_truth[data_index] * 64
    ground_truth = create_image_from_data(ground_truth)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    colorbar1 = ax1.imshow(test_image)
    fig.colorbar(colorbar1, ax=ax1)
    colorbar2 = ax2.imshow(result_image)
    fig.colorbar(colorbar2, ax=ax2)
    colorbar3 = ax3.imshow(ground_truth)
    fig.colorbar(colorbar3, ax=ax3)

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
