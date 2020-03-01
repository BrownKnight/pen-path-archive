import glob
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

NUM_OF_TRAIN_SAMPLES = 40000


def load_x_data(file_dir):
    """Loads all the character data from a given directory and outputs it into a numpy array"""
    file_list = glob.glob(file_dir)
    file_list.sort()
    num_of_files = len(file_list)
    print("Found %s character files" % num_of_files)
    comparison_list = []

    # Create an array of character point arrays
    # 64,64,1 for 2D / 128, 3 for 1D
    chars = np.zeros((num_of_files, 64, 64, 1), dtype=float)
    for index, file in enumerate(file_list):
        comparison_list.append(Path(file).stem)
        with open(file) as f:
            lines = f.readlines()
            lines = [line.strip("\n").split(",") for line in lines]
            # FOR 2D STUFF
            char = np.zeros((64, 64))
            for line in lines:
                char[int(line[1]), int(line[0])] = float(line[2])
            char = char.reshape((64, 64, 1))
            # char = [(int(line[0]), int(line[1]), int(line[2])) for line in lines]
            chars[index] = char
    # normalise the character data
    normalise_x(chars)

    # Split the data into train/test data
    train_data, test_data = chars[:NUM_OF_TRAIN_SAMPLES], chars[NUM_OF_TRAIN_SAMPLES:]
    return train_data, test_data, comparison_list


def normalise_x(chars):
    # FOR 2D ONLY
    chars[:, :, :, 0] /= 3
    # For 1D Only
    # chars[:, :, 0] /= 63
    # chars[:, :, 1] /= 63
    # chars[:, :, 2] /= 2


def denormalise_x(chars):
    # FOR 2D ONLY
    chars[:, :, :, 0] *= 3
    # For 1D Only
    # chars[:, :, 0] *= 63
    # chars[:, :, 1] *= 63
    # chars[:, :, 2] *= 2


def load_y_data(file_dir, comparison_list):
    """Loads all the ground truth character data from a given directory and outputs it into a numpy array"""
    file_list = glob.glob(file_dir)
    # if comparison_list is not None:
    #     file_list = [file for file in file_list if Path(file).stem in [Path(file).stem for file in comparison_list]]
    file_list.sort()
    num_of_files = len(comparison_list)
    print("Found %s ground truth character files" % num_of_files)

    # Create an array of character point arrays
    # 64,64,1 for 2D / 128, 3 for 1D
    chars = np.zeros((num_of_files, 64, 64, 1), dtype=float)
    index = 0
    for file in file_list:
        if Path(file).stem not in comparison_list:
            continue

        with open(file) as f:
            lines = f.readlines()
            lines = [line.strip("\n").split(",") for line in lines]
            # FOR 2D STUFF
            char = np.zeros((64, 64))
            for line in lines:
                char[int(line[1]), int(line[0])] = float(line[2])
            char = char.reshape((64, 64, 1))
            # FOR 1D STUFF
            # char = [(int(line[0]), int(line[1]), int(line[2]), int(line[3])) for line in lines]
            chars[index] = char
            index += 1
    # normalise the character data
    normalise_y(chars)

    # Split the data into train/test data
    train_data, test_data = chars[:NUM_OF_TRAIN_SAMPLES], chars[NUM_OF_TRAIN_SAMPLES:]
    return train_data, test_data


def normalise_y(chars):
    # FOR 2D ONLY
    chars[:, :, :, 0] /= 128
    # For 1D Only
    # chars[:, :, 0] /= 63
    # chars[:, :, 1] /= 63
    # chars[:, :, 2] /= 2
    # chars[:, :, 3] /= 128


def denormalise_y(chars):
    # FOR 2D ONLY
    chars[:, :, :, 0] *= 128
    # For 1D Only
    # chars[:, :, 0] *= 63
    # chars[:, :, 1] *= 63
    # chars[:, :, 2] *= 2
    # chars[:, :, 3] *= 128


def create_network():
    autoencoder = models.Sequential()
    autoencoder.add(layers.Input(shape=(64, 64, 1)))
    autoencoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(layers.MaxPooling2D((2, 2)))
    autoencoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(layers.MaxPooling2D((2, 2)))
    # Bottleneck Here
    autoencoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(layers.UpSampling2D((2, 2)))
    autoencoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(layers.UpSampling2D((2, 2)))
    autoencoder.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    print(autoencoder.summary())

    autoencoder.compile(optimizer='adam',
                        loss=tf.keras.losses.binary_crossentropy,
                        metrics=['accuracy'])
    return autoencoder


def train():
    train_x, test_x, file_list = load_x_data("test/image_output/char-*.csv")
    train_y, test_y = load_y_data("test/ground_truth/char-*.txt", file_list)

    model = create_network()

    history = model.fit(train_y, train_y, batch_size=50, epochs=8, verbose=1,
                        validation_data=(test_y, test_y))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(train_x, train_y, verbose=2)
    print("Loss: %s" % test_loss)
    print("Acc: %s" % test_acc)
    model.save("model.h5")


def predict():
    model = models.load_model("model.h5")
    model.summary()
    input_data, _, file_list = load_x_data("test/image_output/char-01-08-002.csv")
    # input_data, _ = load_y_data("test/ground_truth/char-01-08-002.txt", file_list)
    ground_truth, _ = load_y_data("test/ground_truth/char-01-08-002.txt", file_list)

    denormalise_x(input_data)
    # denormalise_y(input_data)
    input_data = input_data[:1]

    denormalise_y(ground_truth)
    ground_truth = ground_truth[:1]

    result = model.predict(input_data)
    denormalise_y(result)
    with open('test/result.txt', 'w+') as result_file:
        result_file.writelines(["%s" % line for line in result])
    # print(result)

    input_image = create_image_from_coords(input_data[0])
    output_image = create_image_from_coords(result[0])
    ground_truth_image = create_image_from_coords(ground_truth[0])

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)

    colorbar1 = ax1.imshow(input_image)
    fig.colorbar(colorbar1, ax=ax1)
    colorbar2 = ax2.imshow(output_image)
    fig.colorbar(colorbar2, ax=ax2)
    colorbar3 = ax3.imshow(ground_truth_image)
    fig.colorbar(colorbar3, ax=ax3)

    plt.show()


def create_image_from_coords(input_image: np.ndarray):
    image = np.zeros((64, 64))
    input_image = input_image.astype(int)
    if input_image.shape == (128, 3):
        for coord in input_image:
            image[coord[1], coord[0]] = coord[2] + 1
    else:
        image = input_image.reshape((64, 64))

    return image


if __name__ == "__main__":
    # create_network()
    # train()
    predict()
