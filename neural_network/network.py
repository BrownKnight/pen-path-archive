import glob
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

NUM_OF_TRAIN_SAMPLES = 70
NUM_OF_TEST_SAMPLES = 15


def load_x_data(file_dir):
    """Loads all the character data from a given directory and outputs it into a numpy array"""
    file_list = glob.glob(file_dir)
    num_of_files = len(file_list)
    print("Found %s character files" % num_of_files)

    # Create an array of character point arrays
    chars = np.zeros((num_of_files, 128, 3))
    for index, file in enumerate(file_list):
        with open(file) as f:
            lines = f.readlines()
            lines = [line.strip("\n").split(",") for line in lines]
            char = [(line[0], line[1], line[2]) for line in lines]
            chars[index] = np.array(char)

    train_data, test_data = chars[:NUM_OF_TRAIN_SAMPLES], chars[NUM_OF_TRAIN_SAMPLES:NUM_OF_TRAIN_SAMPLES+NUM_OF_TEST_SAMPLES]
    train_data, test_data = train_data / 255, test_data / 255
    return train_data, test_data


def create_network():
    model = models.Sequential()
    model.add(layers.Conv1D(32, 16, input_shape=(128, 3), padding="same"))
    model.add(layers.UpSampling1D(16))
    model.add(layers.Conv1D(64, 16, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv1D(128, 16, strides=4, activation='tanh', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv1D(64, 16, strides=2, activation='tanh', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv1D(32, 16, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(3, 16, activation='tanh', padding="same"))
    print(model.summary())
    return model


def main():
    train_x, test_x = load_x_data("test/image_output/*.csv")
    train_y, test_y = load_x_data("test/ground_truth/*.txt")

    model = create_network()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_x, train_y, epochs=8, validation_data=(test_x, test_y))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(train_x, train_y, verbose=2)
    print("Loss: %s" % test_loss)
    print("Acc: %s" % test_acc)


if __name__ == "__main__":
    main()
