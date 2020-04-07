from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow_core.python.keras.layers import Input, LSTM, TimeDistributed, Dense, Bidirectional, Concatenate, \
    RepeatVector
from tensorflow.keras import models, losses, optimizers, activations

import numpy as np
import matplotlib.pyplot as plt

# MODEL_PATH = "models/model_300_neurons_0.00001_lr_char-01-000-*-*.h5"
from image_data_utils import load_y, normalize_y, load_x, normalize_x, create_image_from_data

MODEL_PATH = "models/bi-lstm-s2s-all_data_w_rotation-training.h5"
# MODEL_PATH = "models/auto_save.h5"
TEST_SPLIT = 0.1


def capped_relu(x):
    return activations.relu(x, max_value=1.0)


def main():
    x_path = "test.nosync/image_output/char-*-*-*-*.csv"
    y_path = "test.nosync/ground_truth/char-*-*-*-*.txt"

    print("Loading image data from %s" % x_path)
    x_data = load_x(x_path)
    normalize_x(x_data)
    print("Loaded %s character image files" % len(x_data))

    print("Loading target data from %s" % y_path)
    y_data = load_y(y_path)
    normalize_y(y_data)
    print("Loaded %s character target data files" % len(y_data))

    print("Shuffling Data")
    np.random.seed(42)
    randomize = np.arange(len(x_data))
    np.random.shuffle(randomize)
    x_data = x_data[randomize]
    y_data = y_data[randomize]

    # model = create_model()
    model = models.load_model(MODEL_PATH, custom_objects={"capped_relu": capped_relu})
    print(model.summary())

    train_model(model, x_data, y_data)

    test(model, x_data, y_data)


def create_model():
    # # ENCODER DECODER
    # encoder_inputs = Input(shape=(128, 3))
    # # masked_encoder_inputs = Masking()(encoder_inputs)
    # masked_encoder_inputs = encoder_inputs
    # encoder_lstm = Bidirectional(LSTM(500, return_state=True))
    #
    # # We discard `encoder_outputs` and only keep the states.
    # _, forward_h, forward_c, backward_h, backward_c = encoder_lstm(masked_encoder_inputs)
    # state_c = Concatenate()([forward_c, backward_c])
    # state_h = Concatenate()([forward_h, backward_h])
    # encoder_states = [state_h, state_c]
    #
    # # Bottleneck Here
    #
    # decoder_inputs = Input(shape=(128, 3))
    # # masked_decoder_inputs = Masking()(decoder_inputs)
    # masked_decoder_inputs = decoder_inputs
    # decoder_lstm = LSTM(1000, return_state=True, return_sequences=True)
    # decoder_outputs, _, _ = decoder_lstm(masked_decoder_inputs, initial_state=encoder_states)
    #
    # outputs = TimeDistributed(Dense(2, activation=capped_relu))(decoder_outputs)
    #
    # model = Model([encoder_inputs, decoder_inputs], outputs)

    # SEQUENTIAL
    model = models.Sequential([
        Input((128, 3)),
        Bidirectional(LSTM(500, return_sequences=True)),
        Bidirectional(LSTM(500, return_sequences=True)),
        TimeDistributed(Dense(2, activation=capped_relu))
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss=losses.MeanAbsoluteError(),
                  metrics=["accuracy"])

    print(model.summary())

    return model


def learning_rate_scheduler(epoch, lr):
    if epoch > 50:
        lr = 0.00001
    return lr


def train_model(model: models.Sequential, train_x: np.ndarray, train_y: np.ndarray):
    print("Training Model")
    checkpoint = ModelCheckpoint("models/auto_save.h5", monitor='accuracy', verbose=1,
                                 save_best_only=True, mode='auto', period=2)
    lr_scheduler = LearningRateScheduler(learning_rate_scheduler, verbose=1)

    history = model.fit([train_x, train_x], train_y, batch_size=160, epochs=50, verbose=1, validation_split=TEST_SPLIT,
                        shuffle=True, callbacks=[checkpoint, lr_scheduler])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(history.history['accuracy'], label='accuracy')
    ax1.plot(history.history['val_accuracy'], label='val_accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')

    ax2.plot(history.history['loss'], label='loss')
    ax2.plot(history.history['val_loss'], label='val_loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='lower right')
    plt.savefig("models/training_progress.png")
    plt.show()

    model.save(MODEL_PATH)


def test(model: models.Sequential, test_data: np.ndarray, ground_truth: np.ndarray):
    data_index = 2
    test_data = test_data[data_index:data_index + 1]
    result = model.predict([test_data, test_data])

    result = result[0] * 63
    print(result)
    np.savetxt("test.nosync/result.txt", result)
    result_image = create_image_from_data(result)

    test = test_data[0] * 63
    test_image = create_image_from_data(test)

    ground_truth = ground_truth[data_index] * 63
    ground_truth = create_image_from_data(ground_truth)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    colorbar1 = ax1.imshow(test_image)
    fig.colorbar(colorbar1, ax=ax1)
    colorbar2 = ax2.imshow(result_image)
    fig.colorbar(colorbar2, ax=ax2)
    colorbar3 = ax3.imshow(ground_truth)
    fig.colorbar(colorbar3, ax=ax3)

    plt.show()


def predict(model, image_data):
    normalize_x(image_data)
    predicted_path = model.predict([image_data, image_data])

    predicted_path = predicted_path[0] * 63

    return predicted_path


def load_model(model_path):
    return models.load_model(model_path, custom_objects={"capped_relu": capped_relu})


if __name__ == "__main__":
    main()
