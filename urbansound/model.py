import logging
from typing import Tuple, Optional

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, Dropout, TimeDistributed
from keras.layers import Dense, LSTM, BatchNormalization
from keras.models import Sequential, Model

from preprocessing import UrbanSoundData

NUM_CLASSES = 10


class NotTrainedError(Exception):
    pass


class BaseModel:
    def __init__(self, data: UrbanSoundData):
        self.data = data
        self.model: Sequential = None
        self.history = None

    def train(self, batch_size: Optional[int] = 32, epochs: Optional[int] = 10,
              short_data: Optional[bool] = False, verbose: int=0):

        if short_data:
            train_features, test_features, train_labels, test_labels = self.data.train_data_short
        else:
            train_features, test_features, train_labels, test_labels = self.data.train_data_long

        train_features = self._process_features(train_features)
        test_features = self._process_features(test_features)

        input_shape = train_features.shape[1:]
        self.model = self._model(input_shape)

        early_stop_callback = EarlyStopping(patience=20, verbose=1)

        logging.info("Start training")
        self.history = self.model.fit(train_features, train_labels,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_data=(test_features, test_labels),
                                      callbacks=[early_stop_callback],
                                      verbose=verbose)
        return self.history

    def evaluate(self):
        if self.model is None:
            raise NotTrainedError("Model needs to be trained before evaluation")

        features, labels = self.data.test_data
        features = self._process_features(features)

        return self.model.evaluate(features, labels)

    def predict(self):
        if self.model is None:
            raise NotTrainedError("Model needs to be trained before prediction")
        raise NotImplementedError()

    def visualize_training(self):
        import matplotlib.pyplot as plt
        #  accuracy history
        plt.plot(self.history.history['categorical_accuracy'])
        plt.plot(self.history.history['val_categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # loss history
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def _process_features(self, features: np.ndarray) -> np.ndarray:
        """Model specific data preprocessing. May be overwritten by derived classes."""
        return features

    def _model(self, input_shape: Tuple[int]) -> Model:
        """Returns a compiled keras model."""
        raise NotImplementedError()


class MLPModel(BaseModel):
    def __init__(self, data: UrbanSoundData):
        super().__init__(data)

    def _model(self, input_shape: Tuple[int]) -> Model:
        model = Sequential()
        model.add(Dense(64, input_shape=input_shape, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(64, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(NUM_CLASSES, activation="softmax"))
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["categorical_accuracy"])
        return model

    def _process_features(self, features: np.ndarray) -> np.ndarray:
        """Reshape 2D data into a 1D vector."""
        return features.reshape(features.shape[0], -1)


class CNNModel(BaseModel):
    def __init__(self, data: UrbanSoundData):
        super().__init__(data)

    def _model(self, input_shape: Tuple[int]) -> Model:
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), padding="same",
                         input_shape=input_shape, activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())

        model.add(Conv2D(64, kernel_size=(3, 3), padding="same",
                         input_shape=input_shape, activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())

        model.add(Conv2D(64, kernel_size=(3, 3), padding="same",
                         input_shape=input_shape, activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())

        model.add(Flatten())
        model.add(Dense(512, use_bias=False, activation="relu"))
        model.add(BatchNormalization())

        model.add(Dense(NUM_CLASSES, activation="softmax"))
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["categorical_accuracy"])
        return model

    def _process_features(self, features: np.ndarray) -> np.ndarray:
        """Reshape 2D data into a 3D matrix with shape[-1] == 1.
        This last dimension is only 1 wide because we have only one
        channel (i.e. no color information)"""
        return features.reshape((*features.shape, 1))


class LSTMModel(BaseModel):
    def __init__(self, data: UrbanSoundData):
        super().__init__(data)

    def _model(self, input_shape: Tuple[int]) -> Model:
        model = Sequential()

        model.add(LSTM(512, input_shape=input_shape, return_sequences=True))

        model.add(LSTM(512, input_shape=input_shape))
        model.add(Dropout(0.25))

        model.add(Dense(NUM_CLASSES, activation="softmax"))
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["categorical_accuracy"])
        return model

    def _process_features(self, features: np.ndarray) -> np.ndarray:
        """For LSTMs, the input should be of shape (batch_size, time steps, input_dim).
        That means we need to swap the 2nd and 3rd axis.
        """
        print(np.swapaxes(features, 1, 2).shape)
        return np.swapaxes(features, 1, 2)

