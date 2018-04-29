from typing import Tuple, Callable, List, Optional
import logging

from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
import numpy as np

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

        self.model.evaluate(features, labels)

    def predict(self):
        if self.model is None:
            raise NotTrainedError("Model needs to be trained before prediction")
        raise NotImplementedError()

    def visualize_training(self):
        import matplotlib.pyplot as plt
        #  accuracy history
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
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


class MlpModel(BaseModel):
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
                      metrics=["accuracy"])
        return model

    def _process_features(self, features: np.ndarray) -> np.ndarray:
        """Reshape 2D data into a 1D vector."""
        return features.reshape(features.shape[0], -1)


class CnnModel(BaseModel):
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
                      metrics=["accuracy"])
        return model

    def _process_features(self, features: np.ndarray) -> np.ndarray:
        """Reshape 2D data into a 3D matrix with shape[-1] == 1."""
        return features.reshape((*features.shape, 1))

