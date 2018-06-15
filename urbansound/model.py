import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Sequence

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from keras.layers import Dense, LSTM, BatchNormalization
from keras.models import Sequential, Model

from preprocessing import UrbanSoundData

NUM_CLASSES = 10


class NotTrainedError(Exception):
    pass


class BaseModel:
    def __init__(self, data: UrbanSoundData, hidden_layer_sizes: Sequence[int],
                 dropout_probabilities: Optional[Sequence[Optional[int]]] = None,
                 use_batch_norm: bool = True, model_name: str = None):
        if dropout_probabilities is not None and \
                len(hidden_layer_sizes) != len(dropout_probabilities):
            raise ValueError("Length of hidden_layer_sizes and "
                             "dropout_probabilities need to be the same")

        self.data = data
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_probabilities = dropout_probabilities if dropout_probabilities else \
            [None] * len(hidden_layer_sizes)
        self.use_batch_norm = use_batch_norm
        self.model: Sequential = None
        self.history = None
        now = datetime.now().isoformat('_', 'seconds')
        self.name = model_name if model_name else \
            f"{self.__class__.__name__}_{now}".replace(":", "-")

    def train(self, batch_size: Optional[int] = 32, epochs: Optional[int] = 10,
              verbose: int = 0):

        train_features, val_features, train_labels, val_labels = self.data.train_data

        train_features = self._process_features(train_features)
        val_features = self._process_features(val_features)

        input_shape = train_features.shape[1:]
        self.model = self._model(input_shape)

        save_dir = Path("..", "model", f"{self.name}")

        if save_dir.exists():
            raise ValueError(f"Model with name {self.name} exists already")
        else:
            save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / "weights.epoch{epoch:02d}-loss{val_categorical_accuracy:.2f}.hdf5"
        save_callback = ModelCheckpoint(str(save_path), "val_categorical_accuracy", save_best_only=True)
        early_stop_callback = EarlyStopping(patience=10, verbose=1)

        with open(save_dir / "model_structure.json", "w") as model_struc_file:
            json.dump(self.model.to_json(), model_struc_file)

        try:
            self.history = self.model.fit(train_features, train_labels,
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          validation_data=(val_features, val_labels),
                                          callbacks=[early_stop_callback, save_callback],
                                          verbose=verbose)
        except KeyboardInterrupt:
            print()
            print("Early stopping (by user)")

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
    def __init__(self, data: UrbanSoundData,
                 hidden_layer_sizes: Sequence[int],
                 dropout_probabilities: Optional[Sequence[Optional[float]]] = None,
                 use_batch_norm: bool = True):
        super().__init__(data, hidden_layer_sizes,
                         dropout_probabilities, use_batch_norm)

    def _model(self, input_shape: Tuple[int]) -> Model:
        model = Sequential()

        for layer_idx, (layer_size, dropout_proba) in enumerate(
                zip(self.hidden_layer_sizes, self.dropout_probabilities)):
            if layer_idx == 0:
                model.add(Dense(layer_size, input_shape=input_shape, activation=None))
            else:
                model.add(Dense(layer_size, activation=None))

            if self.use_batch_norm:
                model.add(BatchNormalization())

            model.add(Activation("relu"))

            if dropout_proba:
                model.add(Dropout(dropout_proba))

        model.add(Dense(NUM_CLASSES, activation="softmax"))
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["categorical_accuracy"])
        return model

    def _process_features(self, features: np.ndarray) -> np.ndarray:
        """Reshape 2D data into a 1D vector."""
        return features.reshape(features.shape[0], -1)


class CNNModel(BaseModel):
    """Simplified Interface to Keras CNN Models
    The last layer will always be fully connected, all other layers will include a Conv2D layer
    with tanh activation and a MaxPooling2D layer."""

    def __init__(self, data: UrbanSoundData,
                 hidden_layer_sizes: Sequence[int],
                 dropout_probabilities: Optional[Sequence[Optional[float]]] = None,
                 use_batch_norm: bool = True):
        if len(hidden_layer_sizes) < 2:
            raise ValueError("CNNModel needs at least two hidden layers (last layer is fully connected)")
        super().__init__(data, hidden_layer_sizes,
                         dropout_probabilities, use_batch_norm)

    def _model(self, input_shape: Tuple[int]) -> Model:
        model = Sequential()

        for layer_idx, (layer_size, dropout_proba) in enumerate(
                zip(self.hidden_layer_sizes[:-1], self.dropout_probabilities[:-1])):
            if layer_idx == 0:
                model.add(Conv2D(layer_size, (3, 3), padding="same",
                                 input_shape=input_shape, activation=None))
            else:
                model.add(Conv2D(layer_size, (3, 3), padding="same",
                                 activation=None))

            if self.use_batch_norm:
                model.add(BatchNormalization())

            model.add(Activation("relu"))

            if dropout_proba:
                model.add(Dropout(dropout_proba))

            model.add(MaxPooling2D())

        model.add(Flatten())
        model.add(Dense(self.hidden_layer_sizes[-1], use_bias=False, activation=None))

        if self.use_batch_norm:
            model.add(BatchNormalization())

        model.add(Activation("relu"))

        if self.dropout_probabilities[-1]:
            model.add(Dropout(self.dropout_probabilities[-1]))

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
    def __init__(self, data: UrbanSoundData,
                 hidden_layer_sizes: Sequence[int],
                 dropout_probabilities: Optional[Sequence[Optional[float]]] = None):
        # LSTMs do not work well with batch norm
        super().__init__(data, hidden_layer_sizes,
                         dropout_probabilities, use_batch_norm=False)

    def _model(self, input_shape: Tuple[int]) -> Model:
        model = Sequential()

        for layer_idx, (layer_size, dropout_proba) in enumerate(
                zip(self.hidden_layer_sizes[:-1], self.dropout_probabilities[:-1])):
            if layer_idx == 0:
                model.add(LSTM(layer_size, return_sequences=True,
                               input_shape=input_shape))
            else:
                model.add(LSTM(layer_size, return_sequences=True))

            if dropout_proba:
                model.add(Dropout(dropout_proba))
        if len(self.hidden_layer_sizes) == 1:
            model.add(LSTM(self.hidden_layer_sizes[-1], input_shape=input_shape))
        else:
            model.add(LSTM(self.hidden_layer_sizes[-1]))
        if self.dropout_probabilities[-1]:
            model.add(Dropout(self.dropout_probabilities[-1]))

        model.add(Dense(NUM_CLASSES, activation="softmax"))
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["categorical_accuracy"])
        return model

    def _process_features(self, features: np.ndarray) -> np.ndarray:
        """For LSTMs, the input should be of shape (batch_size, time steps, input_dim).
        That means we need to swap the 2nd and 3rd axis.
        """
        return np.swapaxes(features, 1, 2)
