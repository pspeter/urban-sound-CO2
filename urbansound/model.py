import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Sequence

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, Dropout, SimpleRNN
from keras.layers import Dense, LSTM, BatchNormalization
from keras.models import Sequential, Model

from preprocessing import UrbanSoundData

NUM_CLASSES = 10


class NotTrainedError(Exception):
    pass


class TimerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.total_time = 0
        self.epoch_start = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.total_time += time.time() - self.epoch_start


class BaseModel:
    def __init__(self, data: UrbanSoundData, hidden_layer_sizes: Sequence[int],
                 dropout_probabilities: Optional[Sequence[Optional[int]]] = None,
                 use_batch_norm: bool = True, model_name: str = None,
                 log_tensorboard: bool = False, save_model: bool = True,
                 overwrite: bool = False):

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

        self.log_tensorboard = log_tensorboard
        self.overwrite = overwrite
        self.save_model = save_model

        self.train_seconds_per_sample = None

    def train(self, batch_size: Optional[int] = 32, epochs: Optional[int] = 10,
              verbose: int = 0):

        train_features, val_features, train_labels, val_labels = self.data.train_data

        train_features = self._process_features(train_features)
        val_features = self._process_features(val_features)

        input_shape = train_features.shape[1:]
        self.model = self._model(input_shape)

        save_dir = Path("..", "model", f"{self.name}")

        if self.save_model or self.log_tensorboard:
            if save_dir.exists():
                if not self.overwrite:
                    raise ValueError(f"Model with name {self.name} exists already. Set overwrite=True "
                                     f"on {self.__class__.__name__} to overwrite old model.")

                shutil.rmtree(save_dir)

            save_dir.mkdir(parents=True, exist_ok=False)

        save_path = save_dir / "weights.epoch{epoch:02d}-loss{val_categorical_accuracy:.2f}.hdf5"

        early_stop_callback = EarlyStopping(patience=10, verbose=1)
        timer_callback = TimerCallback()
        callbacks = [early_stop_callback, timer_callback]

        if self.save_model:
            with open(save_dir / "model_structure.json", "w") as model_struc_file:
                json.dump(self.model.to_json(), model_struc_file)

            save_callback = ModelCheckpoint(str(save_path), "val_categorical_accuracy", save_best_only=True)
            callbacks.append(save_callback)

        if self.log_tensorboard:
            tensorboard_callback = TensorBoard(str(save_dir / "logs"), write_grads=True,
                                               write_images=True, histogram_freq=1)
            callbacks.append(tensorboard_callback)

        self.history = self.model.fit(train_features, train_labels,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_data=(val_features, val_labels),
                                      callbacks=callbacks,
                                      verbose=verbose)

        num_epochs = len(self.history.history["loss"])
        self.train_seconds_per_sample = round(timer_callback.total_time /
                                              num_epochs /
                                              train_features.shape[0], 5)
        return self.history

    def evaluate(self, log_dir: str = None):
        if self.model is None:
            raise NotTrainedError("Model needs to be trained before evaluation")

        train_features, _, train_labels, _ = self.data.train_data
        test_features, test_labels = self.data.test_data

        train_features = self._process_features(train_features)
        test_features = self._process_features(test_features)

        metrics = {
            "test_acc": round(self.model.evaluate(test_features, test_labels, verbose=0)[1], 5),
            "train_acc": round(self.model.evaluate(train_features, train_labels, verbose=0)[1], 5)
        }

        if log_dir:
            log_dir = Path(log_dir)

            if not log_dir.exists():
                with open(log_dir, "a") as fp:
                    fp.write("name,train_acc,test_acc,layer_sizes,dropout_rates,"
                             "use_batch_norm,train_seconds_per_sample\n")

            with open(log_dir, "a") as fp:
                fp.write(f"{self.name},{metrics['train_acc']},{metrics['test_acc']}"
                         f",'{self.hidden_layer_sizes}','{self.dropout_probabilities}',"
                         f"{self.use_batch_norm},"
                         f"{self.train_seconds_per_sample}\n")

        return metrics

    def predict(self, features: np.ndarray):
        if self.model is None:
            raise NotTrainedError("Model needs to be trained before prediction")

        features = self._process_features(features)
        return self.data.inverse_transform(self.model.predict(features))

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
                 use_batch_norm: bool = True, model_name: str = None,
                 log_tensorboard: bool = False, save_model: bool = True,
                 overwrite: bool = False):
        super().__init__(data, hidden_layer_sizes,
                         dropout_probabilities, use_batch_norm,
                         model_name, log_tensorboard, save_model, overwrite)

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
        """Reshape 2D data into 1D vectors."""
        return features.reshape(features.shape[0], -1)


class CNNModel(BaseModel):
    """Simplified Interface to Keras CNN Models
    The last layer will always be fully connected, all other layers will include a Conv2D layer
    with tanh activation and a MaxPooling2D layer."""

    def __init__(self, data: UrbanSoundData,
                 hidden_layer_sizes: Sequence[int],
                 dropout_probabilities: Optional[Sequence[Optional[float]]] = None,
                 use_batch_norm: bool = True, model_name: str = None,
                 log_tensorboard: bool = False, save_model: bool = True,
                 overwrite: bool = False):
        if len(hidden_layer_sizes) < 2:
            raise ValueError("CNNModel needs at least two hidden layers (last layer is fully connected)")

        super().__init__(data, hidden_layer_sizes, dropout_probabilities, use_batch_norm,
                         model_name, log_tensorboard, save_model, overwrite)

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
                 dropout_probabilities: Optional[Sequence[Optional[float]]] = None,
                 use_batch_norm: bool = False, model_name: str = None,
                 log_tensorboard: bool = False, save_model: bool = True,
                 overwrite: bool = False):

        # LSTMs do not work well with batch norm
        if use_batch_norm:
            print("LSTMs do not work with batch_norm, setting use_batch_norm to False")

        super().__init__(data, hidden_layer_sizes,
                         dropout_probabilities, False, model_name,
                         log_tensorboard, save_model, overwrite)

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


class RNNModel(BaseModel):
    def __init__(self, data: UrbanSoundData,
                 hidden_layer_sizes: Sequence[int],
                 dropout_probabilities: Optional[Sequence[Optional[float]]] = None,
                 use_batch_norm: bool = False, model_name: str = None,
                 log_tensorboard: bool = False, save_model: bool = True,
                 overwrite: bool = False):

        # LSTMs do not work well with batch norm
        if use_batch_norm:
            print("RNNs do not work with batch_norm, setting use_batch_norm to False")

        super().__init__(data, hidden_layer_sizes,
                         dropout_probabilities, False, model_name,
                         log_tensorboard, save_model, overwrite)

    def _model(self, input_shape: Tuple[int]) -> Model:
        model = Sequential()

        for layer_idx, (layer_size, dropout_proba) in enumerate(
                zip(self.hidden_layer_sizes[:-1], self.dropout_probabilities[:-1])):
            if layer_idx == 0:
                model.add(SimpleRNN(layer_size, return_sequences=True,
                               input_shape=input_shape))
            else:
                model.add(SimpleRNN(layer_size, return_sequences=True))

            if dropout_proba:
                model.add(Dropout(dropout_proba))

        if len(self.hidden_layer_sizes) == 1:
            model.add(SimpleRNN(self.hidden_layer_sizes[-1], input_shape=input_shape))
        else:
            model.add(SimpleRNN(self.hidden_layer_sizes[-1]))

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
