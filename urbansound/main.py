from model import MLPModel, CNNModel, LSTMModel
from preprocessing import UrbanSoundData, UrbanSoundExtractor
import logging
from keras import backend as K
import tensorflow as tf


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    K.set_session(tf.Session(config=config))


    logger = logging.getLogger()
    logger.setLevel("WARN")
    UrbanSoundExtractor().prepare_data()
    data = UrbanSoundData()

    #mlp = MlpModel(data)
    #mlp.train(epochs=100, verbose=0)
    #mlp.visualize_training()

    lstm = LSTMModel(data)
    lstm.train(batch_size=8, epochs=50, verbose=1)
    lstm.evaluate()
    lstm.visualize_training()
