import tensorflow as tf
from keras import backend as K

from model import MLPModel, CNNModel, LSTMModel
from preprocessing import UrbanSoundData, UrbanSoundExtractor

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print(config.gpu_options.per_process_gpu_memory_fraction)
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    K.set_session(tf.Session(config=config))

    UrbanSoundExtractor().prepare_data()
    data = UrbanSoundData()

    # mlp = MLPModel(data, [512, 512])
    # mlp.train(batch_size=64, epochs=10, verbose=1)
    # print(mlp.evaluate())
    # mlp.visualize_training()
    # del mlp

    cnn = CNNModel(data, [64, 128], dropout_probabilities=[0.5, 0.25])
    cnn.train(batch_size=8, epochs=10, verbose=1)
    print(cnn.evaluate())
    cnn.visualize_training()
    del cnn

    lstm = LSTMModel(data, [64])
    lstm.train(batch_size=8, epochs=1, verbose=1)
    print(lstm.evaluate())
    lstm.visualize_training()
    del lstm
