from preprocessing import UrbanSoundData

if __name__ == "__main__":
    from model import MLPModel, CNNModel, LSTMModel
    import tensorflow as tf
    from keras import backend as K

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    data = UrbanSoundData(n_mfccs=20, n_augmentations=1)

    # mlp = MLPModel(data, [512, 512], [0.7, 0.5)
    # mlp.train(batch_size=64, epochs=50, verbose=1)
    # print(mlp.evaluate())
    # mlp.visualize_training()
    # del mlp

    cnn = CNNModel(data, [32, 64, 256, 512], dropout_probabilities=None, use_batch_norm=True)
    cnn.train(batch_size=32, epochs=50, verbose=1)
    print("Metrics on held out test data:", cnn.evaluate())
    cnn.visualize_training()
    del cnn

    # lstm = LSTMModel(data, [512], [0.25])
    # lstm.train(batch_size=32, epochs=5, verbose=1)
    # print(lstm.evaluate())
    # lstm.visualize_training()
    # del lstm
