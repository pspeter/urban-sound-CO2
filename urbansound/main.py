from preprocessing import UrbanSoundData


def all_test_runs():
    layer_configs = [
        [16, 128],
        [16, 16, 16, 128],
        [16, 32, 64, 128],
    ]

    for layer_config in layer_configs:
        for dropout in [None, 0.5]:
            for batch_norm in [True, False]:
                run_test(data, CNNModel, layer_config, dropout, batch_norm)

    layer_configs = [
        [512],
        [32, 32, 32, 32, 32, 32],
        [32, 64, 128, 256],
        [256, 128, 64, 32],
    ]

    for layer_config in layer_configs:
        for dropout in [None, 0.5]:
            for batch_norm in [True, False]:
                run_test(data, MLPModel, layer_config, dropout, batch_norm)

    layer_configs = [
        [128],
        [32, 32, 32, 32],
        [64, 32],
        [32, 64],
    ]

    for layer_config in layer_configs:
        for dropout in [0, 0.5]:
                run_test(data, LSTMModel, layer_config, dropout, False)


def run_test(data, model_type, layer_config, dropout, batch_norm):
    print(model_type.__name__, layer_config, dropout, batch_norm)
    dropouts = [dropout for _ in range(len(layer_config))]

    for _ in range(5):
        model = model_type(data, layer_config, dropouts, batch_norm,
                           model_type.__name__, save_model=False)
        model.train(batch_size=32, epochs=20, verbose=0)
        model.evaluate(log_dir="../exploration/model_log.txt")


if __name__ == "__main__":
    import tensorflow as tf
    from keras import backend as K

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    from model import MLPModel, CNNModel, LSTMModel

    data = UrbanSoundData(n_mfccs=20, n_augmentations=0)

    # all_test_runs()

    cnn = CNNModel(data, [64, 128, 512], dropout_probabilities=None, use_batch_norm=True,
                   log_tensorboard=False, model_name="CNN", save_model=False, overwrite=True)
    cnn.train(batch_size=32, epochs=50, verbose=1)
    print("Evaluation:", cnn.evaluate())
    cnn.visualize_training()
    del cnn

