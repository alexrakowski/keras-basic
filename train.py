import os

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from models import simple
from data import gen_data

CHECKPOINTS_DIR = './checkpoints'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def make_dirs():
    import os
    if not os.path.isdir(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)


def train():
    train_data_path = ''
    test_data_path = ''

    # Create the model
    input_shape = IMG_DIMS
    model = simple(no_classes=3, input_shape=input_shape)
    print(model.summary())

    # Train the model
    checkpoints_filepath = os.path.join(CHECKPOINTS_DIR,
                                        '{}_{}.hdf5'.format(model.name, datetime.datetime.now().strftime("%Y%m%d_%H%M")
                                                            ))
    callbacks = [TensorBoard(log_dir='./tensorboard', write_images=True),
                 ModelCheckpoint(filepath=checkpoints_filepath,
                                 verbose=1,
                                 save_best_only=True),
                 ]
    model.fit_generator(
        gen_data(dataset_path=train_data_path),
        epochs=50,
        steps_per_epoch=64,
        validation_data=gen_data(dataset_path=test_data_path),
        validation_steps=16,
        callbacks=callbacks,
    )
    # Evaluate the model
    scores = model.evaluate_generator(gen_data(dataset_path=test_data_path), steps=1000)
    print("Accuracy: {}%".format(scores[1] * 100))


if __name__ == '__main__':
    make_dirs()
    train()
