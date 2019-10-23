from .callbacks import F1score
from .sequence_encoder import NERSequence
from .__common__ import LOGGER_NAME
import logging
logger = logging.getLogger(LOGGER_NAME)

class Trainer(object):
    """
    --> A trainer to train the model.
    """
    def __init__(self,
                 model):
        """
        --> initialize trainer
        :param model: keras sequence model (untrained architecture, model to be trained)
        """
        self._model = model

    def train(self, x_train, y_train, x_valid=None, y_valid=None,
              epochs=1, batch_size=32, verbose=1, callbacks=None, shuffle=True):
        """
        --> Trains the model for a fixed number of epochs (iterations on a dataset).
        :param x_train: list of training data.
        :param y_train: list of training target (label) data.
        :param x_valid: list of validation data.
        :param y_valid: list of validation target (label) data.
        :param batch_size: Integer.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
        :param epochs: Integer. Number of epochs to train the model.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during training.
        :param shuffle: Boolean (whether to shuffle the training data
            before each epoch). `shuffle` will default to True.
        """

        train_seq = NERSequence(x_train, y_train, batch_size)

        if x_valid and y_valid:
            valid_seq = NERSequence(x_valid, y_valid, batch_size)
            f1 = F1score(valid_seq)
            callbacks = [f1] + callbacks if callbacks else [f1]

        self._model.fit_generator(generator=train_seq,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  verbose=verbose,
                                  shuffle=shuffle)