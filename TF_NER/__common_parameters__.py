## data type to be used
import tensorflow as tf
import numpy as np

DTYPE = {
    'float' :{
        'tf' : tf.float32,
        'np': np.float32
    },
    'int' : {
        'tf': tf.int32,
        'np': np.int32
    }
}

LOGGER_NAME = 'TF_NER_LOGGER'
