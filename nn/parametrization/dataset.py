import numpy as np
import pandas as pd
import tensorflow as tf 

from physics.hstar import gghzz, hzz, c6
from physics.simulation import msq

def build_datasets(sample, order=1, *, buffer_size=100, batch_size=32):

    '''
    x | y_(order) | w
    '''

    term = msq.Component.SIG

    events = sample[term]
    kinematics, weights = events.kinematics, events.weights

    mod = c6.Modifier(sample, term, c6_values = [-5,0,5])
    coefficients = mod.coeffs[:, order]

    # Convert to TensorFlow tensors
    features = tf.convert_to_tensor(kinematics.to_numpy(), dtype=tf.float32)
    targets = tf.convert_to_tensor(coefficients, dtype=tf.float32)
    weights = tf.convert_to_tensor(weights.to_numpy(), dtype=tf.float32)

    # Combine into a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, targets, weights))

    # Shuffle and batch the dataset for training
    dataset = dataset.shuffle(buffer_size).batch(batch_size)

    return dataset


