import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, backend # type: ignore
from tensorflow.keras.layers import Dense, Activation, Input # type: ignore
from tensorflow.keras.optimizers import Nadam 

import numpy as np

@keras.utils.register_keras_serializable()
def swish_activation(x, b=1):
    return x*backend.sigmoid(b*x)

keras.utils.get_custom_objects().update({'swish_activation': Activation(swish_activation)})

@keras.utils.register_keras_serializable()
class CoefficientEstimator(Model):
    def __init__(self, layers=5, nodes_per_layer=100, input_dim=9, **kwargs):
        super().__init__(**kwargs)
        
        swish = Activation(swish_activation, name='Swish')

        if np.isscalar(nodes_per_layer) or len(nodes_per_layer) == 1:
            nodes_per_layer = np.array(nodes_per_layer).item() * np.ones(layers)
        elif len(np.array(nodes_per_layer).shape) != 0 and np.array(nodes_per_layer).shape[0] == layers:
            nodes_per_layer = np.array(nodes_per_layer)
        else:
            raise ValueError('nodes_per_layer has to be single valued or has to fulfill len(nodes_per_layer)=layers') 
                
        # input layer
        self.custom_layers = [Dense(nodes_per_layer[0], activation=swish, input_dim=input_dim, kernel_initializer='he_normal')]

        # hidden layers
        for i in range(1,layers):
            self.custom_layers.append(Dense(nodes_per_layer[i], activation=swish, kernel_initializer='he_normal'))

        # output layer
        self.custom_layers.append(Dense(1, activation='linear', kernel_initializer='he_normal'))

    def call(self, inputs):
        x = self.custom_layers[0](inputs)
        for i in range(1,len(self.custom_layers)):
            x = self.custom_layers[i](x)
        return x

def build_model(model, args):
    model = CoefficientEstimator(layers=args.layers, nodes_per_layer=args.nodes_per_layer, input_dim=8)

    optimizer = Nadam(
        learning_rate=args.learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'], weighted_metrics=['binary_accuracy'])
    
    return model