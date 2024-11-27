import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Input # type: ignore
from tensorflow.keras import Model, backend # type: ignore

import numpy as np

@keras.utils.register_keras_serializable()
def swish_activation(x, b=1):
    return x*backend.sigmoid(b*x)

keras.utils.get_custom_objects().update({'swish_activation': Activation(swish_activation)})

@keras.utils.register_keras_serializable()
class C6_4l_clf(Model):
    def __init__(self, num_layers=10, num_nodes=2000, input_dim=9, **kwargs):
        super(C6_4l_clf, self).__init__(**kwargs)
        
        swish = Activation(swish_activation, name='Swish')

        if np.isscalar(num_nodes) or len(num_nodes) == 1:
            num_nodes = np.array(num_nodes).item() * np.ones(num_layers)
        elif len(np.array(num_nodes).shape) != 0 and np.array(num_nodes).shape[0] == num_layers:
            num_nodes = np.array(num_nodes)
        else:
            raise ValueError('num_nodes has to be single valued or has to fulfill len(num_nodes)=num_layers') 
                

        self.custom_layers = [Dense(num_nodes[0], activation=swish, input_dim=input_dim, kernel_initializer='he_normal')]

        for i in range(1,num_layers):
            self.custom_layers.append(Dense(num_nodes[i], activation=swish, kernel_initializer='he_normal'))

        self.custom_layers.append(Dense(1, activation='sigmoid'))

    def call(self, inputs):
        x = self.custom_layers[0](inputs)

        for i in range(1,len(self.custom_layers)):
            x = self.custom_layers[i](x)

        return x
    
    
@keras.utils.register_keras_serializable()
class ToyClassifierPrm(Model):
    def __init__(self, **kwargs):
        super(ToyClassifierPrm, self).__init__(**kwargs)

        swish = Activation(swish_activation, name='Swish')

        self.dense = Dense(10, activation=swish, input_dim=2, kernel_initializer='he_normal')
        self.dense1 = Dense(10, activation=swish, kernel_initializer='he_normal')
        self.out = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.dense1(x)
        return self.out(x)

@keras.utils.register_keras_serializable()
class ToyClassifier(Model):
    def __init__(self, **kwargs):
        super(ToyClassifier, self).__init__(**kwargs)

        swish = Activation(swish_activation, name='Swish')

        self.dense = Dense(2, activation=swish, input_dim=1, kernel_initializer='he_normal')
        self.out = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense(inputs)
        return self.out(x)