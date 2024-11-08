import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation # type: ignore
from tensorflow.keras import Model, backend # type: ignore

@keras.utils.register_keras_serializable()
def swish_activation(x, b=1):
    return x*backend.sigmoid(b*x)

keras.utils.get_custom_objects().update({'swish_activation': Activation(swish_activation)})

@keras.utils.register_keras_serializable()
class C6_4l_clf(Model):
    def __init__(self, **kwargs):
        super(C6_4l_clf, self).__init__(**kwargs)
        
        swish = Activation(swish_activation, name='Swish')

        self.dense1 = Dense(1000, activation=swish, input_dim=8, kernel_initializer='he_normal')
        self.dense2 = Dense(1000, activation=swish, kernel_initializer='he_normal')
        self.dense3 = Dense(1000, activation=swish, kernel_initializer='he_normal')
        self.dense4 = Dense(1000, activation=swish, kernel_initializer='he_normal')
        self.dense5 = Dense(1000, activation=swish, kernel_initializer='he_normal')
        self.out = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return self.out(x)