import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Input # type: ignore
from tensorflow.keras import Model, backend # type: ignore

@keras.utils.register_keras_serializable()
def swish_activation(x, b=1):
    return x*backend.sigmoid(b*x)

keras.utils.get_custom_objects().update({'swish_activation': Activation(swish_activation)})

@keras.utils.register_keras_serializable()
class C6_4l_clf_maxi(Model):
    def __init__(self, **kwargs):
        super(C6_4l_clf_maxi, self).__init__(**kwargs)
        
        swish = Activation(swish_activation, name='Swish')

        self.dense1 = Dense(2000, activation=swish, input_dim=9, kernel_initializer='he_normal')
        self.dense2 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.dense3 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.dense4 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.dense5 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.dense6 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.dense7 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.dense8 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.dense9 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.dense10 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.out = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        x = self.dense8(x)
        x = self.dense9(x)
        x = self.dense10(x)
        return self.out(x)
    
@keras.utils.register_keras_serializable()
class C6_4l_clf_maxi_nonprm(Model):
    def __init__(self, **kwargs):
        super(C6_4l_clf_maxi_nonprm, self).__init__(**kwargs)
        
        swish = Activation(swish_activation, name='Swish')

        self.dense1 = Dense(2000, activation=swish, input_dim=8, kernel_initializer='he_normal')
        self.dense2 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.dense3 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.dense4 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.dense5 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.dense6 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.dense7 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.dense8 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.dense9 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.dense10 = Dense(2000, activation=swish, kernel_initializer='he_normal')
        self.out = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        x = self.dense8(x)
        x = self.dense9(x)
        x = self.dense10(x)
        return self.out(x)
    
@keras.utils.register_keras_serializable()
class C6_4l_clf_big_nonprm(Model):
    def __init__(self, **kwargs):
        super(C6_4l_clf_big_nonprm, self).__init__(**kwargs)
        
        swish = Activation(swish_activation, name='Swish')

        self.dense1 = Dense(2500, activation=swish, input_dim=8, kernel_initializer='he_normal')
        self.dense2 = Dense(2500, activation=swish, kernel_initializer='he_normal')
        self.dense3 = Dense(2500, activation=swish, kernel_initializer='he_normal')
        self.dense4 = Dense(2500, activation=swish, kernel_initializer='he_normal')
        self.dense5 = Dense(2500, activation=swish, kernel_initializer='he_normal')
        self.dense6 = Dense(2500, activation=swish, kernel_initializer='he_normal')
        self.dense7 = Dense(2500, activation=swish, kernel_initializer='he_normal')
        self.dense8 = Dense(2500, activation=swish, kernel_initializer='he_normal')
        self.dense9 = Dense(2500, activation=swish, kernel_initializer='he_normal')
        self.dense10 = Dense(2500, activation=swish, kernel_initializer='he_normal')
        self.out = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        x = self.dense8(x)
        x = self.dense9(x)
        x = self.dense10(x)
        return self.out(x)

@keras.utils.register_keras_serializable()
class C6_4l_clf(Model):
    def __init__(self, **kwargs):
        super(C6_4l_clf, self).__init__(**kwargs)
        
        swish = Activation(swish_activation, name='Swish')

        self.dense1 = Dense(1000, activation=swish, input_dim=9, kernel_initializer='he_normal')
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
    

@keras.utils.register_keras_serializable()
class C6_4l_clf_reduced(Model):
    def __init__(self, **kwargs):
        super(C6_4l_clf_reduced, self).__init__(**kwargs)
        
        swish = Activation(swish_activation, name='Swish')

        self.dense1 = Dense(100, activation=swish, input_dim=9, kernel_initializer='he_normal')
        self.dense2 = Dense(100, activation=swish, kernel_initializer='he_normal')
        self.dense3 = Dense(100, activation=swish, kernel_initializer='he_normal')
        self.dense4 = Dense(100, activation=swish, kernel_initializer='he_normal')
        self.dense5 = Dense(100, activation=swish, kernel_initializer='he_normal')
        self.out = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return self.out(x)
    

@keras.utils.register_keras_serializable()
class C6_4l_clf_reduced_nonprm(Model):
    def __init__(self, **kwargs):
        super(C6_4l_clf_reduced_nonprm, self).__init__(**kwargs)
        
        swish = Activation(swish_activation, name='Swish')

        self.dense1 = Dense(100, activation=swish, input_dim=8, kernel_initializer='he_normal')
        self.dense2 = Dense(100, activation=swish, kernel_initializer='he_normal')
        self.dense3 = Dense(100, activation=swish, kernel_initializer='he_normal')
        self.dense4 = Dense(100, activation=swish, kernel_initializer='he_normal')
        self.dense5 = Dense(100, activation=swish, kernel_initializer='he_normal')
        self.out = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return self.out(x)
    
@keras.utils.register_keras_serializable()
class C6_4l_clf_mini_nonprm(Model):
    def __init__(self, **kwargs):
        super(C6_4l_clf_mini_nonprm, self).__init__(**kwargs)
        
        swish = Activation(swish_activation, name='Swish')

        self.dense1 = Dense(10, activation=swish, input_dim=1, kernel_initializer='he_normal')
        self.dense2 = Dense(10, activation=swish, kernel_initializer='he_normal')
        self.dense3 = Dense(10, activation=swish, kernel_initializer='he_normal')
        self.dense4 = Dense(10, activation=swish, kernel_initializer='he_normal')
        self.dense5 = Dense(10, activation=swish, kernel_initializer='he_normal')
        self.out = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return self.out(x)

@keras.utils.register_keras_serializable()
class C6_4l_clf_mini(Model):
    def __init__(self, **kwargs):
        super(C6_4l_clf_mini, self).__init__(**kwargs)
        
        swish = Activation(swish_activation, name='Swish')

        self.dense1 = Dense(10, activation=swish, input_dim=2, kernel_initializer='he_normal')
        self.dense2 = Dense(10, activation=swish, kernel_initializer='he_normal')
        self.dense3 = Dense(10, activation=swish, kernel_initializer='he_normal')
        self.dense4 = Dense(10, activation=swish, kernel_initializer='he_normal')
        self.dense5 = Dense(10, activation=swish, kernel_initializer='he_normal')
        self.out = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return self.out(x)
    
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