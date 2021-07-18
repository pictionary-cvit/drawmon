import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow as tf


def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def leaky_relu(x):
    """Leaky Rectified Linear activation.
    
    # References
        - [Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)
    """
    #return K.relu(x, alpha=0.1, max_value=None)
    
    # requires less memory than keras implementation
    alpha = 0.1
    zero = _to_tensor(0., x.dtype.base_dtype)
    alpha = _to_tensor(alpha, x.dtype.base_dtype)
    x = alpha * tf.minimum(x, zero) + tf.maximum(x, zero)
    return x


@tf.custom_gradient
def custom_op(x):
    result = x*tf.math.tanh(tf.math.softplus(x))
    
    def custom_grad(dy):
        # v = 1. + tf.math.exp(x)
        # h = tf.math.log(v)
        # grad_gh = 1./(tf.math.square(tf.math.cosh(h)))

        grad_hx = tf.math.sigmoid(x)
       
        grad_gh = 1 - tf.math.square(tf.math.tanh(tf.math.softplus(x))) 
        grad_gx = grad_gh * grad_hx

        grad_f = tf.math.tanh(tf.math.softplus(x)) + x*grad_gx

        grad = dy * grad_f
        
        return grad   


    return result, custom_grad


class Mish(Layer):
    def __init__(self):
        super(Mish, self).__init__()


    def call(self, x):
        return custom_op(x)


class Normalize(Layer):
    """Normalization layer as described in ParseNet paper.
    # Arguments
        scale: Default feature scale.
    # Input shape
        4D tensor with shape: (samples, rows, cols, channels)
    # Output shape
        Same as input
    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf
    # TODO
        Add possibility to have one scale for all features.
    """
    def __init__(self, scale=20, **kwargs):
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name=self.name+'_gamma', 
                                     shape=(input_shape[-1],),
                                     initializer=initializers.Constant(self.scale), 
                                     trainable=True)
        super(Normalize, self).build(input_shape)
        
    def call(self, x, mask=None):
        return self.gamma * K.l2_normalize(x, axis=-1)
