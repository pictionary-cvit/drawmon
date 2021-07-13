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




def mish(x: TensorLike) -> tf.Tensor:
    r"""Mish: A Self Regularized Non-Monotonic Neural Activation Function.
    Computes mish activation:
    $$
    \mathrm{mish}(x) = x \cdot \tanh(\mathrm{softplus}(x)).
    $$
    See [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681).
    Usage:
    >>> x = tf.constant([1.0, 0.0, 1.0])
    >>> tfa.activations.mish(x)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.865098..., 0.       , 0.865098...], dtype=float32)>
    Args:
        x: A `Tensor`. Must be one of the following types:
            `bfloat16`, `float16`, `float32`, `float64`.
    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    x = tf.convert_to_tensor(x)
    return x * tf.math.tanh(tf.math.softplus(x))




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
