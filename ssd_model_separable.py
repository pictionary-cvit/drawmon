
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate

from utils.layers import leaky_relu


def bn_acti_conv(x, filters, kernel_size=1, stride=1, padding='same', activation='relu'):
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    if kernel_size > 1:
        x = SeparableConv2D(filters, kernel_size, depth_multiplier=1, strides=stride, padding=padding)(x)
    else:
        x = Conv2D(filters, kernel_size, strides=stride, padding=padding)(x)
    return x

def dense_block(x, n, growth_rate, width=4, activation='relu'):
    input_shape = K.int_shape(x)
    c = input_shape[3]
    for i in range(n):
        x1 = x
        x2 = bn_acti_conv(x, growth_rate*width, 1, 1, activation=activation)
        x2 = bn_acti_conv(x2, growth_rate, 3, 1, activation=activation)
        x = concatenate([x1, x2], axis=3)
        c += growth_rate
    return x

def downsampling_block(x, filters, width, padding='same', activation='relu'):
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x1 = MaxPooling2D(pool_size=2, strides=2, padding=padding)(x)
    x2 = DepthwiseConv2D(3, depth_multiplier=1, strides=2, padding=padding)(x)
    x = concatenate([x1, x2], axis=3)
    x = Conv2D(filters, 1, strides=1)(x)
    return x


def ssd512_dense_separable_body(x, activation='relu', num_dense_segs=3, use_prev_feature_map=True, num_multi_scale_maps=5):
    # used for SegLink and TextBoxes++ variantes with separable convolution
    
    if activation == 'leaky_relu':
        activation = leaky_relu
    
    growth_rate = 48
    compressed_features = 224
    source_layers = []
    
    x = SeparableConv2D(96, 3, depth_multiplier=32, strides=2, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = SeparableConv2D(96, 3, depth_multiplier=1, strides=1, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = SeparableConv2D(96, 3, depth_multiplier=1, strides=1, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    
    if (num_dense_segs >= 4):
        x = MaxPooling2D(pool_size=2, strides=2)(x)
        x = dense_block(x, 6, growth_rate, 4, activation)
        x = bn_acti_conv(x, compressed_features, 1, 1, activation=activation)
    
    if (num_dense_segs >= 3):
        x = MaxPooling2D(pool_size=2, strides=2)(x)
        x = dense_block(x, 6, growth_rate, 4, activation)
        x = bn_acti_conv(x, compressed_features, 1, 1, activation=activation)
    
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    x = dense_block(x, 6, growth_rate, 4, activation)
    x = bn_acti_conv(x, compressed_features, 1, 1, activation=activation)
    if (use_prev_feature_map): source_layers.append(x) # 64x64

    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = dense_block(x, 6, growth_rate, 4, activation)
    x = bn_acti_conv(x, compressed_features, 1, 1, activation=activation)
    source_layers.append(x) # 32x32
    
    if (num_multi_scale_maps >= 5):
        x = downsampling_block(x, 192, 1, activation=activation)
        source_layers.append(x) # 16x16
    
    if (num_multi_scale_maps >= 4):
        x = downsampling_block(x, 160, 1, activation=activation)
        source_layers.append(x) # 8x8
    
    
    x = downsampling_block(x, 128, 1, activation=activation)
    source_layers.append(x) # 4x4

    x = downsampling_block(x, 96, 1, activation=activation)
    source_layers.append(x) # 2x2
    
    x = downsampling_block(x, 64, 1, activation=activation)
    source_layers.append(x) # 1x1
    
    return source_layers

