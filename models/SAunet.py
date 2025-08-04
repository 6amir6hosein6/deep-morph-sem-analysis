import tensorflow as tf
from tensorflow.keras import layers, Model

def spatial_attention_module(input_feature):
    kernel_size = 7
    avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    attention = layers.Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid')(concat)
    return input_feature * attention

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

def sa_unet(input_shape=(128,128,3), num_classes=1):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck
    c5 = conv_block(p4, 1024)

    # Decoder with spatial attention
    u6 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    sa6 = spatial_attention_module(c4)
    u6 = layers.concatenate([u6, sa6])
    c6 = conv_block(u6, 512)

    u7 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    sa7 = spatial_attention_module(c3)
    u7 = layers.concatenate([u7, sa7])
    c7 = conv_block(u7, 256)

    u8 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    sa8 = spatial_attention_module(c2)
    u8 = layers.concatenate([u8, sa8])
    c8 = conv_block(u8, 128)

    u9 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    sa9 = spatial_attention_module(c1)
    u9 = layers.concatenate([u9, sa9])
    c9 = conv_block(u9, 64)

    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(c9)

    model = Model(inputs, outputs)
    return model

if __name__ == "__main__":
    model = sa_unet()
    model.summary()

