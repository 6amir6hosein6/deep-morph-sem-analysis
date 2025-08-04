import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(inputs, filters, kernel_size=3, activation='relu', padding='same'):
    x = layers.Conv2D(filters, kernel_size, padding=padding)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x

def encoder_block(inputs, filters):
    x = conv_block(inputs, filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, filters):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, filters)
    return x

def build_unet(input_shape, base_filters=64):
    inputs = layers.Input(shape=input_shape)


    s1, p1 = encoder_block(inputs, base_filters)
    s2, p2 = encoder_block(p1, base_filters*2)
    s3, p3 = encoder_block(p2, base_filters*4)
    s4, p4 = encoder_block(p3, base_filters*8)


    b1 = conv_block(p4, base_filters*16)


    d1 = decoder_block(b1, s4, base_filters*8)
    d2 = decoder_block(d1, s3, base_filters*4)
    d3 = decoder_block(d2, s2, base_filters*2)
    d4 = decoder_block(d3, s1, base_filters)

    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(d4)

    return Model(inputs, outputs)

def build_mini_unet(input_shape, base_filters=32):

    inputs = layers.Input(shape=input_shape)


    s1, p1 = encoder_block(inputs, base_filters)
    s2, p2 = encoder_block(p1, base_filters*2)
    s3, p3 = encoder_block(p2, base_filters*4)


    b1 = conv_block(p3, base_filters*8)


    d1 = decoder_block(b1, s3, base_filters*4)
    d2 = decoder_block(d1, s2, base_filters*2)
    d3 = decoder_block(d2, s1, base_filters)

    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(d3)

    return Model(inputs, outputs)

def build_iternet(input_shape, num_mini_unets=3):

    inputs = layers.Input(shape=input_shape)


    main_unet = build_unet(input_shape)
    main_output = main_unet(inputs)

    x = main_output

    for i in range(num_mini_unets):

        mini_input = layers.Concatenate()([inputs, x])
        mini_unet = build_mini_unet(mini_input.shape[1:])
        x = mini_unet(mini_input)

    return Model(inputs, x)



if __name__ == "__main__":
    input_shape = (256, 256, 1)
    model = build_iternet(input_shape, num_mini_unets=3)
    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



