import tensorflow as tf
from tensorflow.keras import layers, Model

def segnet(input_shape=(128, 128, 3), num_classes=1):
    inputs = layers.Input(input_shape)

    def encoder_block(x, filters):
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        p = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
        return x, p


    def decoder_block(x, skip, filters):
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Add()([x, skip])
        return x

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b = layers.Conv2D(512, 3, padding='same')(p4)
    b = layers.BatchNormalization()(b)
    b = layers.Activation('relu')(b)

    d4 = decoder_block(b, s4, 512)
    d3 = decoder_block(d4, s3, 256)
    d2 = decoder_block(d3, s2, 128)
    d1 = decoder_block(d2, s1, 64)

    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(d1)

    model = Model(inputs, outputs)
    return model

if __name__ == "__main__":
    model = segnet()
    model.summary()

