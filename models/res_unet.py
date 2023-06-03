import tensorflow as tf
from common_blocks import conv_block, residual_block


def residual_unet(input_shape, starting_filters=16, activation='relu'):
    filters = starting_filters
    # Encoder path
    model_input = tf.keras.layers.Input(shape=input_shape)
    x = conv_block(model_input, filters, activation=activation)
    sc_1 = tf.keras.layers.Dropout(0.1)(x)
    print(sc_1.shape)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_1)
    x = residual_block(x, filters)
    # x = conv_block(x, filters)
    sc_2 = tf.keras.layers.Dropout(0.1)(x)
    print(sc_2.shape)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_2)
    x = residual_block(x, filters)
    # x = conv_block(x, filters)
    sc_3 = tf.keras.layers.Dropout(0.2)(x)
    print(sc_3.shape)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_3)
    x = residual_block(x, filters)
    # x = conv_block(x, filters)
    sc_4 = tf.keras.layers.Dropout(0.2)(x)
    print(sc_4.shape)

    x = tf.keras.layers.AvgPool2D(2)(sc_4)

    # Bottleneck path
    filters *= 2
    x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Decoder path
    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, sc_4])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, sc_3])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, sc_2])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, sc_1])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)
    output = tf.keras.layers.Conv2D(
        filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model
