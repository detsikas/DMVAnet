import tensorflow as tf
from common_blocks import conv_block, residual_block, dilated_residual_block, aspp_with_image_level_features


def deeplabv3plus(input_shape, starting_filters=16, activation='relu'):
    filters = starting_filters
    # Encoder path
    model_input = tf.keras.layers.Input(shape=input_shape)
    x = conv_block(model_input, filters, activation=activation)
    # sc_1 = tf.keras.layers.Dropout(0.1)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(4)(x)
    x = residual_block(x, filters)
    sc_1 = x
    # x = conv_block(x, filters)
    # sc_2 = tf.keras.layers.Dropout(0.1)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(x)
    x = residual_block(x, filters)
    # x = conv_block(x, filters)
    # sc_3 = tf.keras.layers.Dropout(0.2)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(x)
    x = dilated_residual_block(x, filters, 2)
    # x = conv_block(x, filters)
    # sc_4 = tf.keras.layers.Dropout(0.2)(x)

    filters *= 2
    x = dilated_residual_block(x, filters, 2)
    # x = conv_block(x, filters)
    # sc_5 = tf.keras.layers.Dropout(0.2)(x)

    # x = aspp(x, filters)
    x = aspp_with_image_level_features(
        x, model_input, filters, activation=activation)

    # Decoder
    x1 = tf.keras.layers.UpSampling2D(size=4, interpolation='bilinear')(x)
    x2 = conv_block(sc_1, kernel_size=1, filters=filters,
                    activation=activation)
    x = tf.keras.layers.Concatenate()([x1, x2])
    x = conv_block(x, kernel_size=3, filters=filters, activation=activation)
    x = conv_block(x, kernel_size=3, filters=filters, activation=activation)
    x = tf.keras.layers.UpSampling2D(size=4, interpolation='bilinear')(x)
    output = tf.keras.layers.Conv2D(
        filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model
