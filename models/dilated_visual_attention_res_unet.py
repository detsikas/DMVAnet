import tensorflow as tf
from .common_blocks import conv_block, residual_block, dilated_residual_block, visual_attention_block


def dilated_visual_attention_residual_unet(input_shape, starting_filters=16, activation='relu'):
    # Encoder path
    filters = starting_filters
    model_input = tf.keras.layers.Input(shape=input_shape)
    x = conv_block(model_input, filters, activation=activation)
    sc_1 = tf.keras.layers.Dropout(0.1)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_1)
    x = residual_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    sc_2 = tf.keras.layers.Dropout(0.1)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_2)
    x = residual_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    sc_3 = tf.keras.layers.Dropout(0.2)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_3)
    x = dilated_residual_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    sc_4 = tf.keras.layers.Dropout(0.2)(x)

    filters *= 2
    x = dilated_residual_block(x, filters, 2)
    # x = conv_block(x, filters)
    sc_5 = tf.keras.layers.Dropout(0.2)(x)

    # Bottleneck path
    filters *= 2
    x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Decoder path
    filters //= 2
    x = residual_block(x, filters)
    attn_output = visual_attention_block(sc_5, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    # x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = residual_block(x, filters)
    attn_output = visual_attention_block(sc_4, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    # x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    attn_output = visual_attention_block(sc_3, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    # x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    attn_output = visual_attention_block(sc_2, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    # x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    attn_output = visual_attention_block(sc_1, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    # x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)

    # Output
    output = tf.keras.layers.Conv2D(
        filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model
