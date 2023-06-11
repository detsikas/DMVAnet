import tensorflow as tf
from .common_blocks import multi_res_block, res_path, dilated_multi_res_block, visual_attention_block


def dilated_multires_visual_attention(input_shape, starting_filters=16, with_dropout=False, activation='relu'):
    layer_filters = starting_filters
    # Encoder path
    model_input = tf.keras.layers.Input(shape=input_shape)
    x = multi_res_block(model_input, layer_filters, activation=activation)
    if with_dropout:
        x = tf.keras.layers.Dropout(0.1)(x)
    sc_1 = res_path(x, layer_filters, 4)

    layer_filters *= 2
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = multi_res_block(x, layer_filters, activation=activation)
    if with_dropout:
        x = tf.keras.layers.Dropout(0.1)(x)
    sc_2 = res_path(x, layer_filters, 3)

    layer_filters *= 2
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = multi_res_block(x, layer_filters, activation=activation)
    if with_dropout:
        x = tf.keras.layers.Dropout(0.2)(x)
    sc_3 = res_path(x, layer_filters, 2)

    layer_filters *= 2
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = dilated_multi_res_block(x, layer_filters, activation=activation)
    if with_dropout:
        x = tf.keras.layers.Dropout(0.2)(x)
    sc_4 = res_path(x, layer_filters, 1)

    layer_filters *= 2
    x = dilated_multi_res_block(x, layer_filters, activation=activation)
    if with_dropout:
        x = tf.keras.layers.Dropout(0.2)(x)
    sc_5 = res_path(x, layer_filters, 1)

    # Bottleneck path
    layer_filters *= 2
    x = dilated_multi_res_block(x, layer_filters, activation=activation)
    if with_dropout:
        x = tf.keras.layers.Dropout(0.3)(x)

    # Decoder path
    layer_filters //= 2
    x = multi_res_block(x, layer_filters, activation=activation)
    attn_output = visual_attention_block(sc_5, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    if with_dropout:
        x = tf.keras.layers.Dropout(0.2)(x)

    layer_filters //= 2
    x = multi_res_block(x, layer_filters, activation=activation)
    attn_output = visual_attention_block(sc_4, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    if with_dropout:
        x = tf.keras.layers.Dropout(0.2)(x)

    layer_filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=layer_filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, layer_filters, activation=activation)
    attn_output = visual_attention_block(sc_3, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    if with_dropout:
        x = tf.keras.layers.Dropout(0.2)(x)

    layer_filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=layer_filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, layer_filters, activation=activation)
    attn_output = visual_attention_block(sc_2, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    if with_dropout:
        x = tf.keras.layers.Dropout(0.1)(x)

    layer_filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=layer_filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, layer_filters, activation=activation)
    attn_output = visual_attention_block(sc_1, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    if with_dropout:
        x = tf.keras.layers.Dropout(0.1)(x)

    # Output
    output = tf.keras.layers.Conv2D(
        filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model
