import tensorflow as tf
from .common_blocks import multi_res_block, res_path, visual_attention_block


def unet_pp(input_shape, with_dropout=False, number_of_levels=5, starting_filters=16):
    X = []
    model_input = tf.keras.layers.Input(shape=input_shape)
    x_enc = model_input
    filters = starting_filters
    # Unets
    for i in range(number_of_levels):

        # Encoder path
        if i > 0:
            x_enc = tf.keras.layers.MaxPool2D(2)(x_enc)

        x_enc = multi_res_block(x_enc, filters, activation=None)
        if with_dropout:
            x_enc = tf.keras.layers.Dropout(0.2)(x_enc)

        if i < (number_of_levels - 1):
            X.append([])
            sc = res_path(x_enc, filters, number_of_levels - 1 - i)
            X[i].append(sc)

        # Upsampling and skip connections
        x_dec = x_enc
        upsample_filters = filters
        for j in range(i):
            upsample_filters = upsample_filters // 2
            x_dec = tf.keras.layers.Conv2DTranspose(filters=upsample_filters, kernel_size=1, strides=2, padding='same')(
                x_dec)
            x_dec = multi_res_block(x_dec, upsample_filters, activation=None)
            # skip connections
            horizontal_level = i - j - 1
            for k in range(j + 1):
                # Multiple visual attention blocks
                attn_output = visual_attention_block(
                    X[horizontal_level][k], x_dec)
                x_dec = tf.keras.layers.Concatenate()([x_dec, attn_output])
                if with_dropout:
                    x_dec = tf.keras.layers.Dropout(0.2)(x_dec)
            x_dec = tf.keras.layers.Conv2D(
                filters=1, kernel_size=1, padding='same', activation='sigmoid')(x_dec)
            sc = res_path(x_dec, upsample_filters,
                          number_of_levels - horizontal_level - 1)
            X[horizontal_level].append(sc)
        filters *= 2

    # Output
    selected_segmentation_output = X[0][-1]
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(
        selected_segmentation_output)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model
