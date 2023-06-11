import tensorflow as tf
from .common_blocks import conv_block, residual_block


def residual_unet_with_mobilenet_v2_pretrained_input(input_shape, activation='relu'):
    model_input = tf.keras.layers.Input(shape=input_shape)
    # x = tf.keras.applications.mobilenet_v2.preprocess_input(model_input)
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1)(model_input)
    x = tf.keras.layers.Resizing(224, 224)(x)
    pretrained_MobileNetV2 = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                                            input_shape=[
                                                                                224, 224, 3],
                                                                            input_tensor=x)
    pretrained_MobileNetV2.trainable = False

    skip_connections = [pretrained_MobileNetV2.get_layer('block_1_expand_relu').output,
                        pretrained_MobileNetV2.get_layer(
                            'block_3_expand_relu').output,
                        pretrained_MobileNetV2.get_layer(
                            'block_6_expand_relu').output,
                        pretrained_MobileNetV2.get_layer('block_13_expand_relu').output]

    connection_layer = pretrained_MobileNetV2.get_layer(
        'block_13_expand_relu').output
    filters = connection_layer.shape[3]
    x = tf.keras.layers.AvgPool2D(2)(connection_layer)
    # Bottleneck path
    # filters *= 2
    x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Decoder path
    # filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, skip_connections[3]])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, skip_connections[2]])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, skip_connections[1]])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)

    filters //= 4
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, skip_connections[0]])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = tf.keras.layers.Resizing(256, 256)(x)
    output = tf.keras.layers.Conv2D(
        filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model, pretrained_MobileNetV2
