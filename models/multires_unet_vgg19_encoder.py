import tensorflow as tf
from .common_blocks import conv_block, multi_res_block, res_path, visual_attention_block


def multires_unet_with_vgg19_pretrained_input(input_shape, activation='relu'):
    model_input = tf.keras.layers.Input(shape=input_shape)
    # x = tf.keras.applications.mobilenet_v2.preprocess_input(model_input)
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1)(model_input)
    pretrained_model = tf.keras.applications.vgg19.VGG19(include_top=False,
                                                         input_shape=[
                                                             224, 224, 3],
                                                         input_tensor=x)
    pretrained_model.trainable = False

    skip_connections = [pretrained_model.get_layer('block1_conv2').output,
                        pretrained_model.get_layer('block2_conv2').output,
                        pretrained_model.get_layer('block3_conv4').output,
                        pretrained_model.get_layer('block4_conv4').output]

    connection_layer = pretrained_model.get_layer('block4_conv4').output
    filters = connection_layer.shape[3]
    x = tf.keras.layers.AvgPool2D(2)(connection_layer)
    # Bottleneck path
    # filters *= 2
    x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Decoder path
    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, filters, activation=None)
    encoder_input = res_path(skip_connections[3], filters, 1)
    attn_output = visual_attention_block(encoder_input, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, filters, activation=None)
    encoder_input = res_path(skip_connections[2], filters, 2)
    attn_output = visual_attention_block(encoder_input, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, filters, activation=None)
    encoder_input = res_path(skip_connections[1], filters, 3)
    attn_output = visual_attention_block(encoder_input, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    x = tf.keras.layers.Dropout(0.1)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, filters, activation=None)
    encoder_input = res_path(skip_connections[0], filters, 4)
    attn_output = visual_attention_block(encoder_input, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    x = tf.keras.layers.Dropout(0.1)(x)

    output = tf.keras.layers.Conv2D(
        filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model, pretrained_model
