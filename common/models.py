import tensorflow as tf
import sys


def conv_block(inputs, filters, activation, kernel_size=3, dilation_rate=1):
    cx = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate)(
        inputs)
    cx = tf.keras.layers.BatchNormalization()(cx)
    if activation == 'relu':
        return tf.keras.layers.ReLU()(cx)
    elif activation == 'leaky_relu':
        return tf.keras.layers.LeakyReLU()(cx)
    elif activation is None:
        return cx
    else:
        print('Bad activation')
        sys.exit(0)


def residual_block(inputs, filters, activation='relu'):
    x = conv_block(inputs, filters, activation=activation)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same')(x)

    res = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same')(inputs)
    x = tf.keras.layers.Add()([x, res])
    x = tf.keras.layers.BatchNormalization()(x)
    if activation == 'leaky_relu':
        return tf.keras.layers.LeakyReLU()(x)
    elif activation == 'relu':
        return tf.keras.layers.ReLU()(x)
    else:
        print('Bad activation')
        sys.exit(0)


def dilated_residual_block(inputs, filters, dilation_rate=2, activation='relu'):
    x = conv_block(inputs, filters, activation=activation)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', dilation_rate=dilation_rate)(x)

    res = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', dilation_rate=dilation_rate)(inputs)
    x = tf.keras.layers.Add()([x, res])
    x = tf.keras.layers.BatchNormalization()(x)
    if activation == 'leaky_relu':
        return tf.keras.layers.LeakyReLU()(x)
    elif activation == 'relu':
        return tf.keras.layers.ReLU()(x)
    else:
        print('Bad activation')
        sys.exit(0)


def dense_block(inputs, filters):
    x1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same')(inputs)
    x2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same')(x1)
    x = tf.keras.layers.Concatenate()([inputs, x1, x2])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same')(x)


def MLP(inputs, C, r=8):
    mx = tf.keras.layers.Dense(C / r, activation='relu')(inputs)
    mx = tf.keras.layers.Dense(C, activation='relu')(mx)
    return mx


def visual_attention_block(enc_input, dec_input, r=8):
    C = enc_input.shape[3]

    # Channel attention for decoder input
    f_ch_avg = tf.keras.layers.GlobalAvgPool2D()(dec_input)
    m_ch = MLP(f_ch_avg, C, r)
    # f_ch = tf.keras.layers.RepeatVector(dimension*dimension)(f_ch)
    # f_ch = tf.keras.layers.Reshape((dimension,dimension,C))(f_ch)

    # Spatial attention for decoder input
    f_sp_avg = tf.reduce_mean(dec_input, 3, keepdims=True)
    m_sp = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid')(f_sp_avg)

    f_ch = tf.keras.layers.Multiply()([m_ch, enc_input])
    f_sp = tf.keras.layers.Multiply()([f_ch, m_sp])

    return f_sp


def local_refinement_module(b, e, m=8):
    # b_flat = tf.keras.layers.Reshape((b.shape[1] * b.shape[2], b.shape[3]))(b)
    be = tf.linalg.matmul(b, e, transpose_a=True)
    be = tf.keras.layers.Softmax(axis=(1, 2))(be)
    f = tf.linalg.matmul(b, be)
    m = tf.keras.layers.Activation('sigmoid')(m)
    g = tf.keras.layers.Multiply()([f, m])
    return g


def semantic_aggergation_block(x, c, n):
    x_feat = tf.keras.layers.Conv2D(filters=c, kernel_size=1)(x)
    x_feat = tf.keras.layers.Reshape((x_feat.shape[1] * x_feat.shape[2], x_feat.shape[3]))(x_feat)
    x_am = tf.keras.layers.Conv2D(filters=n, kernel_size=1)(x)
    x_am = tf.keras.layers.Softmax(axis=(1, 2))(x_am)
    x_am = tf.keras.layers.Reshape((x_am.shape[1] * x_am.shape[2], x_am.shape[3]))(x_am)
    return tf.linalg.matmul(x_feat, x_am, transpose_a=True)


def semantic_distribution_block(A, D, N, filters, activation):
    x = tf.keras.layers.Conv2D(filters=N, kernel_size=1)(A)
    x = tf.keras.layers.Softmax(axis=3)(x)
    x = tf.keras.layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x)
    m = tf.linalg.matmul(x, D, transpose_b=True)
    m = tf.keras.layers.Reshape((A.shape[1], A.shape[2], A.shape[3]))(m)
    e = tf.keras.layers.Add()([A, m])
    e = conv_block(e, filters=filters, activation=activation)
    return e, m


def context_fusion_block(b, a, d, filters, n, activation):
    e, m = semantic_distribution_block(a, d, n, filters, activation=activation)
    g = local_refinement_module(b, e, m)
    o = tf.keras.layers.Add()([g, e])
    o = conv_block(o, filters, activation=activation)
    return o


def multi_res_block(inputs, filters, activation, alpha=1.67):
    W = alpha * filters
    shortcut = conv_block(inputs, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), kernel_size=1, activation=None)
    conv3x3 = conv_block(inputs, int(W * 0.167), activation=None)
    conv5x5 = conv_block(conv3x3, int(W * 0.333), activation=None)
    conv7x7 = conv_block(conv5x5, int(W * 0.5), activation=None)

    if activation is None:
        mresx = tf.keras.layers.Concatenate()([conv3x3, conv5x5, conv7x7])
        mresx = tf.keras.layers.Add()([mresx, shortcut])
        return tf.keras.layers.BatchNormalization()(mresx)

    mresx = tf.keras.layers.Concatenate()([conv3x3, conv5x5, conv7x7])
    mresx = tf.keras.layers.BatchNormalization()(mresx)
    mresx = tf.keras.layers.Add()([mresx, shortcut])
    if activation == 'leaky_relu':
        mresx = tf.keras.layers.LeakyReLU()(mresx)
    elif activation == 'relu':
        mresx = tf.keras.layers.ReLU()(mresx)
    else:
        print('Bad extivation')
        sys.exit(0)
    return tf.keras.layers.BatchNormalization()(mresx)


def dilated_multi_res_block(inputs, filters, activation, alpha=1.67, dilation_rate=2):
    W = alpha * filters
    shortcut = conv_block(inputs, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), kernel_size=1,
                          dilation_rate=dilation_rate, activation=None)
    conv3x3 = conv_block(inputs, int(W * 0.167), dilation_rate=dilation_rate, activation=None)
    conv5x5 = conv_block(conv3x3, int(W * 0.333), dilation_rate=dilation_rate, activation=None)
    conv7x7 = conv_block(conv5x5, int(W * 0.5), dilation_rate=dilation_rate, activation=None)

    if activation is None:
        mresx = tf.keras.layers.Concatenate()([conv3x3, conv5x5, conv7x7])
        mresx = tf.keras.layers.Add()([mresx, shortcut])
        return tf.keras.layers.BatchNormalization()(mresx)

    mresx = tf.keras.layers.Concatenate()([conv3x3, conv5x5, conv7x7])
    mresx = tf.keras.layers.BatchNormalization()(mresx)
    mresx = tf.keras.layers.Add()([mresx, shortcut])
    if activation == 'leaky_relu':
        mresx = tf.keras.layers.LeakyReLU()(mresx)
    elif activation == 'relu':
        mresx = tf.keras.layers.ReLU()(mresx)
    else:
        print('Bad extivation')
        sys.exit(0)
    return tf.keras.layers.BatchNormalization()(mresx)


def res_path(inputs, filters, length, activation='relu'):
    shortcut = conv_block(inputs, filters, kernel_size=1, activation=None)
    rx = conv_block(inputs, filters, activation=activation)
    rx = tf.keras.layers.Add()([shortcut, rx])
    if activation == 'leaky_relu':
        rx = tf.keras.layers.LeakyReLU()(rx)
    elif activation == 'relu':
        rx = tf.keras.layers.ReLU()(rx)
    else:
        print('Bad extivation')
        sys.exit(0)

    rx = tf.keras.layers.BatchNormalization()(rx)

    for i in range(length - 1):
        shortcut = conv_block(rx, filters, kernel_size=1, activation=None)
        rx = conv_block(rx, filters, activation=activation)
        rx = tf.keras.layers.Add()([shortcut, rx])
        if activation == 'leaky_relu':
            rx = tf.keras.layers.LeakyReLU()(rx)
        elif activation == 'relu':
            rx = tf.keras.layers.ReLU()(rx)
        else:
            print('Bad extivation')
            sys.exit(0)
        rx = tf.keras.layers.BatchNormalization()(rx)

    return rx


def base_unet(input_shape, starting_filters=16, activation='relu'):
    filters = starting_filters
    # Encoder path
    model_input = tf.keras.layers.Input(shape=input_shape)
    x = conv_block(model_input, filters, activation=activation)
    sc_1 = tf.keras.layers.Dropout(0.1)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_1)
    x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    sc_2 = tf.keras.layers.Dropout(0.1)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_2)
    x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    sc_3 = tf.keras.layers.Dropout(0.2)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_3)
    x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    sc_4 = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.AvgPool2D(2)(sc_4)

    # Bottleneck path
    filters *= 2
    x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Decoder path
    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, sc_4])
    x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, sc_3])
    x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, sc_2])
    x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, sc_1])
    x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)

    # Output
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)

    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model


def dense_unet(input_shape, starting_filters=16, activation='relu'):
    filters = starting_filters
    # Encoder path
    model_input = tf.keras.layers.Input(shape=input_shape)
    x = conv_block(model_input, filters, activation=activation)
    sc_1 = tf.keras.layers.Dropout(0.1)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_1)
    x = dense_block(x, filters)
    sc_2 = tf.keras.layers.Dropout(0.1)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_2)
    x = dense_block(x, filters)
    sc_3 = tf.keras.layers.Dropout(0.2)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_3)
    x = dense_block(x, filters)
    sc_4 = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.AvgPool2D(2)(sc_4)

    # Bottleneck path
    filters *= 2
    x = dense_block(x, filters)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Decoder path
    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, sc_4])
    x = dense_block(x, filters)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, sc_3])
    x = dense_block(x, filters)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, sc_2])
    x = dense_block(x, filters)
    x = tf.keras.layers.Dropout(0.1)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, sc_1])
    x = dense_block(x, filters)
    x = tf.keras.layers.Dropout(0.1)(x)

    # Output
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)

    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model


def residual_unet_with_mobilenet_v2_pretrained_input(input_shape, activation='relu'):
    model_input = tf.keras.layers.Input(shape=input_shape)
    # x = tf.keras.applications.mobilenet_v2.preprocess_input(model_input)
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1)(model_input)
    x = tf.keras.layers.Resizing(224, 224)(x)
    pretrained_MobileNetV2 = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                                            input_shape=[224, 224, 3],
                                                                            input_tensor=x)
    pretrained_MobileNetV2.trainable = False

    skip_connections = [pretrained_MobileNetV2.get_layer('block_1_expand_relu').output,
                        pretrained_MobileNetV2.get_layer('block_3_expand_relu').output,
                        pretrained_MobileNetV2.get_layer('block_6_expand_relu').output,
                        pretrained_MobileNetV2.get_layer('block_13_expand_relu').output]

    connection_layer = pretrained_MobileNetV2.get_layer('block_13_expand_relu').output
    filters = connection_layer.shape[3]
    x = tf.keras.layers.AvgPool2D(2)(connection_layer)
    # Bottleneck path
    #filters *= 2
    x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Decoder path
    #filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, skip_connections[3]])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, skip_connections[2]])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, skip_connections[1]])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)

    filters //= 4
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, skip_connections[0]])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = tf.keras.layers.Resizing(256,256)(x)
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model, pretrained_MobileNetV2


def residual_unet_with_vgg19_pretrained_input(input_shape, activation='relu'):
    model_input = tf.keras.layers.Input(shape=input_shape)
    # x = tf.keras.applications.mobilenet_v2.preprocess_input(model_input)
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1)(model_input)
    pretrained_model = tf.keras.applications.vgg19.VGG19(include_top=False,
                                                                            input_shape=[224, 224, 3],
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
    #filters *= 2
    x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Decoder path
    #filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, skip_connections[3]])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, skip_connections[2]])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, skip_connections[1]])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)

    filters //= 4
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, skip_connections[0]])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)

    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model, pretrained_model


def multires_unet_with_vgg19_pretrained_input(input_shape, activation='relu'):
    model_input = tf.keras.layers.Input(shape=input_shape)
    # x = tf.keras.applications.mobilenet_v2.preprocess_input(model_input)
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1)(model_input)
    pretrained_model = tf.keras.applications.vgg19.VGG19(include_top=False,
                                                                            input_shape=[224, 224, 3],
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
    #filters *= 2
    x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Decoder path
    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, filters, activation=None)
    encoder_input = res_path(skip_connections[3], filters, 1)
    attn_output = visual_attention_block(encoder_input, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, filters, activation=None)
    encoder_input = res_path(skip_connections[2], filters, 2)
    attn_output = visual_attention_block(encoder_input, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, filters, activation=None)
    encoder_input = res_path(skip_connections[1], filters, 3)
    attn_output = visual_attention_block(encoder_input, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    x = tf.keras.layers.Dropout(0.1)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, filters, activation=None)
    encoder_input = res_path(skip_connections[0], filters, 4)
    attn_output = visual_attention_block(encoder_input, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    x = tf.keras.layers.Dropout(0.1)(x)

    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model, pretrained_model


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
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, sc_4])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, sc_3])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, sc_2])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, sc_1])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model


def visual_attention_residual_unet(input_shape, starting_filters=16, activation='relu'):
    # Encoder path
    filters = starting_filters
    model_input = tf.keras.layers.Input(shape=input_shape)
    x = conv_block(model_input, filters, activation=activation)
    sc_1 = tf.keras.layers.Dropout(0.1)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_1)
    x = residual_block(x, filters)
    #x = conv_block(x, filters, activation=activation)
    sc_2 = tf.keras.layers.Dropout(0.1)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_2)
    x = residual_block(x, filters)
    #x = conv_block(x, filters, activation=activation)
    sc_3 = tf.keras.layers.Dropout(0.2)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_3)
    x = residual_block(x, filters)
    #x = conv_block(x, filters, activation=activation)
    sc_4 = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.AvgPool2D(2)(sc_4)

    # Bottleneck path
    filters *= 2
    x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Decoder path
    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    attn_output = visual_attention_block(sc_4, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    #x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    attn_output = visual_attention_block(sc_3, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    #x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    attn_output = visual_attention_block(sc_2, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    #x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    attn_output = visual_attention_block(sc_1, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    #x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)

    # Output
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model


def multires_visual_attention(input_shape, starting_filters=16, with_dropout=False, activation='relu'):
    layer_filters = starting_filters
    # Encoder path
    model_input = tf.keras.layers.Input(shape=input_shape)
    x = multi_res_block(model_input, layer_filters, activation=None)
    if with_dropout:
        x = tf.keras.layers.Dropout(0.1)(x)
    sc_1 = res_path(x, layer_filters, 4)

    layer_filters *= 2
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = multi_res_block(x, layer_filters, activation=None)
    if with_dropout:
        x = tf.keras.layers.Dropout(0.1)(x)
    sc_2 = res_path(x, layer_filters, 3)

    layer_filters *= 2
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = multi_res_block(x, layer_filters, activation=None)
    if with_dropout:
        x = tf.keras.layers.Dropout(0.2)(x)
    sc_3 = res_path(x, layer_filters, 2)

    layer_filters *= 2
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = multi_res_block(x, layer_filters, activation=None)
    if with_dropout:
        x = tf.keras.layers.Dropout(0.2)(x)
    sc_4 = res_path(x, layer_filters, 1)

    x = tf.keras.layers.MaxPool2D(2)(x)

    layer_filters *= 2
    # Bottleneck path
    x = multi_res_block(x, layer_filters, activation=None)
    if with_dropout:
        x = tf.keras.layers.Dropout(0.3)(x)

    # Decoder path
    layer_filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=layer_filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, layer_filters, activation=None)
    attn_output = visual_attention_block(sc_4, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    if with_dropout:
        x = tf.keras.layers.Dropout(0.2)(x)

    layer_filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=layer_filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, layer_filters, activation=None)
    attn_output = visual_attention_block(sc_3, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    if with_dropout:
        x = tf.keras.layers.Dropout(0.2)(x)

    layer_filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=layer_filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, layer_filters, activation=None)
    attn_output = visual_attention_block(sc_2, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    if with_dropout:
        x = tf.keras.layers.Dropout(0.1)(x)

    layer_filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=layer_filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, layer_filters, activation=None)
    attn_output = visual_attention_block(sc_1, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    if with_dropout:
        x = tf.keras.layers.Dropout(0.1)(x)

    # Output
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model


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
                attn_output = visual_attention_block(X[horizontal_level][k], x_dec)  # Multiple visual attention blocks
                x_dec = tf.keras.layers.Concatenate()([x_dec, attn_output])
                if with_dropout:
                    x_dec = tf.keras.layers.Dropout(0.2)(x_dec)
            x_dec = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(x_dec)
            sc = res_path(x_dec, upsample_filters, number_of_levels - horizontal_level - 1)
            X[horizontal_level].append(sc)
        filters *= 2

    # Output
    selected_segmentation_output = X[0][-1]
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(
        selected_segmentation_output)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model


def dilated_resnet(input_shape, starting_filters=16, activation='relu'):
    filters = starting_filters
    # Encoder path
    model_input = tf.keras.layers.Input(shape=input_shape)
    x = conv_block(model_input, filters, activation=activation)
    sc_1 = tf.keras.layers.Dropout(0.1)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_1)
    x = residual_block(x, filters)
    # x = conv_block(x, filters)
    sc_2 = tf.keras.layers.Dropout(0.1)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_2)
    x = residual_block(x, filters)
    # x = conv_block(x, filters)
    sc_3 = tf.keras.layers.Dropout(0.2)(x)

    filters *= 2
    x = tf.keras.layers.AvgPool2D(2)(sc_3)
    x = dilated_residual_block(x, filters, 2)
    # x = conv_block(x, filters)
    sc_4 = tf.keras.layers.Dropout(0.2)(x)

    filters *= 2
    x = dilated_residual_block(x, filters, 2)
    # x = conv_block(x, filters)
    sc_5 = tf.keras.layers.Dropout(0.2)(x)

    # Bottleneck path
    #filters *= 2
    #x = conv_block(x, filters, activation=activation)
    #x = conv_block(x, filters, activation=activation)
    #x = tf.keras.layers.Dropout(0.3)(x)

    #x = aspp(x, filters, activation=activation)
    x = aspp_with_image_level_features(x, sc_1, filters, activation=activation)

    # Decoder path
    filters //= 2
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, sc_5])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, sc_4])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, sc_3])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, sc_2])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = tf.keras.layers.Concatenate()([x, sc_1])
    # x = conv_block(x, filters)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model


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
    #x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = residual_block(x, filters)
    attn_output = visual_attention_block(sc_4, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    #x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    attn_output = visual_attention_block(sc_3, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    #x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.2)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    attn_output = visual_attention_block(sc_2, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    #x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)

    filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1, strides=2, padding='same')(x)
    x = residual_block(x, filters)
    attn_output = visual_attention_block(sc_1, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    #x = conv_block(x, filters, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = tf.keras.layers.Dropout(0.1)(x)

    # Output
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model


# Atrous Spatial Pyramid Pooling from DeepLabV3
def aspp(input, filters, activation):
    output_0 = conv_block(input, kernel_size=1, filters=filters, activation=activation)
    output_1 = conv_block(input, kernel_size=3, dilation_rate=6, filters=filters, activation=activation)
    output_2 = conv_block(input, kernel_size=3, dilation_rate=12, filters=filters, activation=activation)
    output_3 = conv_block(input, kernel_size=3, dilation_rate=18, filters=filters, activation=activation)
    output_4 = tf.keras.layers.GlobalAvgPool2D(keepdims=True)(input)
    output_4 = conv_block(output_4, filters=filters, kernel_size=1, activation=activation)
    output_4 = tf.keras.layers.UpSampling2D(size=input.shape[1], interpolation='bilinear')(output_4)
    result = tf.keras.layers.Concatenate()([output_0, output_1, output_2, output_3, output_4])

    return conv_block(result, kernel_size=1, filters=filters, activation=activation)


def aspp_with_image_level_features(input, image_level, filters, activation):
    output_0 = conv_block(input, kernel_size=1, filters=filters, activation=activation)
    output_1 = conv_block(input, kernel_size=3, dilation_rate=6, filters=filters, activation=activation)
    output_2 = conv_block(input, kernel_size=3, dilation_rate=12, filters=filters, activation=activation)
    output_3 = conv_block(input, kernel_size=3, dilation_rate=18, filters=filters, activation=activation)
    image_pooling = tf.keras.layers.AvgPool2D(8)(image_level)
    output_4 = conv_block(image_pooling, filters=filters, kernel_size=1, activation=activation)
    result = tf.keras.layers.Concatenate()([output_0, output_1, output_2, output_3, output_4])

    return conv_block(result, kernel_size=1, filters=filters, activation=activation)


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

    #x = aspp(x, filters)
    x = aspp_with_image_level_features(x, model_input, filters, activation=activation)

    # Decoder
    x1 = tf.keras.layers.UpSampling2D(size=4, interpolation='bilinear')(x)
    x2 = conv_block(sc_1, kernel_size=1, filters=filters, activation=activation)
    x = tf.keras.layers.Concatenate()([x1, x2])
    x = conv_block(x, kernel_size=3, filters=filters, activation=activation)
    x = conv_block(x, kernel_size=3, filters=filters, activation=activation)
    x = tf.keras.layers.UpSampling2D(size=4, interpolation='bilinear')(x)
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model


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
    x = tf.keras.layers.Conv2DTranspose(filters=layer_filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, layer_filters, activation=activation)
    attn_output = visual_attention_block(sc_3, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    if with_dropout:
        x = tf.keras.layers.Dropout(0.2)(x)

    layer_filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=layer_filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, layer_filters, activation=activation)
    attn_output = visual_attention_block(sc_2, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    if with_dropout:
        x = tf.keras.layers.Dropout(0.1)(x)

    layer_filters //= 2
    x = tf.keras.layers.Conv2DTranspose(filters=layer_filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, layer_filters, activation=activation)
    attn_output = visual_attention_block(sc_1, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    if with_dropout:
        x = tf.keras.layers.Dropout(0.1)(x)

    # Output
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model


