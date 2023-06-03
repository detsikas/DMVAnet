import tensorflow as tf


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
    x = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=3, padding='same')(x)

    res = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=1, padding='same')(inputs)
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
    x = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=3, padding='same', dilation_rate=dilation_rate)(x)

    res = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=1, padding='same', dilation_rate=dilation_rate)(inputs)
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
    x1 = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=3, padding='same')(inputs)
    x2 = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=3, padding='same')(x1)
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
    m_sp = tf.keras.layers.Conv2D(
        filters=1, kernel_size=3, padding='same', activation='sigmoid')(f_sp_avg)

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
    x_feat = tf.keras.layers.Reshape(
        (x_feat.shape[1] * x_feat.shape[2], x_feat.shape[3]))(x_feat)
    x_am = tf.keras.layers.Conv2D(filters=n, kernel_size=1)(x)
    x_am = tf.keras.layers.Softmax(axis=(1, 2))(x_am)
    x_am = tf.keras.layers.Reshape(
        (x_am.shape[1] * x_am.shape[2], x_am.shape[3]))(x_am)
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
    shortcut = conv_block(inputs, int(
        W * 0.167) + int(W * 0.333) + int(W * 0.5), kernel_size=1, activation=None)
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
    conv3x3 = conv_block(inputs, int(W * 0.167),
                         dilation_rate=dilation_rate, activation=None)
    conv5x5 = conv_block(conv3x3, int(W * 0.333),
                         dilation_rate=dilation_rate, activation=None)
    conv7x7 = conv_block(conv5x5, int(W * 0.5),
                         dilation_rate=dilation_rate, activation=None)

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
