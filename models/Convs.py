from .Activations import *

def Conv2D_(inputs, filters, kernel_size, strides, activation='relu', bn='BN', padding='same', model='conv'):
    if(model == 'dw'):
        x = layers.SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    elif(model == 'conv'):
        x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    else:
        x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)

    if(isinstance(bn, str)):
        if(bn == 'IN'):
            x = InstanceNormalization()(x)
        elif(bn == 'LN'):
            x = LayerInsNorm()(x)
        else:
            x = layers.BatchNormalization()(x)

    if(activation == 'relu'):
        # return layers.ReLU(max_value=6.0)(x)
        return layers.Activation(tf.nn.relu)(x)
    elif(activation == 'hs'):
        return h_swish(x)
    elif(activation == 'tanh'):
        return tf.keras.activations.tanh(x)
    elif(activation == "softmax"):
        return tf.keras.activations.softmax(x)
    else:
        return x


def SEblock(inputs, alpha=1):
    input_channel = inputs.shape[-1]
    branch = layers.GlobalAvgPool2D()(inputs)
    branch = layers.Dense(units=input_channel*alpha)(branch)
    branch = layers.Activation(tf.nn.relu)(branch)
    branch = layers.Dense(units=input_channel)(branch)
    branch = h_sigmoid(branch)
    branch = tf.expand_dims(branch, axis=1)
    branch = tf.expand_dims(branch, axis=1)

    output = inputs * branch
    return output

def Mv3Resblock(inputs, in_filter, out_filter, kernel_size=3, strides=1, expand_ratio=1, use_se=True, use_res=False, activation='relu'):
    up_channel = int(in_filter * expand_ratio)
    # 1x1升维
    x = Conv2D_(inputs=inputs, filters=up_channel, kernel_size=(1, 1), strides=1, activation=activation)

    # 分离卷积
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', depth_multiplier=1)(x)
    x = layers.BatchNormalization()(x)
    if(activation == "relu"):
        # x = layers.ReLU(max_value=6.0)(x)
        x = layers.Activation(tf.nn.relu)(x)
    elif(activation == "hs"):
        # x = h_swish(x)
        x = layers.Activation(tf.nn.swish)(x)
    elif(activation == "selu"):
        x = layers.Activation(tf.nn.selu)(x)

    # 添加se模块
    if(use_se):
        x = SEblock(x)
        # x = CBAMblock(x)

    # 1x1将维
    x = Conv2D_(inputs=x, filters=out_filter, kernel_size=(1, 1), strides=1, activation=activation)
    x = layers.BatchNormalization()(x)


    if use_res and in_filter == out_filter:
        x = layers.Add()([inputs, x])

    return x

def Inverted_res_block(inputs, in_filter, out_filter, kernel_size=3, strides=1, expand_ratio=1, repeat=2):
    out = Mv3Resblock(inputs, in_filter, out_filter, kernel_size, strides=strides, expand_ratio=expand_ratio, use_res=False)
    for i in range(1, repeat):
        out = Mv3Resblock(out, out_filter, out_filter, kernel_size, strides=1, expand_ratio=expand_ratio)

    return out

def DenseBlock(inputs, filtes, iters):

    collects = [inputs]
    out = collects[0]
    acts = ["relu", "relu", "relu", "relu", "relu", "relu"]

    for i in range(iters):
        out = Mv3Resblock(out, filtes, filtes, kernel_size=3, strides=1, expand_ratio=2, use_res=False, activation=acts[i])
        for layer in collects:
            out = layers.Add()([out, layer])
        collects.append(out)
    return out



def CAblock(inputs, ratio=8):

    input_channel = inputs.shape[-1]

    maxpool = layers.GlobalMaxPool2D()(inputs)
    maxpool = tf.reshape(maxpool, shape=(-1, 1, 1, maxpool.shape[-1]))
    avgpool = layers.GlobalAveragePooling2D()(inputs)
    avgpool = tf.reshape(avgpool, shape=(-1, 1, 1, avgpool.shape[-1]))

    shared_fc1 = layers.Conv2D(filters=int(input_channel/8), kernel_size=1, strides=1)
    shared_fc2 = layers.Conv2D(filters=input_channel, kernel_size=1, strides=1)
    shared_act = layers.Activation(tf.nn.relu)

    max_out = shared_fc2(shared_act(shared_fc1(maxpool)))
    avg_out = shared_fc2(shared_act(shared_fc1(avgpool)))

    outputs = h_swish(max_out + avg_out)

    return outputs

def SAblock(inputs, kernel_size=7):

    maxpool = tf.reduce_mean(inputs, axis=-1)
    avgpool = tf.reduce_max(inputs, axis=-1)

    concat = tf.stack([maxpool, avgpool], axis=-1)

    outputs = layers.Conv2D(filters=1, kernel_size=kernel_size, padding='same', strides=1)(concat)
    outputs = h_swish(outputs)

    return outputs


def CBAMblock(inputs):

    outputs = SAblock(inputs) * inputs
    outputs = CAblock(outputs) * outputs

    return outputs