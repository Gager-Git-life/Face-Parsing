from .Convs import *


def FS_UNET(input_shape=(256, 256, 3)):

    # [256, 256, 3]
    inputs = layers.Input(shape=input_shape)

    # [256, 256, 3] --> [256, 256, 8]
    conv1 = Conv2D_(inputs, filters=8, kernel_size=3, strides=1)

    # [256, 256, 8] --> [128, 128, 16]
    conv2_1 = Mv3Resblock(conv1, in_filter=8, out_filter=16, kernel_size=3, strides=2, expand_ratio=2)
    # [128, 128, 16] --> [64, 64, 16]
    conv2_2 = Mv3Resblock(conv2_1, in_filter=16, out_filter=16, kernel_size=3, strides=2, expand_ratio=2)

    # [64, 64, 16] --> [32, 32, 32]
    conv3_1 = Mv3Resblock(conv2_2, in_filter=16, out_filter=32, kernel_size=3, strides=2, expand_ratio=2)
    # [32, 32, 32] --> [16, 16, 32]
    conv3_2 = Mv3Resblock(conv3_1, in_filter=32, out_filter=32, kernel_size=3, strides=2, expand_ratio=2)

    # [16, 16, 32] --> [8, 8, 48]
    conv4 = Mv3Resblock(conv3_2, in_filter=32, out_filter=48, kernel_size=3, strides=2, expand_ratio=2)

    # [8, 8, 48]
    conv5 = DenseBlock(conv4, filtes=48, iters=6)
    # conv5 = Mv3Resblock(conv4, in_filter=128, out_filter=64, kernel_size=3, strides=1, expand_ratio=1.5)
    # conv5 = Inverted_res_block(conv4, in_filter=128, out_filter=128, kernel_size=2, strides=1, expand_ratio=1.5, repeat=5)

    # [8, 8, 96]
    concat_5 = tf.concat([conv5, conv4], axis=-1)
    # [16, 16, 64]
    dconv5 = Conv2D_(concat_5, filters=64, kernel_size=3, strides=2, model='convTra')

    # [16, 16, 96]
    concat_4 = tf.concat([dconv5, conv3_2], axis=-1)
    # [32, 32, 48]
    dconv4 = Conv2D_(concat_4, filters=48, kernel_size=3, strides=2, model='convTra')

    # [32, 32, 80]
    concat_3 = tf.concat([dconv4, conv3_1], axis=-1)
    # [64, 64, 32]
    dconv3 = Conv2D_(concat_3, filters=32, kernel_size=3, strides=2, model='convTra')

    # [64, 64, 48]
    concat_2 = tf.concat([dconv3, conv2_2], axis=-1)
    # [128, 128, 16]
    dconv2 = Conv2D_(concat_2, filters=16, kernel_size=3, strides=2, model='convTra')

    # [128, 128, 32]
    concat_1 = tf.concat([dconv2, conv2_1], axis=-1)
    # [256, 256, 8]
    dconv1 = Conv2D_(concat_1, filters=8, kernel_size=3, strides=2, model='convTra')

    # [256, 256, 16]
    concat_0 = tf.concat([dconv1, conv1], axis=-1)
    # [256, 256, 16]
    # out = Conv2D_(concat_0, filters=16, kernel_size=3, strides=1)
    # # [256, 256, 5]
    out = Conv2D_(concat_0, filters=7, kernel_size=1, strides=1, activation='')

    out = tf.keras.activations.sigmoid(out)
    # out_mask = tf.argmax(out, axis=-1)
    print(out.shape)

    model = keras.Model(inputs=inputs, outputs=[out])
    return model



