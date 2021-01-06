import tensorflow as tf

def read_mask(filename):
    file_contents = tf.io.read_file(filename)
    image = tf.image.decode_png(file_contents, channels=1)
    image = tf.image.resize(image, (256, 256))
    return image

def cross_entropyCC(input, target):
    cross_entropy = tf.keras.losses.CategoricalCrossentropy()
    loss_ = cross_entropy(input, target)
    return loss_


def cross_entropyBC(input, target):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss_ = cross_entropy(input, target)
    return loss_

def Multicross_entropyBC(inputs, targets):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    inputs_lists = tf.unstack(inputs, axis=-1)
    targets_lists = tf.unstack(targets, axis=-1)
    # zero_mask = tf.zeros(shape=(inputs.shape[0], 256, 256, 1))
    # add_inputs = tf.concat([zero_mask, inputs], axis=-1)
    # add_tragets = tf.concat([zero_mask, targets], axis=-1)
    # max_train_mask = tf.argmax([add_inputs], axis=-1)
    # max_label_mask = tf.argmax([add_tragets], axis=-1)
    # max_train_mask = tf.cast(max_train_mask, dtype=tf.float32)
    # max_label_mask = tf.cast(max_label_mask, dtype=tf.float32)
    # class_loss = tf.reduce_mean(tf.losses.MSE(max_train_mask, max_label_mask)) * 5
    class_lists = ["hair", "skin", "brow", "eye", "nose", "lip", "mouse"]
    loss_lists = []
    factors = [2, 2, 10, 15, 10, 10, 15]
    count = 0
    for input, target in zip(inputs_lists, targets_lists):
        loss_ = cross_entropy(input, target)
        loss_lists.append(loss_ * factors[count])
        count += 1
    # loss_lists.append(class_loss)
    sum_loss = sum(loss_lists)
    out = dict(zip(class_lists, loss_lists))
    return sum_loss, out

def Multicross_entropyCC(inputs, targets):
    cc_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # max_train_mask = tf.argmax([inputs], axis=-1)
    # max_label_mask = tf.argmax([targets], axis=-1)
    # max_train_mask = tf.cast(max_train_mask, dtype=tf.float32)
    # max_label_mask = tf.cast(max_label_mask, dtype=tf.float32)
    inputs_lists = tf.unstack(inputs, axis=-1)
    targets_lists = tf.unstack(targets, axis=-1)

    class_lists = ["brow", "eye", "nose", "lip", "mouse"]
    loss_facotr = [1., 1., 1., 1., 1., 1., 1.]
    loss_lists = []
    count = 1
    for input, target in zip(inputs_lists, targets_lists):
        loss_ = cc_loss(input, target/count)
        loss_lists.append(loss_)
        count += 1
    # loss_ = cc_loss(max_train_mask, max_label_mask) / 10
    # loss_lists.append(loss_)
    sum_loss = sum(loss_lists)
    out = dict(zip(class_lists, loss_lists))
    return sum_loss, out
