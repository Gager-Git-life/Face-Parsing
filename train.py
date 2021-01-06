import cv2
import datetime
import tensorflow as tf
from models.utils import *
from models.FS_model import FS_UNET
from models.Datasets import *
from models.Loss import *
from tensorflow.keras import optimizers
from models.lr_scheduler import MultiStepWarmUpLR

init_lr = 2e-3
train_batch = 32
test_batch = 4
max_epoch = 100
repeat = 1000

input_path = './dataset/inputs'
label_path = './dataset/masks'

hair_path = './dataset/hair'
ear_path = './dataset/ear'
skin_path = './dataset/skin_merger'
neck_path = './dataset/neck'
brow_path = './dataset/brow'
eye_path = './dataset/eye'
nose_path = './dataset/nose'
lip_path = './dataset/lip'
mouth_path = './dataset/mouth'
skin_no_f_path = './dataset/skin_no_face'


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# 1.数据集准备
inputs_dataset = CelebAMaskHQ(input_path, 256, 3, train_batch, test_batch, repeat)
multidataset = SequenceData([hair_path, skin_no_f_path, brow_path, eye_path, nose_path, lip_path, mouth_path],
                            256, 1, train_batch, test_batch, repeat)

train_inputs = inputs_dataset.train_dataset
test_inputs = inputs_dataset.test_dataset
train_labels = multidataset.train_dataset
test_labels = multidataset.test_dataset

# display_test(train_inputs, train_labels)

# dataset = NewDatasets(input_path, label_path)
# train_datasets = dataset.train_dataset
# test_datasets = dataset.test_dataset
# display_data(train_datasets)



# 固定测试数据
# print("#" * 32)
def get_next_test_batch(num):
    # test_input_batch, test_label_batch = next(test_datasets)
    for _ in range(num):
        test_input_names, test_input_batch = next(test_inputs)
        test_label_names, test_label_batch = next(test_labels)
    return test_input_batch, test_label_batch

test_input_batch, test_label_batch = get_next_test_batch(6)


# 2.模型准备
fs_model = FS_UNET()
fs_model.summary()
# tf.keras.utils.plot_model(fs_model, to_file='pic/fc_segment.png', show_shapes=True, dpi=96)

# 优化器及学习策略选择
steps_per_epoch = inputs_dataset.nums // inputs_dataset.train_batch
# 阶梯学习率
# learning_rate = MultiStepWarmUpLR(
#     initial_learning_rate=init_lr,
#     lr_steps=[e * steps_per_epoch for e in [4, 30, 50, 70, 100, 200, 300]],
#     lr_rate=[1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1], #[1e-4, 1.4e-4, 1.5e-4, 1e-6]
#     warmup_steps=steps_per_epoch,
#     min_lr=1e-7)

# cos学习率衰减
learning_rate = tf.keras.experimental.CosineDecay(
                initial_learning_rate=init_lr, decay_steps=20000)

fs_optimizer = optimizers.Adam(learning_rate=learning_rate)

# 验证
val_metrices = tf.keras.metrics.BinaryCrossentropy()

# 模型存储
checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                 fs_optimizer=fs_optimizer,
                                 fs_model=fs_model
                                 )
ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

# 如果存在检查点，恢复最新版本检查点
if ckpt_manager.latest_checkpoint:
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    print('[INFO]>>> load ckpt from {} at step {}.'.format(
        ckpt_manager.latest_checkpoint, checkpoint.step.numpy()))

per_steps = checkpoint.step.numpy()
if(per_steps > 0):
    print('[INFO]>>> find per train steps:{}'.format(per_steps))

# 训练进度条
prog_bar = ProgressBar(steps_per_epoch,
                       checkpoint.step.numpy() % steps_per_epoch)

# 训练日志设置
log_dir = "logs/"
summary_writer = tf.summary.create_file_writer(
  log_dir + datetime.datetime.now().strftime("%Y%m%d-%H"))

@tf.function
def train_step(model, input, label):
    input_name, input_img = input
    label_name, label_img = label

    with tf.GradientTape() as tape:
        mask = model(input_img, training=True)
        # out_mask = tf.argmax(mask, axis=-1)
        # label_mask = tf.argmax(label_img, axis=-1)
        # 计算损失
        print("#"*32)
        cross_entropy_loss, loss_dict = Multicross_entropyBC(mask, label_img)
        # class_loss = cross_entropyBC(out_mask, label_mask)

    fs_gradients = tape.gradient([cross_entropy_loss], model.trainable_variables)
    fs_optimizer.apply_gradients(zip(fs_gradients, model.trainable_variables))

    return mask, cross_entropy_loss, loss_dict,

def average_val_loss(model):
    # test_input_batch, test_label_batch, test_index_batch = get_next_test_batch()
    test_mask = model(test_input_batch, training=False)
    val_metrices.reset_states()
    val_metrices.update_state(test_mask, test_label_batch)
    return val_metrices.result().numpy()

# 训练
def train():
    for epoch in range(max_epoch):
        print("[INFO]>>> epoch:{}".format(epoch))
        for img_batch, label_batch in zip(train_inputs, train_labels):
            checkpoint.step.assign_add(1)
            steps = checkpoint.step.numpy()

            mask, loss, loss_dict = train_step(fs_model, img_batch, label_batch)
            # if(fs_optimizer.lr(steps) < 1e-6 or loss < 0.5):
            #     break
            # val_loss = average_val_loss(fs_model)
            prog_bar.update("epoch={}/{}, step{}, loss={:.4f}, lr={:.2e}".format(
                ((steps - 1) // steps_per_epoch) + 1, max_epoch, steps, loss, fs_optimizer.lr(steps).numpy()))

            if(steps % 2 == 0):
                with summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=steps)
                    # tf.summary.scalar('val_loss', val_loss, step=steps)
                    tf.summary.scalar('learning_rate', fs_optimizer.lr(steps), step=steps)
                    for key, value in loss_dict.items():
                        tf.summary.scalar(key, value, step=steps)
                    # tf.summary.scalar("class_loss", class_loss, step=steps)

            if(steps % 100 == 0):
                ctime = time.strftime("%Y/%m/%d/ %H:%M:%S")
                ckpt_save_path = ckpt_manager.save()
                logs = 'Time:{}, Epoch={}, save_path={}'
                tf.print(tf.strings.format(logs, (ctime, epoch, ckpt_save_path)), output_stream=sys.stdout)
                # test_input_batch, test_label_batch, test_index_batch = get_next_test_batch()
                test_save(fs_model, test_input_batch, test_label_batch, steps / 100, "pic/save_batch_new")


if __name__ == "__main__":
    # all_begin()
    train()
    # fs_model.save('models/fs_model.h5')
