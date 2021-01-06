import cv2
import time
import datetime
import tensorflow as tf
from models.utils import *
from models.FS_model import FS_UNET
from models.Datasets import *
from models.Loss import *
from tensorflow.keras import optimizers

# 是否使用gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

train_batch = 32
test_batch = 1
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


# 1.数据集准备
inputs_dataset = CelebAMaskHQ(input_path, 256, 3, train_batch, test_batch, repeat)
multidataset = SequenceData([hair_path, skin_no_f_path, brow_path, eye_path, nose_path, lip_path, mouth_path],
                            256, 1, train_batch, test_batch, repeat)

train_inputs = inputs_dataset.train_dataset
test_inputs = inputs_dataset.test_dataset
train_labels = multidataset.train_dataset
test_labels = multidataset.test_dataset


# 1.tf模型推理
# model = tf.keras.models.load_model('Model/h5/face_parse.h5', compile=False)
# print(model.summary())
#
#
# for i in range(100):
#     test_input_names, test_input_batch = next(test_inputs)
#     test_label_names, test_label_batch = next(test_labels)
#     print(test_input_batch.shape, test_label_batch.shape)
#     bt = time.time()
#     test_save(model, test_input_batch, test_label_batch, i, "pic/test_batch")
#     print("[INFO]>>> 用时:{}".format(time.time() - bt))



# 2.tflite模型推理
color_lists = [[0, 0, 0], [30, 144, 255], [255, 222, 173], [128, 0, 0], [238, 130, 238],
               [255, 140, 0], [255, 0, 0], [0, 250, 154]]

def draw(ori_image, mask_image):
    # print(type(ori_image))
    shape_ori = ori_image.shape
    mix_image = copy.deepcopy(ori_image)

    mask_image = np.concatenate([np.zeros(shape=(256, 256, 1)), mask_image], axis=-1)
    max_index = np.argmax(mask_image, axis=-1)

    for m in range(shape_ori[0]):
        for n in range(shape_ori[1]):

            index = int(max_index[m, n])
            if(index != 0 and mask_image[m, n, index] > 0.5):
                color_ = color_lists[index]
                mix_image[m, n] = 0.7 * np.array(color_) / 255. + 0.3 * mix_image[m, n]

    return mix_image



interpreter = tf.lite.Interpreter(model_path='Model/tflite/face_parse.tflite')
interpreter.allocate_tensors()

print(interpreter.get_input_details())
print(interpreter.get_output_details())
# 获取输入和输出张量。
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 使用随机数据作为输入测试 TensorFlow Lite 模型。
for i in range(100):
    test_input_names, test_input_batch = next(test_inputs)
    test_label_names, test_label_batch = next(test_labels)

    test_input_batch = np.array(test_input_batch + 1) / 2
    test_label_batch = np.array(test_label_batch)
    bt = time.time()
    interpreter.set_tensor(input_details[0]['index'], test_input_batch)
    interpreter.invoke()
#
#     # 函数 `get_tensor()` 会返回一份张量的拷贝。
#     # 使用 `tensor()` 获取指向张量的指针。
    masks = interpreter.get_tensor(output_details[0]['index'])
    print('[INFO]>>> 用时:{}'.format(time.time() - bt))
    plt.figure(figsize=(9, 9))
    for j in range(masks.shape[0]):
        plt.subplot(4, 3, j * 3 + 1)
        plt.axis('off')
        plt.imshow(test_input_batch[j, :, :, :])
        plt.subplot(4, 3, j * 3 + 2)
        plt.axis('off')
        plt.imshow(draw(test_input_batch[j, :, :, :], test_label_batch[j, :, :, :]))
        plt.subplot(4, 3, j * 3 + 3)
        plt.axis('off')
        plt.imshow(draw(test_input_batch[j, :, :, :], masks[j, :, :, :]))

    plt.savefig('{}/image_at_step_{}.png'.format("pic/tflite_batch", i))
    plt.close()