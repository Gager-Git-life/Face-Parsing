import os
import cv2
import numpy as np
import os.path as osp
import tensorflow as tf
from multiprocessing import Process, Queue

queue = Queue(maxsize=8)

class CelebAMaskHQ():
    def __init__(self, img_path, resize=256, channel=3, train_batch=16, test_batch=4, repeat=10):
        self.img_path = img_path
        self.image_channels = channel
        self.resize = resize
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.repeat = repeat

        self.files_input = os.listdir(self.img_path)
        self.files_input.sort()
        self.nums = len(self.files_input[:-100]) * repeat

        self.img_paths = list(map(lambda x: osp.join(self.img_path, x), self.files_input))
        # 训练数据
        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.img_paths[:-100])
        self.train_dataset = self.train_dataset.map(self.singlechannel)
        self.train_dataset = self.train_dataset.repeat(self.repeat)
        self.train_dataset = self.train_dataset.batch(self.train_batch)
        self.train_dataset = iter(self.train_dataset)

        # 测试数据
        self.test_dataset = tf.data.Dataset.from_tensor_slices(self.img_paths[-100:])
        self.test_dataset = self.test_dataset.map(self.singlechannel)
        self.test_dataset = self.test_dataset.repeat(self.repeat)
        self.test_dataset = self.test_dataset.batch(self.test_batch)
        self.test_dataset = iter(self.test_dataset)

    def loadfile(self, filename, channel):
        file_contents = tf.io.read_file(filename)
        image = tf.image.decode_png(file_contents, channels=channel)
        image = tf.image.resize(image, (self.resize, self.resize))
        return image

    def singlechannel(self, filename):
        image = self.loadfile(filename, self.image_channels)
        image = tf.cast(image, tf.float32)
        image = (image / 127.5 - 1)
        return [filename, image]


class SequenceData():
    def __init__(self, mask_dirs, size, channel=1, train_batch=16, test_batch=4, repeat=1):
        self.mask_dirs = mask_dirs
        self.mask_train_lists = []
        self.mask_test_lists = []
        self.repeat = repeat
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.resize = size
        self.channel = channel
        self.image_channels = 1
        # self.queue = Queue(maxsize=8)

        assert isinstance(self.mask_dirs, list), "输入不是列表"
        self.getmultimasklists()
        # for x in self.mask_train_lists:
        #     print(x[0])
        # 训练数据
        self.train_dataset = tf.data.Dataset.from_tensor_slices(tuple(x for x in self.mask_train_lists))
        self.train_dataset = self.train_dataset.map(self.multipic)
        self.train_dataset = self.train_dataset.repeat(self.repeat)
        self.train_dataset = self.train_dataset.batch(self.train_batch)
        self.train_dataset = iter(self.train_dataset)

        # 测试数据
        self.test_dataset = tf.data.Dataset.from_tensor_slices(tuple(x for x in self.mask_test_lists))
        self.test_dataset = self.test_dataset.map(self.multipic)
        self.test_dataset = self.test_dataset.repeat(self.repeat)
        self.test_dataset = self.test_dataset.batch(self.test_batch)
        self.test_dataset = iter(self.test_dataset)

    def getmultimasklists(self):
        for dir in self.mask_dirs:
            mask_paths = os.listdir(dir)
            mask_paths.sort()
            mask_paths = list(map(lambda x: osp.join(dir, x), mask_paths))
            mask_train_paths = mask_paths[:-100]
            mask_test_paths = mask_paths[-100:]
            self.mask_train_lists.append(mask_train_paths)
            self.mask_test_lists.append(mask_test_paths)
        # print("done")

    def loadfile(self, filename, indx):
        # print("done")
        file_contents = tf.io.read_file(filename)
        image = tf.image.decode_png(file_contents, channels=self.channel)
        image = tf.image.resize(image, (self.resize, self.resize))
        image = tf.where(image > 225, indx, 0.)
        # image = tf.cast(image, tf.float32)
        # image = (image / 127.5 - 1)
        return image

    def multipic(self, hair_path, skin_no_f_path, brow_path, eye_path, nose_path, lip_path, mouth_path):
        # mask_hair = self.loadfile(hair_path)
        # mask_ear = self.loadfile(ear_path)
        # mask_skin = self.loadfile(skin_path)
        # mask_brow = self.loadfile(brow_path)
        # mask_eye = self.loadfile(eye_path)
        # mask_nose = self.loadfile(nose_path)
        # mask_lip = self.loadfile(lip_path)
        # mask_mouth = self.loadfile(mouth_path)
        # images = tf.zeros(shape=(self.resize, self.resize, 1))
        images = None
        # path_lists = [hair_path, ear_path, skin_path, brow_path, eye_path, nose_path, lip_path, mouth_path]
        path_lists = [hair_path, skin_no_f_path, brow_path, eye_path, nose_path, lip_path, mouth_path]
        # score_list = [0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.00]
        # score_list = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
        # score_list = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        for i, path in enumerate(path_lists, 1):
            img = self.loadfile(path, float(i))

            if(images is None):
                images = img
            else:
                images = tf.concat([images, img], axis=-1)

        # out_mask = tf.argmax(images, axis=-1)


        return [brow_path, images]

class NewDatasets():
    def __init__(self, input_dir, label_dir, size=256, train_batch=16, test_batch=4, repeat=1):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.train_lists = []
        self.test_lists = []

        self.bright = False
        self.contrast = False
        self.random_crop = False

        self.repeat = repeat
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.resize = size

        self.GetFileLists()

        # 训练数据
        self.train_dataset = tf.data.Dataset.from_tensor_slices(tuple(x for x in self.train_lists))
        self.train_dataset = self.train_dataset.map(self.multipic)
        self.train_dataset = self.train_dataset.repeat(self.repeat)
        self.train_dataset = self.train_dataset.batch(self.train_batch)
        self.train_dataset = iter(self.train_dataset)

        # 测试数据
        self.test_dataset = tf.data.Dataset.from_tensor_slices(tuple(x for x in self.test_lists))
        self.test_dataset = self.test_dataset.map(self.multipic)
        self.test_dataset = self.test_dataset.repeat(self.repeat)
        self.test_dataset = self.test_dataset.batch(self.test_batch)
        self.test_dataset = iter(self.test_dataset)

    def GetFileLists(self):
        for path in (self.input_dir, self.label_dir):
            file_paths = os.listdir(path)
            file_paths.sort()
            file_paths = list(map(lambda x: osp.join(path, x), file_paths))
            self.nums = len(file_paths)
            self.train_lists.append(file_paths[:-100])
            self.test_lists.append(file_paths[-100:])

    def loadfile(self, filename, channel=3):
        file_contents = tf.io.read_file(filename)
        image = tf.image.decode_png(file_contents, channels=channel)
        image = tf.image.resize(image, (self.resize, self.resize))
        return image

    def multipic(self, input_path, label_path):
        input_img = self.loadfile(input_path, 3)
        label_img = self.loadfile(label_path, 1)
        # combile_img = tf.concat([input_img, label_img], axis=-1)
        # if(self.bright):
        #     combile_img = tf.image.random_brightness(combile_img, max_delta=0.1)
        # elif(self.contrast):
        #     combile_img = tf.image.random_contrast(combile_img, lower=1, upper=3)
        # elif(self.random_crop):
        #     combile_img = tf.image.random_crop(combile_img, size=(self.resize, self.resize))
        # input_img = tf.concat(tf.unstack(combile_img, axis=-1)[:-1], axis=-1)

        input_img = input_img / 127.5 - 1
        return [input_img, label_img]



