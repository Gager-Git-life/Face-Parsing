import os
import sys
import cv2
import time
import glob
import copy
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# color_lists = [[0, 0, 0], [30, 144, 255], [244, 164, 96], [255, 222, 173], [0, 0, 255],
#                [220, 20, 60], [255, 140, 0], [255, 0, 0], [0, 250, 154]]

color_lists = [[0, 0, 0], [30, 144, 255], [255, 222, 173], [128, 0, 0], [238, 130, 238],
               [255, 140, 0], [255, 0, 0], [0, 250, 154]]

score_list = [0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.00]

def get_color(input, var):
    color = input
    if(var in score_list):
        index = score_list.index(var)
        color = np.array(color_lists[index], dtype=np.float) / 255.

    return 0.5 * color + 0.5 * input


def display_test(dataset1, dataset2, idx=1):
    global color_lists
    color_lists = list(map(lambda x: x[::-1], color_lists))
    _, input = next(dataset1)
    input = input.numpy()
    input = (input[...] + 1) / 2
    input = input[idx, :, :, :]
    input = input[..., ::-1]

    names, masks = next(dataset2)
    masks = masks.numpy()
    mask = masks[idx, :, :, :]
    mask_ = np.concatenate([np.zeros(shape=(256, 256, 1)), mask], axis=-1)
    max_index = np.argmax(mask_, axis=-1)
    # maxindex = maxindex.numpy()
    # maxindex = maxindex[idx, :, :]

    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            color_ = color_lists[int(max_index[i, j])]
            input[i, j] = 0.5 * np.array(color_) / 255. + 0.5 * input[i, j]
            #-----
            # color_ = input[i, j]
            # for k in range(mask.shape[2]):
            #     if(mask[i, j, k] > 0.8):
            #         color_ = color_lists[k]
            # input[i, j] = 0.5 * np.array(color_) / 255. + 0.5 * input[i, j]
            #-----
            # index = int(max_index[i, j])
            # input[i, j] = 0.5 * np.array(color_lists[index]) / 255. + 0.5 * input[i, j]
            # input[i, j] = get_color(color_, mask[i, j])
    cv2.imwrite("./pic/{}_mask.jpg".format(idx), input * 255)
    # cv2.imshow("{}".format(idx), input)
    # cv2.waitKey()


def display_data(datasets, indx=0):
    input_imgs, label_imgs = next(datasets)
    input_imgs = input_imgs.numpy()
    label_imgs = label_imgs.numpy()
    input_img = input_imgs[indx, :, :, :]
    label_img = label_imgs[indx, :, :, 0]
    input_img = (input_img[...] + 1) / 2
    input_img = input_img[..., ::-1] * 255
    out_img = np.zeros(shape=input_img.shape)
    for i in range(1, 9):
        out_img[label_img == i] = np.array(color_lists[i-1])[::-1]
    out_img = 0.5 * out_img + 0.5 * input_img
    cv2.imwrite("./pic/{}.jpg".format(indx), out_img )




def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(len(gpus))
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                print(
                    "Detect {} Physical GPUs, {} Logical GPUs.".format(
                        len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print('[INFO]>>> current use cpu!!!')

class ProgressBar(object):
    """A progress bar which can print the progress modified from
       https://github.com/hellock/cvbase/blob/master/cvbase/progress.py"""
    def __init__(self, task_num=0, completed=0, bar_width=25):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width
                          if bar_width <= max_bar_width else max_bar_width)
        self.completed = completed
        self.first_step = completed
        self.warm_up = False

    def _get_max_bar_width(self):
        if sys.version_info > (3, 3):
            from shutil import get_terminal_size
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider '
                         'widen the terminal for better progressbar '
                         'visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def reset(self):
        """reset"""
        self.completed = 0
        self.fps = 0

    def update(self, inf_str=''):
        """update"""
        self.completed += 1

        if not self.warm_up:
            self.start_time = time.time() - 1e-1
            self.warm_up = True

        if self.completed > self.task_num:
            self.completed = self.completed % self.task_num
            self.start_time = time.time() - 1 / self.fps
            self.first_step = self.completed - 1
            sys.stdout.write('\n')

        elapsed = time.time() - self.start_time
        self.fps = (self.completed - self.first_step) / elapsed
        percentage = self.completed / float(self.task_num)
        mark_width = int(self.bar_width * percentage)
        bar_chars = '>' * mark_width + ' ' * (self.bar_width - mark_width)
        stdout_str = '\rTraining [{}] {}/{}, {}  {:.1f} step/sec'
        sys.stdout.write(stdout_str.format(
            bar_chars, self.completed, self.task_num, inf_str, self.fps))

        sys.stdout.flush()

def get_gif(pic_dir, pic_format, save_path='dcgan.gif'):

    with imageio.get_writer(save_path, mode='I') as writer:
      filenames = glob.glob('{}/{}'.format(pic_dir, pic_format))
      filenames = sorted(filenames)
      last = -1
      for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
          last = frame
        else:
          continue
        image = imageio.imread(filename)
        writer.append_data(image)
      image = imageio.imread(filename)
      writer.append_data(image)

def all_begin():
    os.system("rm training_checkpoints/*")
    # os.system("rm -r logs/*")
    # os.system("rm pic/save_batch/*")

def pic_mix(ori_image, mask_image, train=False):
    shape_ori = ori_image.shape
    mix_image = copy.deepcopy(ori_image)

    mask_image = np.concatenate([np.zeros(shape=(256, 256, 1)), mask_image], axis=-1)
    max_index = np.argmax(mask_image, axis=-1)

    for i in range(shape_ori[0]):
        for j in range(shape_ori[1]):

            # color_ = ori_image[i, j]
            # for k in range(mask_image.shape[2]):
            #     if(mask_image[i, j, k]  > 0.5):
            #         color_ = color_lists[k+1]
            index = int(max_index[i, j])

            if(index != 0 and mask_image[i, j, index] > 0.5):
                color_ = color_lists[index]
                mix_image[i, j] = 0.7 * np.array(color_) / 255. + 0.3 * mix_image[i, j]
            # -----
            # color_ = ori_image[i, j]
            # for k in range(mask_image.shape[2]):
            #     if(mask_image[i, j, k] > 0.8):
            #         color_ = color_lists[k]
            # mix_image[i, j] = 0.5 * np.array(color_) / 255. + 0.5 * mix_image[i, j]
            #-----
            # index_ = int(max_index[i, j])
            # out_img[i, j] = 0.5 * np.array(color_lists[index_]) / 255. + 0.5 * ori_image[i, j]
    return mix_image

def test_save(model, inputs, labels, step, save_root):

    if(not os.path.exists(save_root)):
        os.mkdir(save_root)

    masks = model(inputs, training=False)

    inputs = inputs.numpy()
    labels = labels.numpy()
    # index = index.numpy()
    inputs = (inputs[...] + 1) / 2
    # labels = (labels[...] + 1) / 2
    masks = masks.numpy()
    # print(inputs.shape, labels.shape, masks.shape)

    plt.figure(figsize=(9,9))
    for i in range(inputs.shape[0]):
        plt.subplot(4, 3, i*3+1)
        plt.axis('off')
        plt.imshow(inputs[i, :, :, :])
        plt.subplot(4, 3, i*3+2)
        plt.axis('off')
        plt.imshow(pic_mix(inputs[i,:,:,:], labels[i,:,:,:]))
        plt.subplot(4, 3, i*3+3)
        plt.axis('off')
        plt.imshow(pic_mix(inputs[i,:,:,:], masks[i,:,:,:], train=True))


    plt.savefig('{}/image_at_step_{}.png'.format(save_root, step))
    # plt.show()
    plt.close()

def get_class(images):
    out_mask = np.zeros(shape=images.shape[:-1])
