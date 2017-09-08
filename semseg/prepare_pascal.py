import sys
import os
import pickle
from os.path import join

import numpy as np
from tqdm import trange
import PIL.Image as pimg


def add_padding(img, val, target_size=500):
  #np.all(a == b, axis=tuple(range(-b.ndim, 0)))
  height = img.shape[0]
  width = img.shape[1]
  #assert height == target_size or width == target_size
  new_shape = list(img.shape)
  new_shape[0] = target_size
  new_shape[1] = target_size
  padded_img = np.ndarray(new_shape, dtype=img.dtype)
  padded_img.fill(val)
  #print(padded_img.dtype)
  sh = round((target_size - height) / 2)
  eh = sh + height
  sw = round((target_size - width) / 2)
  ew = sw + width
  padded_img[sh:eh,sw:ew,...] = img
  #plt.imshow(padded_img.astype(np.uint8))
  #plt.show()
  return padded_img

def imread(path):
  img = pimg.open(path)
  #img = img.convert('RGB')
  #print('bands = ', img.getbands())
  return np.array(img)

def prepare_dataset():
  img_dir = join(FLAGS.data_dir, 'JPEGImages')
  labels_dir = join(FLAGS.data_dir, 'SegmentationClass')
  imglist = next(os.walk(labels_dir))[2]
  imglist = [x[:-4] for x in imglist]
  print('Num of images: ', len(imglist))
  #print(imglist)
  save_dir = FLAGS.save_dir
  print(save_dir)
  os.makedirs(save_dir, exist_ok=True)
  #mean_sum = np.zeros(3, dtype=np.float64)
  #std_sum = np.zeros(3, dtype=np.float64)
  class_hist = np.zeros(22)
  for i in trange(len(imglist)):
    img_name = imglist[i]
    img_path = join(img_dir, img_name + '.jpg')
    labels_path = join(labels_dir, img_name + '.png')
    #img = ski.io.imread(img_path)
    img = imread(img_path)
    #img = cv2.imread(img_path, cv2.IMREAD_COLOR) BGR!

    # compute mean
    #mean_sum += img.mean((0,1))
    #std_sum += img.std((0,1))
    #print('mean = ', mean_sum / (i+1))
    #print('std = ', std_sum / (i+1))

    labels_path = join(labels_dir, img_name + '.png')
    #labels = ski.data.load(labels_path)
    #labels = ski.io.imread(labels_path)
    labels = imread(labels_path)
    #print(np.unique(labels))
    labels = labels.astype(np.int8)
    #print(np.unique(labels))
    #collect_hist(labels+1, class_hist)
    #print(class_hist/ class_hist.sum())
    #hist, _ = np.histogram(class_hist, bins=256)
    #plt.hist(hist, 22)  # plt.hist passes it's arguments to np.histogram
    #plt.show()
    img = add_padding(img, 0)
    labels = add_padding(labels, -1)
    #label_mask = labels >= 0
    #num_labels = np.sum(label_mask)
    #weights = label_mask.astype(np.float32)
    weights, num_labels = get_label_weights(labels, NUM_CLASSES)
    create_record(img, labels, weights, num_labels, img_name, save_dir)
  #print(class_hist/ class_hist.sum())


def main(argv):
  prepare_dataset()


if __name__ == '__main__':
  tf.app.run()
