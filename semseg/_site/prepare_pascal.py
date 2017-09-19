import sys
import os
import pickle
from os.path import join

import numpy as np
from tqdm import trange
import PIL.Image as pimg


class_info = [[128,64,128,   'background'],
              [244,35,232,   'aeroplane'],
              [70,70,70,     'bicycle'],
              [102,102,156,  'bird'],
              [190,153,153,  'boat'],
              [153,153,153,  'bottle'],
              [250,170,30,   'bus'],
              [220,220,0,    'car'],
              [107,142,35,   'cat'],
              [152,251,152,  'chair'],
              [70,130,180,   'cow'],
              [220,20,60,    'diningtable'],
              [255,0,0,      'dog'],
              [0,0,142,      'horse'],
              [0,0,70,       'motorbike'],
              [0,60,100,     'person'],
              [0,80,100,     'potted plant'],
              [0,0,230,      'sheep'],
              [0,0,230,      'sofa'],
              [0,0,230,      'train'],
              [119,11,32,    'monitor']]


def add_padding(img, val, target_size):
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
  return pimg.open(path)

def resize(img, size, interpolation=pimg.BICUBIC):
  w, h = img.size
  if w > h:
    new_h = round(size * (h / w))
    new_size = (size, new_h)
  elif h > w:
    new_w = round(size * (w / h))
    new_size = (new_w, size)
  else:
    new_size = (size, size)
  return np.array(img.resize(new_size, interpolation))

def save(img, path):
  img = pimg.fromarray(img)
  img.save(path)

def remap_labels(labels, id_map, ignore_id):
  ids = np.lib.arraysetops.unique(labels)
  for id in ids:
    if id in id_map:
      labels[labels==id] = id_map[id]
    else:
      labels[labels==id] = ignore_id      
  return labels


def prepare_dataset(split_name):
  classes = [['background', 'person', 'car', 'cat', 'dog', 'horse'],
             [0, 15, 7, 8, 12, 13]]
  num_classes = len(classes[0])
  id_map = {-1: num_classes}
  for i, cid in enumerate(classes[1]):
    id_map[cid] = i

  data_dir = '/home/kivan/datasets/VOC2012/'
  save_dir = '/home/kivan/datasets/SSDS/'
  img_size = 128
  # img_size = 256

  imglist = list(map(str.strip, open(join(data_dir, split_name+'.txt')).readlines()))
  img_dir = join(data_dir, 'JPEGImages')
  labels_dir = join(data_dir, 'SegmentationClass')
  # imglist = next(os.walk(labels_dir))[2]
  # imglist = [x[:-4] for x in imglist]
  print('Orig num of images: ', len(imglist))
  os.makedirs(save_dir, exist_ok=True)
  os.makedirs(join(save_dir, 'rgb'), exist_ok=True)
  mean_sum = np.zeros(3, dtype=np.float64)
  std_sum = np.zeros(3, dtype=np.float64)
  all_images = []
  all_labels = []
  for i in trange(len(imglist)):
    img_name = imglist[i]
    labels_path = join(labels_dir, img_name + '.png')
    labels = imread(labels_path)
    labels = resize(labels, img_size, pimg.NEAREST)
    
    # labels = np.array(labels.resize(img_size, pimg.NEAREST))
    
  # return np.array(img.resize(img_size, interpolation))
    
    labels = labels.astype(np.int8)
    class_ids = np.lib.arraysetops.unique(labels)
    class_exist = False
    for cid in class_ids:
      if cid in classes[1]:
        class_exist = True
    if not class_exist:
      continue

    labels = remap_labels(labels, id_map, num_classes)

    img_path = join(img_dir, img_name + '.jpg')
    labels_path = join(labels_dir, img_name + '.png')
    img = imread(img_path)
    # img = np.array(img.resize(img_size, pimg.LANCZOS))
    img = resize(img, img_size, pimg.BICUBIC)
    
    assert img.shape[2] == 3

    # compute mean
    mean_sum += img.mean((0,1))
    std_sum += img.std((0,1))

    #print(np.unique(labels))
    #collect_hist(labels+1, class_hist)
    #print(class_hist/ class_hist.sum())
    #hist, _ = np.histogram(class_hist, bins=256)
    #plt.hist(hist, 22)  # plt.hist passes it's arguments to np.histogram
    #plt.show()
    img = add_padding(img, 0, img_size)
    labels = add_padding(labels, num_classes, img_size)
    # img = resize(img, img_size, pimg.LANCZOS)
    # labels = resize(labels, img_size, pimg.NEAREST)

    # print(join(save_dir, 'rgb', img_name+'.png'))
    # save(img, join(save_dir, 'rgb', img_name+'.png'))
    all_images.append(img)
    all_labels.append(labels)

  num_imgs = len(all_images)
  print('Num images: ', num_imgs)
  print('mean = ', mean_sum / num_imgs)
  print('std = ', std_sum / num_imgs)
  data = {}
  # all_images = np.stack(all_images)
  all_images = np.stack(all_images)
  all_labels = np.stack(all_labels)
  # print(all_images.shape)
  data['rgb'] = all_images
  data['labels'] = all_labels
  pickle.dump(data, open(join(save_dir, split_name+'.pickle'), 'wb'))

  # pickle.dump()

if __name__ == '__main__':
  prepare_dataset('train')
  prepare_dataset('val')