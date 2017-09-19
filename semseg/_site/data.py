import math
import numpy as np
import pickle
from os.path import join

def _shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

class Dataset():
  class_info = [['background', 'person', 'car', 'cat', 'dog', 'horse'],
                []]

  def __init__(self, split_name, batch_size, shuffle=True):
    self.mean = np.array([116.5, 112.1, 104.1])
    self.std = np.array([57.92, 56.97, 58.54])
    self.batch_size = batch_size
    self.shuffle = shuffle
    # load the dataset
    data_dir = '/home/kivan/datasets/SSDS'
    data = pickle.load(open(join(data_dir, split_name+'.pickle'), 'rb'))
    self.x = data['rgb']
    self.y = data['labels']
    self.x = (self.x.astype(np.float32) - self.mean) / self.std

    self.num_examples = self.x.shape[0]
    self.height = self.x.shape[1]
    self.width = self.x.shape[2]
    self.channels = self.x.shape[3]
    self.num_batches = math.ceil(self.num_examples / self.batch_size)

  def __iter__(self):
    if self.shuffle:
      self.x, self.y = _shuffle_data(self.x, self.y)
    self.cnt = 0
    return self

  def __next__(self):
    if self.cnt >= self.num_batches:
      raise StopIteration
    offset = self.cnt * self.batch_size
    x = self.x[offset:offset+self.batch_size]
    y = self.y[offset:offset+self.batch_size]
    self.cnt += 1
    return x, y
  