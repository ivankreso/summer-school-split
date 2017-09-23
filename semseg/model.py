import tensorflow as tf
import numpy as np

# weight_decay = 1e-4
weight_decay = 0.0
reg_func = tf.contrib.layers.l2_regularizer(weight_decay)

# need this placeholder for bach norm
is_training = tf.placeholder(tf.bool)
bn_params = {
  # Decay for the moving averages.
  'momentum': 0.9,
  # epsilon to prevent 0s in variance.
  'epsilon': 1e-5,
  # fused must be false if BN is frozen
  'fused': True,
  'training': is_training
}

def conv(x, num_maps, k=3):
  # x = tf.layers.conv2d(x, num_maps, k,
  #   kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), padding='same')

  x = tf.layers.conv2d(x, num_maps, k, padding='same')
  
  # only 1.5 % better
  # x = tf.layers.conv2d(x, num_maps, k, use_bias=False, padding='same')
  # x = tf.layers.batch_normalization(x, training=is_training)
  x = tf.nn.relu(x)
  return x

def pool(x):
  # return tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='same')
  # return tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='same')
  return tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='same')

def upsample(x, skip, num_maps):
  skip_size = skip.get_shape().as_list()[1:3]
  x = tf.image.resize_bilinear(x, skip_size)
  x = tf.concat([x, skip], 3)
  return conv(x, num_maps)


def block(x, size, name):
  with tf.name_scope(name):
    x = conv(x, size)
  return x

def build_model_adv(x, num_classes):
  # input_size = tf.shape(x)[height_dim:height_dim+2]
  print(x)
  input_size = x.get_shape().as_list()[1:3]
  block_sizes = [64, 64, 64, 64]
  skip_layers = []
  x = conv(x, 32, k=3)
  for i, size in enumerate(block_sizes):
    x = pool(x)
    x = conv(x, size)
    # x = conv(x, size)
    skip_layers.append(x)

  # # 36 without
  for i, skip in reversed(list(enumerate(skip_layers))):
    print(i, x, '\n', skip)
    x = upsample(x, skip, block_sizes[i])
  print('final: ', x)
  # x = pool(x)
  # x = conv(x, 64, 3)
  # logits = conv(x, num_classes, 3, activation=None)
  x = tf.layers.conv2d(x, num_classes, 1, padding='same')
  x = tf.image.resize_bilinear(x, input_size, name='upsample_logits')
  return x, is_training

def build_model(x, num_classes):
  # input_size = tf.shape(x)[height_dim:height_dim+2]
  print(x)
  input_size = x.get_shape().as_list()[1:3]
  block_sizes = [64, 64, 64, 64]
  # block_sizes = [64, 96, 128, 128]
  x = conv(x, 32, k=3)
  for i, size in enumerate(block_sizes):
    with tf.name_scope('block'+str(i)):
      x = pool(x)
      x = conv(x, size)
      # x = conv(x, size)
  print(x)
  with tf.name_scope('logits'):
    x = tf.layers.conv2d(x, num_classes, 1, padding='same')
    x = tf.image.resize_bilinear(x, input_size, name='upsample_logits')
  return x, is_training

# def conv_22(x, num_maps, k=3, activation=tf.nn.relu):
#   return tf.layers.conv2d(x, num_maps, k, activation=activation, padding='same')
# def conv(x, num_maps, k=3):
#   # x = tf.layers.conv2d(x, num_maps, k,
#   #   kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), padding='same')

#   x = tf.layers.conv2d(x, num_maps, k, use_bias=False,
#     kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), padding='same')
#   x = tf.layers.batch_normalization(x, training=is_training)
#   return tf.nn.relu(x)

# def pool(x):
#   return tf.layers.average_pooling2d(x, 2, 2, 'same')
#   # return tf.layers.max_pooling2d(x, 2, 2, 'same')


# def build_model1(x):
#   # input_size = tf.shape(x)[height_dim:height_dim+2]
#   input_size = x.get_shape().as_list()[1:3]
#   print(input_size)
#   x = conv(x, 32, 3)
#   x = pool(x)
#   x = conv(x, 64, 3)
#   x = pool(x)
#   x = conv(x, 128, 3)
#   x = pool(x)
#   x = conv(x, 128, 3)
#   # x = pool(x)
#   # x = conv(x, 64, 3)
#   # logits = conv(x, num_classes, 3, activation=None)
#   logits = tf.layers.conv2d(x, num_classes, 1, padding='same')
  
#   logits = tf.image.resize_bilinear(logits, input_size, name='upsample_logits')
#   return logits

# def upsample(x, skip, num_maps):
#   skip_size = skip.get_shape().as_list()[1:3]
#   x = tf.image.resize_bilinear(x, skip_size)
#   x = tf.concat([x, skip], 3)
#   return conv(x, num_maps)

# # best 78
# def build_model(x, num_classes):
#   # input_size = tf.shape(x)[height_dim:height_dim+2]
#   print(x)
#   input_size = x.get_shape().as_list()[1:3]
#   # maps = [32, 64, 128, 256, 128]
  
#   # maps = [64, 128, 128, 128]
#   maps = [64, 64, 64, 64]
#   # maps = [32, 64, 64, 64]
#   # maps = [32, 64, 64, 64, 64]
#   # maps = [64, 128, 256, 256]
#   skip_layers = []
#   x = conv(x, 32, k=5)
#   # x = conv(x, 32, k=3)
#   # x = conv(x, maps[0])
#   # skip_layers.append(x)
#   x = pool(x)
#   x = conv(x, maps[0])
#   # x = conv(x, maps[0])
#   skip_layers.append(x)
#   x = pool(x)
#   x = conv(x, maps[1])
#   # x = conv(x, maps[1])
#   skip_layers.append(x)
#   x = pool(x)
#   x = conv(x, maps[2])
#   # x = conv(x, maps[2])
#   skip_layers.append(x)
#   x = pool(x)
#   x = conv(x, maps[3])
#   # x = conv(x, maps[3])

#   # skip_layers.append(x)  
#   # x = pool(x)
#   # x = conv(x, maps[4])
  
#   # # 36 without
#   for i, skip in reversed(list(enumerate(skip_layers))):
#     print(i, x, '\n', skip)
#     x = upsample(x, skip, maps[i])
#   print('final: ', x)
#   # x = pool(x)
#   # x = conv(x, 64, 3)
#   # logits = conv(x, num_classes, 3, activation=None)
#   logits = tf.layers.conv2d(x, num_classes, 1, padding='same')
  
#   logits = tf.image.resize_bilinear(logits, input_size, name='upsample_logits')
#   return logits, is_training