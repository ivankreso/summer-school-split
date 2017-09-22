import tensorflow as tf


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
  x = tf.layers.conv2d(x, num_maps, k, use_bias=False,
    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4), padding='same')
  x = tf.layers.batch_normalization(x, training=is_training)
  return tf.nn.relu(x)

def pool(x):
  return tf.layers.max_pooling2d(x, 2, 2, 'same')

def upsample(x, skip, num_maps):
  skip_size = skip.get_shape().as_list()[1:3]
  x = tf.image.resize_bilinear(x, skip_size)
  x = tf.concat([x, skip], 3)
  return conv(x, num_maps)


def block(x, size, name):
  with tf.name_scope(name):
    x = conv(x, size)
  return x

def build_model(x, num_classes):
  # input_size = tf.shape(x)[height_dim:height_dim+2]
  print(x)
  input_size = x.get_shape().as_list()[1:3]
  block_sizes = [64, 64, 64, 64]
  skip_layers = []
  x = conv(x, 32, k=5)
  for i, size in enumerate(block_sizes):
    x = pool(x)
    x = conv(x, size)
    x = conv(x, size)
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
  return tf.nn.sigmoid(x), is_training



def make_convnet(x, numoutmaps):
	# 1
	s1 = x.get_shape().as_list()[1:3]
	x = conv(x, 24)
	x = conv(x, 24)
	x = tf.layers.max_pooling2d(x, 2, 2, 'same')
	# 2
	s2 = x.get_shape().as_list()[1:3]
	x = conv(x, 48)
	x = conv(x, 48)
	x = tf.layers.max_pooling2d(x, 2, 2, 'same')
	# 3
	s3 = x.get_shape().as_list()[1:3]
	x = conv(x, 64)
	x = conv(x, 64)
	x = tf.layers.max_pooling2d(x, 2, 2, 'same')
	#
	x = conv(x, 128)
	#
	x = tf.image.resize_bilinear(x, s3)
	x = conv(x, 64)
	x = conv(x, 64)
	#
	x = tf.image.resize_bilinear(x, s2)
	x = conv(x, 48)
	x = conv(x, 48)
	#
	x = tf.image.resize_bilinear(x, s1)
	x = conv(x, 24)
	x = conv(x, 24)
	#
	x = tf.layers.conv2d(x, numoutmaps, 1, padding='same')
	return tf.nn.sigmoid(x, name='pred')
#

bn_params = {
  # Decay for the moving averages.
  'momentum': 0.9,
  # epsilon to prevent 0s in variance.
  'epsilon': 1e-5,
  # fused must be false if BN is frozen
  # 'fused': True,
  'training': is_training
}
def get_convblock(x, nmaps, relu=True):
	#
	x = tf.layers.conv2d(x, nmaps, 3, padding='same')
	# x = tf.layers.batch_normalization(x, training=is_training)
	# x = tf.layers.batch_normalization(x, **bn_params)
	if relu:
		x = tf.nn.relu(x)
	return x


def make_tinysegnet(x, numoutmaps):
	#
	S = 24
	sizes = []
	#
	sizes.append( x.get_shape().as_list()[1:3] )
	x = get_convblock(x, 1*S)
	x = get_convblock(x, 1*S)
	x = tf.layers.max_pooling2d(x, 3, 2, 'same')
	sizes.append( x.get_shape().as_list()[1:3] )
	x = get_convblock(x, 2*S)
	x = get_convblock(x, 2*S)
	x = tf.layers.max_pooling2d(x, 3, 2, 'same')
	sizes.append( x.get_shape().as_list()[1:3] )
	x = get_convblock(x, 3*S)
	x = get_convblock(x, 3*S)
	x = tf.layers.max_pooling2d(x, 3, 2, 'same')
	sizes.append( x.get_shape().as_list()[1:3] )
	x = get_convblock(x, 4*S)
	x = get_convblock(x, 4*S)
	x = tf.layers.max_pooling2d(x, 3, 2, 'same')
	#
	x = tf.layers.conv2d(x, 4*S, 3, padding='same')
	#
	x = tf.image.resize_bilinear(x, sizes[3])
	x = get_convblock(x, 3*S)
	x = get_convblock(x, 3*S)
	#
	x = tf.image.resize_bilinear(x, sizes[2])
	x = get_convblock(x, 2*S)
	x = get_convblock(x, 2*S)
	#
	x = tf.image.resize_bilinear(x, sizes[1])
	x = get_convblock(x, 1*S)
	x = get_convblock(x, 1*S)
	#
	x = tf.image.resize_bilinear(x, sizes[0])
	x = get_convblock(x, 1*S)
	x = get_convblock(x, numoutmaps, relu=False)
	#
	return tf.nn.sigmoid(x, name='pred'), is_training