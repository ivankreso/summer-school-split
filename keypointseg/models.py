import tensorflow as tf

is_training = tf.placeholder(tf.bool)

def conv(x, num_maps, k=3):
	x = tf.layers.conv2d(x, num_maps, k, use_bias=False, padding='same')
	x = tf.layers.batch_normalization(x, training=is_training)
	return tf.nn.relu(x)

def upsample(x, skip, num_maps):
	skip_size = skip.get_shape().as_list()[1:3]
	x = tf.image.resize_bilinear(x, skip_size)
	x = tf.concat([x, skip], 3)
	return conv(x, num_maps)

def make_convnet_(x, numoutmaps):
	input_size = x.get_shape().as_list()[1:3]
	maps = [32, 64, 128, 128]
	skip_layers = []
	x = conv(x, maps[0])
	x = tf.layers.average_pooling2d(x, 2, 2, 'same')
	x = conv(x, maps[1])
	x = conv(x, maps[1])
	skip_layers.append(x)
	x = tf.layers.average_pooling2d(x, 2, 2, 'same')
	x = conv(x, maps[2])
	x = conv(x, maps[2])
	skip_layers.append(x)
	x = tf.layers.average_pooling2d(x, 2, 2, 'same')
	x = conv(x, maps[3])
	x = conv(x, maps[3])
	skip_layers.append(x)
	x = tf.layers.average_pooling2d(x, 2, 2, 'same')
	x = conv(x, maps[3])

	for i, skip in reversed(list(enumerate(skip_layers))):
		print(i, x, '\n', skip)
		x = upsample(x, skip, maps[i])

	x = tf.layers.conv2d(x, numoutmaps, 1, padding='same')
	x = tf.image.resize_bilinear(x, input_size, name='upsample_logits')
	return tf.nn.sigmoid(x)

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
	return tf.nn.sigmoid(x)