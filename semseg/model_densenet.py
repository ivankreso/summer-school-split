# use_dropout = True
use_dropout = False
drop_rate = 0.2


def BNReluConv(x, num_maps, k=3):
  # net = tf.contrib.layers.batch_norm(net, **bn_params)
  x = tf.layers.batch_normalization(x, **bn_params)    
  x = tf.nn.relu(x)
  x = tf.layers.conv2d(x, num_maps, k, use_bias=False,
    kernel_regularizer=reg_func, padding='same')
  return x

def upsample_dn(x, skip, num_maps):
  skip_size = skip.get_shape().as_list()[1:3]
  top_maps = x.get_shape().as_list()[3]
  skip = BNReluConv(skip, top_maps, k=1)
  x = tf.image.resize_bilinear(x, skip_size)
  print(x, skip)
  x = tf.concat([x, skip], 3)
  return BNReluConv(x, num_maps)

def layer(net, num_filters, name):
  with tf.variable_scope(name):
    net = BNReluConv(net, 4*num_filters, k=1)
    net = BNReluConv(net, num_filters, k=3)
    if use_dropout: 
      net = tf.layers.dropout(net, rate=drop_rate, training=is_training)
  return net

def dense_block(net, size, growth, name):
  with tf.variable_scope(name):
    for i in range(size):
      x = net
      net = layer(net, growth, 'layer'+str(i))
      net = tf.concat([x, net], 3)
  return net


def build_model_densenet(x):
  # input_size = tf.shape(x)[height_dim:height_dim+2]
  input_size = x.get_shape().as_list()[1:3]
  # blocks = [4, 6, 12, 8]
  # blocks = [2, 4, 8, 6]
  blocks = [4, 8, 12]
  #block_sizes = [6,12,24,16]

  # maps = [64, 128, 256, 256]
  skip_layers = []
  # x = conv(x, maps[0], k=5)
  x = tf.layers.conv2d(x, 64, 5, padding='same')
  # x = conv(x, maps[0])
  # skip_layers.append(x)
  x = pool(x)
  for i, size in enumerate(blocks):
    x = dense_block(x, size, 32, 'block'+str(i))
    if i < len(blocks) - 1:
      skip_layers.append(x)
      x = pool(x)
  print('Before :', x)

                      # x = BNReluConv(x, 128, k=1)
  # 36 without
  # for i, skip in reversed(list(enumerate(skip_layers))):
  #   print(i, x, '\n', skip)
  #   x = upsample_dn(x, skip, 128)

  logits = tf.layers.conv2d(tf.nn.relu(x), num_classes, 1, kernel_regularizer=reg_func, padding='same')
  logits = tf.image.resize_bilinear(logits, input_size, name='upsample_logits')
  return logits

