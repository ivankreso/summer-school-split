import tensorflow as tf
import numpy as np

from data import Dataset

# hyperparameters
num_epochs = 50
batch_size = 10

num_classes = 6
# learning_rate = 5e-4
# learning_rate = 1e-2
learning_rate = 1e-3
decay_power = 1.0

#sgd
# learning_rate = 1e-2
# decay_power = 0.9

# learning_rate = 1e-4
weight_decay = 1e-4
# use_dropout = True
use_dropout = False
drop_rate = 0.2

# need this placeholder for bach norm
is_training = tf.placeholder(tf.bool)


def conv_22(x, num_maps, k=3, activation=tf.nn.relu):
  return tf.layers.conv2d(x, num_maps, k, activation=activation, padding='same')
def conv(x, num_maps, k=3):
  x = tf.layers.conv2d(x, num_maps, k, use_bias=False,
    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), padding='same')
  x = tf.layers.batch_normalization(x, training=is_training)
  return tf.nn.relu(x)

def pool(x):
  # return tf.layers.average_pooling2d(x, 2, 2, 'same')
  return tf.layers.max_pooling2d(x, 2, 2, 'same')


def build_model1(x):
  # input_size = tf.shape(x)[height_dim:height_dim+2]
  input_size = x.get_shape().as_list()[1:3]
  print(input_size)
  x = conv(x, 32, 3)
  x = pool(x)
  x = conv(x, 64, 3)
  x = pool(x)
  x = conv(x, 128, 3)
  x = pool(x)
  x = conv(x, 128, 3)
  # x = pool(x)
  # x = conv(x, 64, 3)
  # logits = conv(x, num_classes, 3, activation=None)
  logits = tf.layers.conv2d(x, num_classes, 1, padding='same')
  
  logits = tf.image.resize_bilinear(logits, input_size, name='upsample_logits')
  return logits

def upsample(x, skip, num_maps):
  skip_size = skip.get_shape().as_list()[1:3]
  x = tf.image.resize_bilinear(x, skip_size)
  x = tf.concat([x, skip], 3)
  return conv(x, num_maps)


bn_params = {
  # Decay for the moving averages.
  'momentum': 0.9,
  # epsilon to prevent 0s in variance.
  'epsilon': 1e-5,
  # fused must be false if BN is frozen
  'fused': True,
  'training': is_training
}


reg_func = tf.contrib.layers.l2_regularizer(weight_decay)

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

def build_model(x):
  # input_size = tf.shape(x)[height_dim:height_dim+2]
  input_size = x.get_shape().as_list()[1:3]
  maps = [32, 64, 128, 256, 128]
  # maps = [64, 128, 256, 256]
  skip_layers = []
  x = conv(x, maps[0], k=5)
  # x = conv(x, maps[0])
  # skip_layers.append(x)
  x = pool(x)
  x = conv(x, maps[1])
  x = conv(x, maps[1])
  # skip_layers.append(x)
  x = pool(x)
  x = conv(x, maps[2])
  x = conv(x, maps[2])
  # skip_layers.append(x)
  x = pool(x)
  x = conv(x, maps[3])
  x = conv(x, maps[3])
  skip_layers.append(x)
  x = pool(x)
  x = conv(x, maps[4])
  x = conv(x, maps[4])
  # x = conv(x, maps[3])
  # skip_layers.append(x)  
  # x = pool(x)
  # x = conv(x, maps[4])

  # 36 without
  for i, skip in reversed(list(enumerate(skip_layers))):
    print(i, x, '\n', skip)
    x = upsample(x, skip, maps[i])

  # x = pool(x)
  # x = conv(x, 64, 3)
  # logits = conv(x, num_classes, 3, activation=None)
  logits = tf.layers.conv2d(x, num_classes, 1, padding='same')
  
  logits = tf.image.resize_bilinear(logits, input_size, name='upsample_logits')
  return logits

def add_regularization(loss):
  regularization_losses = tf.losses.get_regularization_losses()
  print(regularization_losses)
  #total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
  return tf.add_n([loss] + regularization_losses, name='total_loss')


def build_loss(logits, labels):
  print('loss: cross-entropy')
  labels = tf.reshape(labels, shape=[-1])
  logits = tf.reshape(logits, [-1, num_classes])
  mask = labels < num_classes
  idx = tf.where(mask)
  labels = tf.to_float(labels)
  labels = tf.gather_nd(labels, idx)
  # labels = tf.boolean_mask(labels, mask)
  labels = tf.to_int32(labels)
  logits = tf.gather_nd(logits, idx)
  # logits = tf.boolean_mask(logits, mask)
  
  onehot_labels = tf.one_hot(labels, num_classes)
  xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels)

  weights = [4] * num_classes
  # weights[0] = 0.1
  weights[0] = 1
  class_weights = tf.constant(weights, dtype=tf.float32)
  weights = tf.gather(class_weights, labels)
  xent = tf.multiply(weights, xent)

  xent = tf.reduce_mean(xent)
  loss = add_regularization(xent)
  return loss, logits, labels


def compute_metrics(conf_mat, name):
  num_correct = conf_mat.trace()
  num_classes = conf_mat.shape[0]
  total_size = conf_mat.sum()
  avg_pixel_acc = num_correct / total_size * 100.0
  TPFP = conf_mat.sum(0)
  TPFN = conf_mat.sum(1)
  FN = TPFN - conf_mat.diagonal()
  FP = TPFP - conf_mat.diagonal()
  class_iou = np.zeros(num_classes)
  # class_recall = np.zeros(num_classes)
  # class_precision = np.zeros(num_classes)
  print('\n', name, ' evaluation metrics:')
  for i in range(num_classes):
    TP = conf_mat[i,i]
    class_iou[i] = (TP / (TP + FP[i] + FN[i])) * 100.0
    class_name = Dataset.class_info[0][i]
    print('\t%s IoU accuracy = %.2f %%' % (class_name, class_iou[i]))
  avg_class_iou = class_iou.mean()
  print(name + ' IoU mean class accuracy - TP / (TP+FN+FP) = %.2f %%' % avg_class_iou)
  print(name + ' pixel accuracy = %.2f %%' % avg_pixel_acc)
  return avg_class_iou


def validate(data, x, y, loss, conf_mat):
  confusion_mat = np.zeros((num_classes, num_classes), dtype=np.uint64) 
  for i, (batch_x, batch_y) in enumerate(data):
    batch_loss, batch_conf_mat = sess.run([loss, conf_mat],
      feed_dict={x: batch_x, y: batch_y, is_training: False})
    confusion_mat += batch_conf_mat.astype(np.uint64)
    # y_pred = logits_val.argmax(3)
    # add_confusion_matrix(batch_y.reshape(-1), y_pred.reshape(-1), confusion_mat)
    if i % 50 == 0:
      string = 'epoch %d / %d loss = %.2f' % (epoch+1, num_epochs, batch_loss)
      print(string)
  print(confusion_mat)
  return compute_metrics(confusion_mat, 'Validation')


train_data = Dataset('train', batch_size)
val_data = Dataset('val', batch_size, shuffle=False)

height = train_data.height
width = train_data.width
channels = train_data.channels
# x = tf.placeholder(tf.float32, shape=(batch_size, height, width, channels))
# y = tf.placeholder(tf.int32, shape=(batch_size, height, width))
x = tf.placeholder(tf.float32, shape=(None, height, width, channels))
y = tf.placeholder(tf.int32, shape=(None, height, width))

logits = build_model(x)
loss, logits_vec, y_vec = build_loss(logits, y)

# build ops for confusion matrix
y_pred = tf.argmax(logits_vec, axis=1, output_type=tf.int32)
print(y, y_pred)
conf_mat = tf.confusion_matrix(y_vec, y_pred, num_classes)

global_step = tf.Variable(0, trainable=False)
decay_steps = num_epochs * train_data.num_batches

lr = tf.train.polynomial_decay(learning_rate, global_step, decay_steps,
                               end_learning_rate=0, power=decay_power)
# opt = tf.train.AdamOptimizer(lr)
# grads = opt.compute_gradients(loss)
# train_op = opt.apply_gradients(grads, global_step=global_step)
# train_step = opt.apply_gradients(grads)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
  # train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
  train_step = tf.train.MomentumOptimizer(lr, momentum=0.9).minimize(loss, global_step=global_step)
# loss = tf.Print(loss, [lr, global_step])

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

step = 0
best_iou = 0
for epoch in range(num_epochs):
  # for i in range(num_batches-1):
  confusion_mat = np.zeros((num_classes, num_classes), dtype=np.uint64)  
  for batch_x, batch_y in train_data:
    batch_loss, batch_conf_mat, _ = sess.run([loss, conf_mat, train_step],
      feed_dict={x: batch_x, y: batch_y, is_training: True})
    
    confusion_mat += batch_conf_mat.astype(np.uint64)
    if step % 30 == 0:
      string = 'epoch %d / %d loss = %.2f' % (epoch+1, num_epochs, batch_loss)
      print(string)
    step += 1
  compute_metrics(confusion_mat, 'Train') 
  iou = validate(val_data, x, y, loss, conf_mat)
  if iou > best_iou:
    best_iou = iou
  print('\nBest IoU = %.2f\n' % best_iou)