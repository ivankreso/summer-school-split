import math
import pickle
from os.path import join
import tensorflow as tf


# hyperparameters
num_epochs = 50
batch_size = 10
mean = [116.9, 110.2, 100.6] 
std = [62.6, 61.4, 62.1]
num_classes = 5

def conv(x, num_maps, k, activation=tf.nn.relu):
  return tf.layers.conv2d(x, num_maps, k, activation=activation, padding='same')

def pool(x):
  return tf.layers.average_pooling2d(x, 2, 2, 'same')


def build_model(x):
  # input_size = tf.shape(x)[height_dim:height_dim+2]
  input_size = x.get_shape().as_list()[1:3]
  print(input_size)
  x = conv(x, 32, 3)
  x = pool(x)
  x = conv(x, 64, 3)
  x = pool(x)
  x = conv(x, 64, 3)
  x = pool(x)
  x = conv(x, 64, 3)
  x = pool(x)
  x = conv(x, 64, 3)
  logits = conv(x, num_classes, 3, activation=None)
  logits = tf.image.resize_bilinear(logits, input_size, name='upsample_logits')
  return logits

def build_loss(logits, labels):
  print('loss: cross-entropy')
  num_pixels = -1
  labels = tf.reshape(labels, shape=[num_pixels])
  logits = tf.reshape(logits, [num_pixels, num_classes])
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
  return tf.reduce_mean(xent)


# load the dataset
data_dir = '/home/kivan/datasets/SSDS'
train_data = pickle.load(open(join(data_dir, 'train.pickle'), 'rb'))
val_data = pickle.load(open(join(data_dir, 'val.pickle'), 'rb'))
train_x = train_data['rgb']
train_y = train_data['labels']
val_x = val_data['rgb']
val_y = val_data['labels']

print(train_x.shape)

num_examples, height, width, channels = train_x.shape
num_batches = math.ceil(num_examples / batch_size)

x = tf.placeholder(tf.float32, shape=(batch_size, height, width, channels))
y = tf.placeholder(tf.int32, shape=(batch_size, height, width))

logits = build_model(x)
loss = build_loss(logits, y)

global_step = tf.Variable(0, trainable=False)
start_lr = 5e-4
decay_steps = num_epochs * num_batches
lr = tf.train.polynomial_decay(start_lr, global_step, decay_steps,
                               end_learning_rate=0, power=1.3)
opt = tf.train.AdamOptimizer(lr)
grads = opt.compute_gradients(loss)
# train_op = opt.apply_gradients(grads, global_step=global_step)
train_step = opt.apply_gradients(grads)

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

for epoch in range(num_epochs):
  for i in range(num_batches-1):
    offset = i * batch_size
    batch_x = train_x[offset:offset+batch_size]
    batch_y = train_y[offset:offset+batch_size]
    loss_val, _ = sess.run([loss, train_step], feed_dict={x: batch_x, y: batch_y})
    if i % 40 == 0:
      string = '%d / %d loss = %.2f' % (epoch+1, num_epochs, loss_val)
      print(string)
