import time
from os.path import join

import tensorflow as tf
import numpy as np

import utils
import libs.cylib as cylib
import model
from data import Dataset

# hyperparameters
num_epochs = 50
batch_size = 10
num_classes = Dataset.num_classes
save_dir = 'assets/color_outputs/'
learning_rate = 5e-4
# learning_rate = 1e-2
# learning_rate = 1e-3
decay_power = 1.2

#sgd
# learning_rate = 1e-2
# decay_power = 0.9

# learning_rate = 1e-4



def add_regularization(loss):
  regularization_losses = tf.losses.get_regularization_losses()
  print(regularization_losses)
  #total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
  return tf.add_n([loss] + regularization_losses, name='total_loss')


def build_loss(logits, y):
  print('loss: cross-entropy')
  y = tf.reshape(y, shape=[-1])
  logits = tf.reshape(logits, [-1, num_classes])
  mask = y < num_classes
  idx = tf.where(mask)
  y = tf.to_float(y)
  y = tf.gather_nd(y, idx)
  # labels = tf.boolean_mask(labels, mask)
  y = tf.to_int32(y)
  logits = tf.gather_nd(logits, idx)
  # logits = tf.boolean_mask(logits, mask)
  
  y_one_hot = tf.one_hot(y, num_classes)
  xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)

  # weights = [4] * num_classes
  # # weights[0] = 0.1
  # weights[0] = 1
  # class_weights = tf.constant(weights, dtype=tf.float32)
  # weights = tf.gather(class_weights, labels)
  # xent = tf.multiply(weights, xent)

  xent = tf.reduce_mean(xent)
  tf.summary.scalar('cross_entropy', xent)
  loss = add_regularization(xent)
  return loss, logits, y


def validate(data, x, y, y_pred, loss):
  print('\nValidation phase:')
  conf_mat = np.zeros((num_classes, num_classes), dtype=np.uint64) 
  for i, (x_np, y_np) in enumerate(data):
    start_time = time.time()    
    # loss_np, y_pred_np, conf_mat_np = sess.run([loss, y_pred, conf_mat],
    #   feed_dict={x: x_np, y: y_np, is_training: False})
    loss_np, y_pred_np = sess.run([loss, y_pred],
      feed_dict={x: x_np, y: y_np, is_training: False})

    duration = time.time() - start_time

    # save_path = join(save_dir, '%03d'%i + '.png')
    # utils.draw_output(y_pred_np[0]fix, Dataset.class_info, save_path=save_path) 

    # net_labels = logits.argmax(3).astype(np.int32)
    #gt_labels = gt_labels.astype(np.int32, copy=False)
    cylib.collect_confusion_matrix(y_pred_np.reshape(-1),
                                   y_np.reshape(-1), conf_mat)
    # conf_mat_all += conf_mat_np.astype(np.uint64)
    if i % 10 == 0:
      string = 'epoch %d / %d loss = %.2f (%.1f examples/sec)' % \
        (epoch+1, num_epochs, loss_np, x_np.shape[0] / duration)
      print(string)
  print(conf_mat)
  return utils.print_metrics(conf_mat, 'Validation', Dataset.class_info)


train_data = Dataset('train', batch_size)
val_data = Dataset('val', batch_size, shuffle=False)

height = train_data.height
width = train_data.width
channels = train_data.channels


# x = tf.placeholder(tf.float32, shape=(batch_size, height, width, channels))
# y = tf.placeholder(tf.int32, shape=(batch_size, height, width))

# create placeholders for inputs
x = tf.placeholder(tf.float32, shape=(None, height, width, channels), name='rgb_images')
y = tf.placeholder(tf.int32, shape=(None, height, width), name='labels')

logits, is_training = model.build_model(x, num_classes)
loss, logits_vec, y_vec = build_loss(logits, y)

# build ops for confusion matrix
# y_pred = tf.argmax(logits_vec, axis=1, output_type=tf.int32)
y_pred = tf.argmax(logits, axis=3, output_type=tf.int32)
# conf_mat = tf.confusion_matrix(y_vec, y_pred, num_classes)

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
  train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
  # train_step = tf.train.MomentumOptimizer(lr, momentum=0.9).minimize(loss, global_step=global_step)
# loss = tf.Print(loss, [lr, global_step])

# sess = tf.Session()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

summary_all = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('assets/logs/3/train', sess.graph)
# test_writer = tf.summary.FileWriter('assets/logs/3/test')

tf.global_variables_initializer().run(session=sess)


step = 0
best_iou = 0
exp_start_time = time.time()
for epoch in range(num_epochs):
  # for i in range(num_batches-1):
  # confusion_mat = np.zeros((num_classes, num_classes), dtype=np.uint64)
  print('\nTraining phase:')  
  for batch_x, batch_y in train_data:
    start_time = time.time()
    # batch_loss, batch_conf_mat, _ = sess.run([loss, conf_mat, train_step],
    # batch_loss, _ = sess.run([loss, train_step],
    #   feed_dict={x: batch_x, y: batch_y, is_training: True})
    batch_loss, summary, _ = sess.run([loss, summary_all, train_step],
      feed_dict={x: batch_x, y: batch_y, is_training: True})
    train_writer.add_summary(summary, step)
    duration = time.time() - start_time
    # confusion_mat += batch_conf_mat.astype(np.uint64)
    if step % 20 == 0:
      string = '%s: epoch %d / %d, iter %05d, loss = %.2f (%.1f examples/sec)' % \
        (utils.get_expired_time(exp_start_time), epoch+1, num_epochs, step,
         batch_loss, batch_size / duration)
      print(string)
    step += 1
  # utils.print_metrics(confusion_mat, 'Train') 
  iou = validate(val_data, x, y, y_pred, loss)
  if iou > best_iou:
    best_iou = iou
  print('\nBest IoU = %.2f\n' % best_iou)