import tensorflow as tf

# prepare the dataset loader
from caltechfaces.loader import get_loader
maxhw = 160 # resize images larger than 160x160 pixels to that size
load_batch = get_loader(maxhw)

# `x` is a batch of input images (each of size (maxhw, maxhw) pixels with 3 channels)
x = tf.placeholder(tf.float32, [None, maxhw, maxhw, 3])
# `y` is a batch of ground-truth per-pixel labels (same size as `x`)
y = tf.placeholder(tf.float32, [None, maxhw, maxhw, 3])

from models import *
# pred, is_training = make_tinysegnet(x, 3) # `x` is input, `3` specifies the number of output channels
pred, is_training = build_model(x, 3) # `x` is input, `3` specifies the number of output channels

# prepare the loss function
thr = 0.1
loss = tf.nn.relu( 0.5*(y - pred)**2 - 0.5*thr**2 )
# print(loss)
# loss = tf.reduce_mean(loss)
loss = tf.reduce_sum(tf.reduce_mean(loss, axis=[1,2,3]))
# print(loss)
# raise 1

# we will use RMSprop to learn the model
# step = tf.train.RMSPropOptimizer(1e-4).minimize(loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	# step = tf.train.RMSPropOptimizer(1e-4).minimize(loss)
	step = tf.train.AdamOptimizer(5e-4).minimize(loss)

#
saver = tf.train.Saver()

#
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for k in range(2048):
	#
	X, Y = load_batch(32)
	# X, Y = load_batch(1)
	l, _ = sess.run([loss, step], feed_dict={x: X, y: Y, is_training: True})
	#
	# l = sess.run(loss, feed_dict={x: X, y: Y, is_training: False})
	print('* loss on batch %d: %.7f' % (k, l))
	#
	#u = sess.run(pred, feed_dict={x: X, is_training: False})
	#import cv2
	#cv2.imwrite('wat.jpg', 255*u[0, :, :, :])

#
import os
os.system('mkdir -p save')
# saver.save(sess, 'save/model')
save_path = saver.save(sess, 'save/model')
print("Model saved in file: %s" % save_path)

print('* done ...')