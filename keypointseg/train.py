import tensorflow as tf

# prepare the dataset loader
from caltechfaces.loader import get_loader
maxhw = 288 # resize images larger than 288x288 pixels to that size
load_batch = get_loader(maxhw)

# `x` is a batch of input images (each of size 288x288 pixels with 3 channels)
x = tf.placeholder(tf.float32, [None, maxhw, maxhw, 3])
# `y` is a batch of ground-truth per-pixel labels (same size as `x`)
y = tf.placeholder(tf.float32, [None, maxhw, maxhw, 3])

from models import *
pred = make_convnet(x, 3) # `x` is input, `3` specifies the number of output channels

# prepare the loss function
thr = 0.1
loss = tf.reduce_mean( tf.nn.relu( 0.5*(y - pred)**2 - 0.5*thr**2 ) )

# we will use RMSprop to learn the model
step = tf.train.RMSPropOptimizer(1e-4).minimize(loss)

#'''
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for k in range(2048):
	#
	X, Y = load_batch(1)
	#
	sess.run(step, feed_dict={x: X, y: Y, is_training: True})
	if True:
		l = sess.run(loss, feed_dict={x: X, y: Y, is_training: False})
		print('* loss on batch %d: %.7f' % (k, l))

print('* done ...')
#'''