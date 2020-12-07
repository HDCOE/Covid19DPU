from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import os, argparse, pathlib
import numpy as np
from data import BalanceCovidDataset
from data import process_image_file
from customcnn import customcnn
from customcnn import Covid
from datadownload import datadownload
from alexnet import alexnet_v2
from lenet import lenet
from sklearn.metrics import confusion_matrix
mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
input_height = 224
input_width = 224
input_chan = 3

def Testset(testfile):
	testfolder = 'Trainset/test'
	correct_predictions = 0
	wrong_predictions = 0
	lentest = len(testfile)
	input_shape = (input_height,input_width)

	#x_test, y_test = np.zeros((lentest, *input_shape,3)), np.zeros(lentest)
	y_test = []
	x_test = np.zeros((lentest, *input_shape,3))
	for i in range(lentest):
		line = testfile[i].split()
		x = process_image_file(os.path.join(testfolder, line[1]), 0.08, 224)
		x = x.astype('float32') / 255.0
		y_test.append(mapping[line[2]])
		x_test[i] = x
	return x_test, keras.utils.to_categorical(y_test, num_classes=3)

def calc_acc(testdata,testlabels,predictions):
    '''
    Accuracy calculation
    '''
    correct_predictions = 0
    wrong_predictions = 0
    for i in range(len(testdata)):
        if (predictions[i] == np.argmax(testlabels[i])):
           correct_predictions += 1
        else:
            wrong_predictions += 1

    # calculate accuracy
    acc = (correct_predictions/len(testdata)) * 100

    return acc

learnrate = 0.0002
batch_size = 8
display_step = 1

with open('train_split_1600.txt') as f:
	trainfiles = f.readlines()
with open('test_split_250.txt') as f:
	testfiles = f.readlines()


generator = BalanceCovidDataset(data_dir='Trainset',
								csv_file='train_split_1600.txt',
								batch_size=batch_size,
								input_shape=(224,224),
								covid_percent=0.3,
								class_weights=[1., 1., 4],
								top_percent=0.08)

infer_graph_path = 'checkpoint/inference_graph.pb'
output_ckpt_path = 'checkpoint/float_model.ckpt'
tboard_path = 'checkpoint/log/'
# Set up directories and files
INFER_GRAPH_DIR = os.path.dirname(infer_graph_path)
INFER_GRAPH_FILENAME =os.path.basename(infer_graph_path)

	#####################################################
	# Create the Computational graph
	#####################################################

epochs = 50
# define placeholders for the input images, labels, training mode
# Tensor input become 4-D: [Batch Size, Height, Width, Channel]
images_in = tf.compat.v1.placeholder(tf.float32, shape=[None,input_height,input_width,input_chan], name='images_in')
labels = tf.compat.v1.placeholder(tf.int32, shape=[None,3], name='labels')
train = tf.compat.v1.placeholder_with_default(False, shape=None, name='train')

	# build the network, input comes from the 'images_in' placeholder
	# training mode is also driven by placeholder
logits = Covid(cnn_in=images_in, is_training=train, drop_rate=0.0)

	# softmax cross entropy loss function - needs one-hot encoded labels
loss = tf.compat.v1.reduce_mean(tf.compat.v1.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels))

	# Adaptive Momentum optimizer - minimize the loss
update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learnrate)
with tf.compat.v1.control_dependencies(update_ops):
	train_op = optimizer.minimize(loss)

	# accuracy calculation during training:
	# Check to see if the predictions matches the labels and then
	# calculate accuracy as mean of the correct predictions
predicted_logit = tf.compat.v1.argmax(input=logits, axis=1, output_type=tf.int32)
 
	# TensorBoard data collection
tf.compat.v1.summary.scalar('cross_entropy_loss', loss)
tf.compat.v1.summary.image('input_images', images_in)    

	# set up saver object
saver = tf.compat.v1.train.Saver()

	#####################################################
	# Run the graph in a Session
	#####################################################
	# Launch the graph
x_test, y_test = Testset(testfiles)
with tf.compat.v1.Session() as sess:
	sess.run(tf.compat.v1.initializers.global_variables())
	saver.restore(sess,output_ckpt_path) 

	#writer = tf.compat.v1.summary.FileWriter(tboard_path, sess.graph)
	tb_summary = tf.compat.v1.summary.merge_all()
	print("\n----------------------------",flush=True)
	print(" TRAINING STARTED...",flush=True)
	print("----------------------------",flush=True)

	total_batches = len(generator) #int(len(x_train)/batchsize)
	progbar = tf.keras.utils.Progbar(total_batches)
	for epoch in range(epochs):
		for i in range(total_batches):
			x_batch, y_batch, weights = next(generator)
			train_feed_dict={images_in: x_batch, labels: y_batch, train: True}	
			_, s = sess.run([train_op, tb_summary], feed_dict=train_feed_dict)
			#writer.add_summary(s, (((epoch+1)*total_batches)-1))
			progbar.update(i+1)

		pred = sess.run(predicted_logit, feed_dict={images_in: x_test, labels: y_test, train: False})
		acc = calc_acc(x_test, y_test, pred)
		print (" Epoch", epoch+1, "/", epochs, '- accuracy {:1.2f}'.format(acc),'%',flush=True)

	#writer.flush()
	#writer.close()

		# save post-training checkpoint, this saves all the parameters of the trained network
	print("\n----------------------------",flush=True)
	print(" SAVING CHECKPOINT & GRAPH...",flush=True)
	print("----------------------------",flush=True)
	saver.save(sess, output_ckpt_path)
	print(' Saved checkpoint to %s' % output_ckpt_path,flush=True)

with tf.compat.v1.Graph().as_default():

      # define placeholders for the input data
      x_1 = tf.compat.v1.placeholder(tf.float32, shape=[None,input_height,input_width,input_chan], name='images_in')

      # call the CNN function with is_training=False
      #logits_1 = customcnn(cnn_in=x_1, is_training=False)
      logits_1 = cifar10(cnn_in=x_1, is_training=False, drop_rate=0.0)
      tf.io.write_graph(tf.compat.v1.get_default_graph().as_graph_def(), INFER_GRAPH_DIR, INFER_GRAPH_FILENAME, as_text=False)
      print(' Saved binary inference graph to %s' % infer_graph_path)

    
print(' Run `tensorboard --logdir=%s --port 6006 --host localhost` to see the results.' % tboard_path,flush=True)

	#####  SESSION ENDS HERE #############
