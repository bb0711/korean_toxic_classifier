	# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers

def get_word(d,x):
	return d[x]

class TextCNN(object):
	"""
	A CNN for text classification.
	Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
	sequence_length = length of sentence
	num_classes = num of output class => 2 for toxic/untoxic
	"""
	def __init__(self, sequence_length, num_classes, word_dict,embedding_matrix,
		embedding_size, filter_sizes, num_filters, l2_reg_lambda=3):
		#none = batch size
		self.input_x = tf.placeholder(tf.int32,[None, sequence_length], name="input_x") # input comment 개수 따라 달라짐
		self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")

		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		vocab_size = len(word_dict)

		
		# Keeping track of l2 regularization loss (optional)
		l2_loss = tf.constant(3.0)

		with tf.device('/cpu:0'), tf.name_scope("embedding"):

			E = tf.convert_to_tensor(embedding_matrix)
			self.embedded_chars = tf.nn.embedding_lookup(E, self.input_x)
			self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

			pooled_outputs=[]
			# input = seqence_length * 300
			for i, filter_size in enumerate(filter_sizes):
				with tf.name_scope("conv-maxpool-%s" % filter_size):
					# Convolution Layer
					filter_shape = [filter_size, embedding_size, 1, num_filters]
					W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
					b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
					conv = tf.nn.conv2d(
						self.embedded_chars_expanded,#self.embedded_chars_expanded, # input= [batch, height, width, in channel]
						W,#filter=[filter h, filter w, in channel, out channel]
						strides=[1, 1, 1, 1],
						padding="VALID",
						name="conv")
					# Apply nonlinearity
					h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
					# Max-pooling over the outputs
					pooled = tf.nn.max_pool(
						h,
						ksize=[1, sequence_length - filter_size + 1, 1, 1],
						strides=[1, 1, 1, 1],
						padding='VALID',
						name="pool")
					pooled_outputs.append(pooled)
		 
		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(pooled_outputs,3)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
		
		
		# Add dropout

		with tf.name_scope("dropout"):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
			
		# Final (unnormalized) scores and predictions
		with tf.name_scope("output"):
			W = tf.get_variable(
				"W",
				shape=[num_filters_total, num_classes],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
			self.predictions = tf.argmax(self.scores, 1, name="predictions")
			
		# Calculate mean cross-entropy loss
		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
			
		
		# Calculate Accuracy
		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
			
		#caculate precision
		# [1,0] = toxic  as positive
		with tf.name_scope("precision"):
			pred = tf.cast(tf.argmax(self.input_y, 1), tf.int64)
			real = tf.cast(self.predictions, tf.int64)
			tp = tf.count_nonzero(pred * real)
			fp = tf.count_nonzero(pred * (real - 1))
			sum = tp + fp
			if sum == 0:
				self.precision = tf.cast(tf.one_like(1), tf.float32)
			else:
				self.precision = tp / sum
		
		with tf.name_scope("recall"):
			pred = tf.cast(tf.argmax(self.input_y, 1), tf.int64)
			real = tf.cast(self.predictions, tf.int64)
			tp = tf.count_nonzero(pred * real)
			fn = tf.count_nonzero((pred - tf.ones_like(tf.size(pred), dtype=tf.int64)) * real)
			sum = tp + fn
			if sum == 0:
				self.recall = tf.cast(tf.one_like(1), tf.float32)
			else:
				self.recall = tp / sum

		with tf.name_scope("wrong"):
			pred = tf.cast(tf.argmax(self.input_y, 1), tf.int64)
			real = tf.cast(self.predictions, tf.int64)
			fp = tf.count_nonzero(pred * (real - 1))
			fn = tf.count_nonzero((pred - tf.ones_like(tf.size(pred), dtype=tf.int64)) * real)
			self.wrong = fp + fn

		with tf.name_scope("f1"):
			self.f1 = 2* self.precision * self.recall / (self.precision + self.recall)



# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "text/pos_cl_decom.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "text/neg_cl_decom.txt", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 300, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.85, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 2000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def preprocess():
	# Data Preparation
	# ==================================================

	# Load data
	#print("Loading data...")
	x, y , word_dict,embedding_matrix, xt, yt = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
	#print(len(word_dict.keys())) = 16336
	#print(len(embedding_matrix), len(embedding_matrix[23])) => 16336, 300
	
	# Randomly shuffle data
	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(len(y)))
	x_shuffled = np.array(x)[shuffle_indices]
	y_shuffled = np.array(y)[shuffle_indices]

	# Split train/test set
	# TODO: This is very crude, should use cross-validation
	dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
	x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
	y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
	print("lenlen\n")
	print(len(y_train), len(y_dev))
	del x, y, x_shuffled, y_shuffled

	#print("Vocabulary Size: {:d}".format(len(word_dict)))
	#print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

	return x_train, y_train, word_dict,embedding_matrix, x_dev, y_dev, np.array(xt), np.array(yt)

def train(x_train, y_train, word_dict,embedding_matrix, x_dev, y_dev, xt,yt):
	# Training
	# ==================================================

	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(
		  allow_soft_placement=FLAGS.allow_soft_placement,
		  log_device_placement=FLAGS.log_device_placement)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn = TextCNN(
				sequence_length=44,#max_document_length,#x_train.shape[0].shape[1],
				num_classes=y_train.shape[1],
				word_dict= word_dict,
				embedding_matrix = embedding_matrix,
				embedding_size=FLAGS.embedding_dim,
				filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
				num_filters=FLAGS.num_filters,
				l2_reg_lambda=FLAGS.l2_reg_lambda)

			# Define Training procedure
			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(1e-3)
			grads_and_vars = optimizer.compute_gradients(cnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

			# Keep track of gradient values and sparsity (optional)
			grad_summaries = []
			for g, v in grads_and_vars:
				if g is not None:
					grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
					sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
					grad_summaries.append(grad_hist_summary)
					grad_summaries.append(sparsity_summary)
			grad_summaries_merged = tf.summary.merge(grad_summaries)

			# Output directory for models and summaries
			timestamp = str(int(time.time()))
			out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
			#print("Writing to {}\n".format(out_dir))

			# Summaries for loss and accuracy
			loss_summary = tf.summary.scalar("loss", cnn.loss)
			acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
			pre_summary = tf.summary.scalar("precision",cnn.precision)

			# Train Summaries
			train_summary_op = tf.summary.merge([loss_summary, acc_summary, pre_summary, grad_summaries_merged])
			train_summary_dir = os.path.join(out_dir, "summaries", "train")
			train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

			# Dev summaries
			dev_summary_op = tf.summary.merge([loss_summary, acc_summary, pre_summary])
			dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
			dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

			# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
			checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			checkpoint_prefix = os.path.join(checkpoint_dir, "model")
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)
			saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)


			# Initialize all variables
			sess.run(tf.global_variables_initializer())

			def train_step(x_batch, y_batch):
				"""
				A single training step
				"""

				feed_dict = {
				  cnn.input_x: x_batch,
				  cnn.input_y: y_batch,
				  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
				}
				_, step, summaries, loss, accuracy, precision, recall, f1, wrong = sess.run(
					[train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall, cnn.f1, cnn.wrong],
					feed_dict)
				time_str = datetime.datetime.now().isoformat()
				if step %100==0:
					print("train --> {}: step {}, loss {:g}, acc {:g}, precision {:g}  recall {:g} f1score {:g} wrong {:g}".format(time_str, step, loss, accuracy, precision, recall, f1,wrong))
				train_summary_writer.add_summary(summaries, step)

			def dev_step(x_batch, y_batch, final, writer=None ):
				"""
				Evaluates model on a dev set
				"""
				feed_dict = {
				  cnn.input_x: x_batch,
				  cnn.input_y: y_batch,
				  cnn.dropout_keep_prob: 1.0
				}
				step, summaries, loss, accuracy, precision, recall, f1, wrong = sess.run(
					[global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall, cnn.f1, cnn.wrong],
					feed_dict)
				time_str = datetime.datetime.now().isoformat()
				
				wf = open('jamo_output_test2.txt', mode='a', encoding='utf-8')
				
				strs = "{}: step {}, loss {:g}, acc {:g}, precision {:g} recall {:g} f1score {:g} wrong{:g}\n".format(
					time_str, step, loss, accuracy,
					precision, recall, f1, wrong)
				wf.write(strs)
				print(strs)
					
				if writer:
					writer.add_summary(summaries, step)

			# Generate batches
			batches = data_helpers.batch_iter(
				list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
			# Training loop. For each batch...

			for batch in batches:
				x_batch, y_batch = zip(*batch)
				train_step(x_batch, y_batch)
				current_step = tf.train.global_step(sess, global_step)
				
				if current_step % FLAGS.evaluate_every == 0:
					print("\nEvaluation:")
					print("len:",len(y_dev))
					dev_step(xt, yt, 0, writer=dev_summary_writer)
				
			
			path = saver.save(sess, checkpoint_prefix)
			print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
	for i in range(10):
		#print("NUM",i)
		x_train, y_train,  word_dict,embedding_matrix, x_dev, y_dev, xt, yt = preprocess()
		break
		train(x_train, y_train, word_dict,embedding_matrix, x_dev, y_dev,xt,yt)
		break
		
if __name__ == '__main__':
	tf.app.run()