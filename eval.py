#! /usr/bin/env python
# -*-coding=utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

emotion_label = {'sad':0, 'joy':1, 'disgust':2 , 'surprise':3, 'anger':4, 'fear':5}

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("test_file", "./data/trial.csv", "Test data source")
tf.flags.DEFINE_string("gold_labels", "./data/trial.labels", "Gold labels")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1534335682/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
#print("\nParameters:")
#for attr, value in sorted(FLAGS.__flags.items()):
#    print("{}={}".format(attr.upper(), value))
#print("")

def evaluation(predict_list, gold_list):
	emotion = {0:'sad', 1:'joy', 2:'disgust', 3:'surprise', 4:'anger', 5:'fear'}
	performance = {0:[0,0,0], 1:[0,0,0], 2:[0,0,0], 3:[0,0,0], 4:[0,0,0], 5:[0,0,0]}
	total_tp = total_tn = total_fp = total_fn = total_precision = total_recall = 0
	#for each emotion, count tp, fp, tn, fn and calculate its precision and recall seperatly
	for e in range(len(emotion)): 
		tp = tn = fp = fn = 0
		accuracy = precision = recall = f1 = 0
		#count tp, fp, tn, fn for e
		for i in range(len(predict_list)):
			gold = gold_list[i]
			predict = int(predict_list[i])
			if gold == predict and gold == e:
				tp += 1
			elif predict == e and gold != e:
				fp += 1
			elif gold == e and predict != e:
				fn += 1
			else:
				tn += 1
				
		if (tp + fp) != 0:
			precision = 1.0 * tp / (tp + fp) * 100
		if (tp + fn) != 0:
			recall = 1.0 * tp / (tp + fn) * 100
		if (precision + recall) != 0:
			f1 = 2.0 * precision * recall / (precision + recall)
		#saving performance for each emotion
		performance[e] = [precision, recall, f1]		
		#adding up together for further calculation
		total_tp += tp
		total_tn += tn
		total_fp += fp
		total_fn += fn
		total_precision += precision
		total_recall += recall
		#print			
		
	#micro_f1 calculation
	if (total_tp + total_fp) != 0:
		micro_precison = 1.0  *total_tp / (total_tp + total_fp) * 100
	if (total_tp + total_fn)!= 0:
		micro_recall = 1.0  *total_tp / (total_tp + total_fn) * 100
	if (micro_precison + micro_recall != 0):
		micro_f1 = 2.0  * micro_precison * micro_recall / (micro_precison + micro_recall)
			
	#macro_f1 calculation
	macro_precison = total_precision / len(emotion)
	macro_recall = total_recall / len(emotion)
	if (macro_precison + macro_recall) != 0 :
		macro_f1 = 2.0 * macro_precison * macro_recall / (macro_precison + macro_recall)
	
	for e in range(len(emotion)):
		print(emotion[e], '	Precision:', performance[e][0], '	Recall:', performance[e][1],'	F1:', performance[e][2])
	print("Micro F1 Score:",micro_f1)
	print("Macro F1 Score:",macro_f1)

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.test_file, FLAGS.gold_labels)
    #y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
        print(type(all_predictions[0]))

# Print accuracy if y_test is defined
if y_test is not None:

    #correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    evaluation(all_predictions, y_test)
    #print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
