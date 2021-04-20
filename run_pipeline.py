#!/usr/bin/env python3

import sys
from pipeline.convert_raw_data import *
from pipeline.feature_utils import *
from pipeline.dataset_utils import NERDataset
from sklearn.model_selection import train_test_split
from pipeline.bert_crf_model import CRFModel
from pipeline.trainer import train
from pipeline.evaluator import evaluate

def main(train_file,
		 test_file,
		 bert_dir,
		 dropout_prob,
		 max_seq_len,
		 batch_size,
		 num_epochs,
		 lr,
		 model_outdir	 
		 ):
	print(f"********** Loading data from {train_file} **********")
	raw_examples = read_raw_file_to_examples(train_file)
	labels2idx, idx2labels, labels_counts = get_labels(raw_examples)
	train_examples, valid_examples = train_test_split(raw_examples, test_size=0.2, random_state=25)
	print("## train set size: {:}; valid set size: {:}".format(len(train_examples), len(valid_examples)))
	# raw_features = convert_examples_to_features(raw_examples, bert_dir, labels2idx, max_seq_len=max_seq_len)
	train_features = convert_examples_to_features(train_examples, bert_dir, labels2idx, max_seq_len=max_seq_len)
	valid_features = convert_examples_to_features(valid_examples, bert_dir, labels2idx, max_seq_len=max_seq_len)
	print()
	print(f"********** Convert data to NER datasets **********")
	# raw_dataset = NERDataset(raw_features)
	train_set = NERDataset(train_features)
	valid_set = NERDataset(valid_features)
	# train_set, valid_set = train_test_split(raw_dataset, test_size=0.2, random_state=25)
	# print("## train set size: {:}; valid set size: {:}".format(len(train_set), len(valid_set)))
	print()
	print(f"********** Training **********")
	model = CRFModel(bert_dir=bert_dir, num_tags=len(labels2idx), dropout_prob=dropout_prob)
	train(train_set,
		  valid_set,
		  model,
		  batch_size,
		  num_epochs,
		  lr,
		  model_outdir)
	print()
	print(f"********** Evaluation **********")
	test_examples = read_raw_file_to_examples(test_file)
	test_features = convert_examples_to_features(test_examples, bert_dir, labels2idx, max_seq_len=max_seq_len)
	test_dataset = NERDataset(test_features)
	print("## test set size: {:}. ".format(len(test_dataset)))
	evaluate(test_dataset,
			 model,
			 model_outdir,
			 batch_size,
			 labels2idx)

if __name__ == '__main__':
	main(sys.argv[1],
		 sys.argv[2],
		 sys.argv[3],
		 float(sys.argv[4]),
		 int(sys.argv[5]),
		 int(sys.argv[6]),
		 int(sys.argv[7]),
		 float(sys.argv[8]),
		 sys.argv[9])
