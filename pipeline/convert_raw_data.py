#!/usr/bin/env python3

from collections import defaultdict
import numpy as np

class InputExample:
	'''
	A single set of example.
	'''
	def __init__(self, sent, labels=None):
		self.sent = sent
		self.labels = labels

def read_raw_file_to_examples(file_path, sep='\t'):
	'''
	Read raw file and convert to examples.
	'''
	
	examples = []
	sent = []
	labels = []
	with open(file_path, 'r') as f:
		for line in f.readlines():
			if line == '\n' or line == '':
				if sent:
					examples.append(InputExample(sent=sent,
												 labels=labels))
					sent = []
					labels = []
			else:
				splits = line.strip().split(sep)
				sent.append(splits[0])
				labels.append(splits[-1])

		if sent:
			examples.append(InputExample(sent=sent,
										 labels=labels))
			sent = []
			labels = []
	return examples

def get_labels(examples):
	'''
	Extract label indexes and counts from raw examples.
	'''

	labels = set()
	labels_counts = defaultdict(int)
	for example in examples:
		labels.update(example.labels)
		
		for label_ in example.labels:
			labels_counts[label_] += 1

	if 'O' not in labels:
		labels.add('O')
		labels_counts['O'] = 0
	
	label_list = list(labels)
	label_list.append('X') # 'X' for labels of [CLS], [SEP] and [PAD].
	# Convert set of labels to mapping labels -> indexes and indexes -> labels
	labels2idx = {label_ : i for i, label_ in enumerate(label_list)}
	idx2labels = {i : label_ for label_, i in labels2idx.items()}

	return labels2idx, idx2labels, dict(labels_counts)

def get_class_weight(labels2idx, labels_counts):
	'''
	Get the class weight based on the class labels frequency.
	'''

	labels2idx_list = [(k, v) for k, v in labels2idx.items() if k != 'X']
	labels2idx_list.sort(key=lambda x: x[1])
	total_labels = sum([count for label_, count in labels_counts.items()])
	class_weights = [total_labels / labels_counts[label_] for label_, _ in labels2idx_list]
	
	return np.array(class_weights) / max(class_weights)


