#!/usr/bin/env python3

import os
from transformers import BertTokenizer

class BertFeature:
	'''
	Base feature for BERT input.
	'''
	def __init__(self,
				 token_ids,
				 attention_masks,
				 token_type_ids):
		self.token_ids = token_ids
		self.attention_masks = attention_masks
		self.token_type_ids = token_type_ids

class CRFFeature(BertFeature):
	'''
	Feature for CRF input.
	'''
	def __init__(self,
				 token_ids,
				 attention_masks,
				 token_type_ids,
				 labels=None):
		super(CRFFeature, self).__init__(token_ids=token_ids,
										 attention_masks=attention_masks,
										 token_type_ids=token_type_ids)
		self.labels = labels

def convert_examples_to_features(examples,
								 bert_dir,
								 labels2idx,
								 # pad_token_label_id=0,
								 max_seq_len=256):
	'''
	Take input examples and convert to features.
	'''
	assert os.path.exists(bert_dir), f'Directory of BERT model {bert_dir} not exists!'
	tokenizer = BertTokenizer.from_pretrained(bert_dir)
	pad_token_label_id = labels2idx['X']
	features = []
	for example in examples:
		tokens = []
		label_ids = []
		for word, label in zip(example.sent, example.labels):
			subword_tokens = tokenizer.tokenize(text=word)
			if len(subword_tokens) > 0:
				tokens.extend(subword_tokens)
				
				# use the read label id for the first token of the word, and padding ids for the remaining tokens
				label_ids.extend([labels2idx[label]] + [pad_token_label_id] * (len(subword_tokens) - 1))
				
		# drop part of the sequence longer than max_seq_len (account also for [CLS] and [SEP])
		if len(tokens) > max_seq_len - 2:
			tokens = tokens[: max_seq_len - 2]
			label_ids = label_ids[: max_seq_len - 2]
		
		# add special tokens for the list of tokens and its corresponding labels.
		tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
		label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]

		# create an attention mask (used to locate the padding)
		padding_len = (max_seq_len - len(tokens))
		attention_mask = [1] * len(tokens) + [0] * padding_len

		# add padding
		tokens += [tokenizer.pad_token] * padding_len
		label_ids += [pad_token_label_id] * padding_len

		# convert tokens to ids
		input_ids = tokenizer.convert_tokens_to_ids(tokens)

		# create segment_ids, all zeros since we only have one sentence
		segment_ids = [0] * max_seq_len

		# assert all input has the expected length
		assert len(input_ids) == max_seq_len
		assert len(label_ids) == max_seq_len
		assert len(attention_mask) == max_seq_len
		assert len(segment_ids) == max_seq_len

		# append input features for each sentence
		features.append(CRFFeature(token_ids=input_ids,
								   attention_masks=attention_mask,
								   token_type_ids=segment_ids,
								   labels=label_ids))

	return features
