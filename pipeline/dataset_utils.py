#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
	'''
	convert features to input datasets for NER
	'''
	def __init__(self, features):
		self.nums = len(features)
		self.input_ids = [torch.tensor(feature.token_ids).long() for feature in features]
		self.attention_mask = [torch.tensor(feature.attention_masks).float() for feature in features]
		self.token_type_ids = [torch.tensor(feature.token_type_ids).long() for feature in features]
		self.labels = [torch.tensor(feature.labels) for feature in features]

	def __len__(self):
		return self.nums

	def __getitem__(self, index):
		data = {'input_ids': self.input_ids[index],
				'attention_mask': self.attention_mask[index],
				'token_type_ids': self.token_type_ids[index],
				'labels': self.labels[index]}

		return data
