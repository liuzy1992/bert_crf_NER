#!/usr/bin/env python3

import os
import torch
from torch.utils.data import DataLoader, RandomSampler
from sklearn.metrics import classification_report
from . import device

def load_model(model, model_path):
	model_config = os.path.join(model_path, 'model.pt')
	assert os.path.exists(model_config), f'Model {model_config} not exists!'
	
	state_dict = torch.load(model_config, map_location=device)
	model.load_state_dict(state_dict)

	print(f'Model has been loaded from <== {model_path} .')

def evaluate(test_set, 
			 model, 
			 model_path, 
			 batch_size,
			 #eval_labels,
			 labels2idx):
	eval_labels2idx = {k: v for k, v in labels2idx.items() if k not in ['O', 'X']}

	test_sampler = RandomSampler(test_set)
	test_loader = DataLoader(dataset=test_set,
							 batch_size=batch_size,
							 sampler=test_sampler,
							 num_workers=0)
	
	y_pred = []
	y_true = []	

	load_model(model, model_path)
	
	model.eval()

	for step, batch in enumerate(test_loader):
		
		for key in batch.keys():
			batch[key] = batch[key].to(device)

		with torch.no_grad():
			out = model(**batch)

		print(out)
		print(batch['labels'].tolist())
		y_pred.extend(torch.argmax(logits, 1).tolist())
		y_true.extend(batch['labels'].tolist())

	print("## Classification Report:")
	print(classification_report(y_true, y_pred, labels=eval_labels, digits=4))
