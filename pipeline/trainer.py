#!/usr/bin/env python3

import os
import time
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
# from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from . import device

def save_model(model, model_outdir):
	if not os.path.exists(model_outdir):
		os.makedirs(model_outdir, exist_ok=True)

	torch.save(model.state_dict(), os.path.join(model_outdir, 'model.pt'))
	print(f'Model has been saved to ==> {model_outdir} .')

def train(train_set,
		  valid_set,
		  model,
		  batch_size,
		  num_epochs,
		  lr,
		  model_outdir):
	model.to(device)

	train_sampler = RandomSampler(train_set)
	valid_sampler = RandomSampler(valid_set)
	
	train_loader = DataLoader(dataset=train_set,
							  batch_size=batch_size,
							  sampler=train_sampler,
							  num_workers=0)
	valid_loader = DataLoader(dataset=valid_set,
							  batch_size=batch_size,
							  sampler=valid_sampler,
							  num_workers=0)

	total_steps = len(train_loader) * num_epochs
	
	optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
	scheduler = get_linear_schedule_with_warmup(optimizer,
												num_warmup_steps=0,
												num_training_steps=total_steps)

	train_loss_list = []
	valid_loss_list = []
#	train_r2_list = []
#	valid_r2_list = []
	epoch_list = []
	
	best_valid_loss = float('Inf')
	total_t0 = time.time()

	for epoch in range(num_epochs):
		t0 = time.time()
		total_train_loss = 0
#		total_train_r2 = 0
		
		model.train()
	
		for step, batch in enumerate(train_loader):

			for key in batch.keys():
				batch[key] = batch[key].to(device)

			loss = model(**batch)

			total_train_loss += loss.item()
			# total_train_r2 += r2_score(batch['labels'].tolist(), torch.argmax(logits, 1).tolist())
			# total_train_r2 += r2_score(batch['labels'].tolist(), logits)
			
			loss.backward()

			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			scheduler.step()

		avg_train_loss = total_train_loss / len(train_loader)
		# avg_train_r2 = total_train_r2 / len(train_loader)

		model.eval()

		total_valid_loss = 0
		# total_valid_r2 = 0
		
		for step, batch in enumerate(valid_loader):
			
			for key in batch.keys():
				batch[key] = batch[key].to(device)

			with torch.no_grad():
				# loss, logits = model(**batch)
				loss = model(**batch)

			total_valid_loss += loss.item()
			# total_valid_r2 += r2_score(batch['labels'].tolist(), torch.argmax(logits, 1).tolist())
			# total_valid_r2 += r2_score(batch['labels'].tolist(), logits)

		avg_valid_loss = total_valid_loss / len(valid_loader)
		# avg_valid_r2 = total_valid_r2 / len(valid_loader)

		# print("## Epoch {}/{} ==> train loss: {:.5f}, train R2: {:.5f}; valid loss: {:.5f}, valid R2: {:.5f}; elapsed time: {:.2f} s.".format(
		#				epoch + 1,
		#				num_epochs,
		#				avg_train_loss,
		#				avg_train_r2,
		#				avg_valid_loss,
		#				avg_valid_r2,
		#				time.time() - t0
		#				))
		print("## Epoch {}/{} ==> train loss: {:.5f}, valid loss: {:.5f}; elapsed time: {:.2f} s.".format(epoch + 1,
							  num_epochs,
							  avg_train_loss,
							  avg_valid_loss,
							  time.time() - t0))

		train_loss_list.append(avg_train_loss)
		# train_r2_list.append(avg_train_r2)
		valid_loss_list.append(avg_valid_loss)
		# valid_r2_list.append(avg_valid_r2)
		epoch_list.append(epoch + 1)

		if best_valid_loss > avg_valid_loss:
			best_valid_loss = avg_valid_loss
			save_model(model, model_outdir)

	print("Training complete! Total elapsed time: {:.2f} s.".format(time.time() - total_t0))

	plt.plot(epoch_list, train_loss_list, label='train loss')
	plt.plot(epoch_list, valid_loss_list, label='valid loss')
	# plt.plot(epoch_list, train_r2_list, label='train R2')
	# plt.plot(epoch_list, valid_r2_list, label='valid R2')
	plt.xlabel('Epoch')
	plt.legend()
	plt.show()
				
