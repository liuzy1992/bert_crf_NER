#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel

class BaseModel(nn.Module):
	def __init__(self, bert_dir, dropout_prob):
		super(BaseModel, self).__init__()
		config_path = os.path.join(bert_dir, 'config.json')

		assert os.path.exists(bert_dir) and os.path.exists(config_path), 'pretrained bert file does not exist!'
		
		self.bert_module = BertModel.from_pretrained(bert_dir,
													 output_hidden_states=True,
													 hidden_dropout_prob=dropout_prob)
		self.bert_config = self.bert_module.config
		
	@staticmethod
	def _init_weights(blocks, **kwargs):
		for block in blocks:
			for module in block.modules():
				if isinstance(module, nn.Linear):
					if module.bias is not None:
						nn.init.zeros_(module.bias)
				elif isinstance(module, nn.Embedding):
					nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
				elif isinstance(module, nn.LayerNorm):
					nn.init.ones_(module.weight)
					nn.init.zeros_(module.bias)

class CRFModel(BaseModel):
	def __init__(self,
				 bert_dir,
				 num_tags,
				 dropout_prob=0.1,
				 task='train'
				 **kwargs):
		super(CRFModel, self).__init__(bert_dir=bert_dir, dropout_prob=dropout_prob)
		self.task = task

		out_dims = self.bert_config.hidden_size
		
		mid_linear_dims = kwargs.pop('mid_linear_dims', 128)

		self.mid_linear = nn.Sequential(
			nn.Linear(out_dims, mid_linear_dims),
			nn.ReLU(),
			nn.Dropout(dropout_prob)
		)
		
		out_dims = mid_linear_dims
		
		self.classifier = nn.Linear(out_dims, num_tags)

		self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
		self.loss_weight.data.fill_(-0.2)
		
		self.crf_module  = CRF(num_tags=num_tags, batch_first=True)

		init_blocks = [self.mid_linear, self.classifier]

		self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

	def forward(self,
				input_ids,
				attention_mask,
				token_type_ids,
				labels=None):

		bert_outputs = self.bert_module(input_ids,
										attention_mask,
										token_type_ids)
			
		seq_out = bert_outputs[0]
		seq_out = self.mid_linear(seq_out)
		emissions = self.classifier(seq_out)

		if self.task == 'train' and labels is not None:
			tokens_loss = -1 * self.crf_module(emissions=emissions,
											   tags=labels.long(),
											   mask=attention_mask.byte(),
											   reduction='mean')

			out = tokens_loss
				
		else:
			tokens_out = self.crf_module.decode(emissions=emissions,
												mask=attention_mask.byte())

			out = tokens_out

		return out
