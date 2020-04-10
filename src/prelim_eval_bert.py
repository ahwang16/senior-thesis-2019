# prelim_eval_bert.py

from bert_embedding import BertEmbedding
from collections import defaultdict
from nltk.corpus import brown
import pickle as pkl
from transformers import BertModel, BertTokenizer, BertConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from pytorch_pretrained_bert import BertTokenizer


class BrownDataset(Dataset) :
	def __init__(self, maxlen) :
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

		self.corpus = brown.sents(categories=["fiction"])
		# The Brown Corpus comes pre-tokenized. join() to use BertTokenizer.
		for x in range(len(self.corpus)) :
			self.corpus[x] = " ".join(corpus[x])

		self.maxlen = maxlen


	def __len__(self) :
		return len(self.corpus)


	def __getitem__(self, index) :

		# select sentence at index
		sentence = self.corpus[index]

		# preprocessing for BERT
		tokens = self.tokenizer.tokenize(sentence)
		tokens = ['[CLS]'] + tokens + ['SEP']

		# padding
		if len(tokens) < self.maxlen :
			tokens += ['[PAD]' for _ in range(self.maxlen - len(tokens))]
		else :
			tokens = tokens[:self.maxlen-1] + ['[SEP]']

		tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
		tokens_ids_tensor = torch.tensor(tokens_ids)

		attn_mask = (tokens_ids_tensor != 0).long()

		return tokens_ids_tensor, attn_mask

# dim: layer number, batch number, word/token number, hidden unit/feature number
# expected dim: [12, 1, num_tok, num_unit]
def get_embeddings() :

	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

	corpus = brown.sents(categories=['fiction'])
	print("corpus length", len(corpus))

	maxlen = 0
	for i in corpus :
		if len(i) > maxlen :
			maxlen = len(i)
	maxlen += 2

	print("max len", maxlen)

	tokenized_text = [tokenizer.encode(c, add_special_tokens=True, max_length=maxlen, pad_to_max_length=True, return_tensors="pt") for c in corpus]
	# ids = [tokenizer.encode(c, add_special_tokens=True, max_length=maxlen, pad_to_max_length=True, return_tensors="pt") for c in corpus]
	# ids = [nn.functional.pad(t, (0, maxlen - len(t)), value=tokenizer.pad_token_id, ) for t in tokenized_text]
	ids = torch.stack(tokenized_text)

	attn_mask = (ids != 0).float()

	# print(ids.shape)
	# print(attn_mask.shape)

	# print(tokenized_text[1])

	

	# padded_text = [i + ["[PAD]"] * (maxlen - len(i)) for i in tokenized_text]
	# print(padded_text[1])
	# attn_mask = [1 if word != "[PAD]" else 0 for sent in tokenized_text for word in sent]
	# attn_mask = tensor()
	# for sent in tokenized_text :
	# 	attn_mask.append([1 if word != 0 else 0 for word in sent])
	# print(attn_mask[1])

	# print(sum([len(padded_text[x]) != len(attn_mask[x]) for x in range(len(padded_text))]))

	# tokenized_text = tokenizer.tokenize("[CLS]" + " ".join(brown.sents(categories=["fiction"])[1]) + "[SEP]")
	# indexed_text = [tokenizer.convert_tokens_to_ids(sent) for sent in padded_text]

	# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

	# segments_ids = []
	# for text in indexed_text :
	# 	segments_ids.append([1] * len(text))

	# input_ids = torch.tensor(indexed_text)
	# attn_tensor = torch.tensor(attn_mask)
	# segments_tensor = torch.tensor(segments_ids)

	# segments_ids = [1] * len(tokenized_text)

	# tokens_tensor = torch.tensor([indexed_tokens])
	# segments_tensor = torch.tensor([segments_ids])

	# print(tokenized_text)
	# print(tokens_tensor)

	# print(input_ids.shape)
	# config = BertConfig(output_hidden_states=True)
	# config.output_hidden_states = True
	model = BertModel.from_pretrained("bert-base-uncased")
	# model = BertModel(config)
	model.eval()


	with torch.no_grad() :
		print(ids.shape, attn_mask.shape)
		out = model(ids, attention_mask=attn_mask)
		embeddings = out[0]
		print(embeddings.shape)


		# embeddings will match up with the token ids
		# DON'T USE CLS, SEP, PAD



		# encoded_layers, _ = model(input_ids, attention_mask=attn_mask)
		# encoded_layers, _ = model(torch.tensor(tokenizer.convert_tokens_to_ids(['[CLS]', 'why', 'does', 'this', 'not', 'work', '[SEP]', '[PAD]'])))

		# sentence = "the red cube is at your left"
		# tokens = ["[CLS]"] + tokenizer.tokenize(sentence) + ["[SEP]"] 
		# input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))

		# encoded_layers, _ = model(input_ids[1].unsqueeze(0), attention_mask=attn_tensor[1].unsqueeze(0))
		# _, _, hidden_states = model(input_ids[1].unsqueeze(0), attn_tensor[1].unsqueeze(0))
		# _ = model(input_ids[1].unsqueeze(0), attn_tensor[1].unsqueeze(0))
		# print(model.get_input_embeddings().shape)


		# print(hidden_states[0].shape, hidden_states[1].shape) # number of examples, number of tokens, number of hidden states

		# embeddings = torch.stack(encoded_layers, dim=0)
		# embeddings = torch.squeeze(embeddings, dim=1)
		# embeddings = embeddings.permute(1, 0, 2)

	# vecs = []
	# embeddings = torch.squeeze(encoded_layers, dim=0)
	# print(embeddings.shape)


	# print(len(vecs), len(vecs[0]))


def get_embeddings2():
	batch_size = 32

	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

	corpus = brown.sents(categories=['fiction'])

	maxlen = 0
	for i in corpus :
		if len(i) > maxlen :
			maxlen = len(i)
	maxlen += 2

	config = BertConfig(output_hidden_states=True)
	model = BertModel(config)
	# model = BertModel.from_pretrained("bert-base-uncased")
	model.eval()

	embeddings_dict = defaultdict(list)


	for sent in corpus :
		print(sent)
		ids = tokenizer.encode(sent, add_special_tokens=True, max_length=maxlen, pad_to_max_length=True, return_tensors="pt")
		# ids = torch.stack(tokenized_text)
		attn_mask = (ids != 0).float()

		with torch.no_grad() :
			print(ids.shape, attn_mask.shape)
			out = model(ids, attention_mask=attn_mask)
			# embeddings = out[2]

			# # print(len(out))
			# print(len(embeddings))
			# print(embeddings[0].shape)

			token_embeddings = torch.squeeze(torch.stack(out[2][1:], dim=0), dim=1).permute(1, 0, 2)
			# token_embeddings = torch.squeeze(token_embeddings, dim=1)
			print(token_embeddings.size())

			s = ["CLS"] + sent + ["SEP"] + ["PAD"] * (maxlen - len(sent) - 2)

			idx = 0
			for token in token_embeddings :
				sum_vec = torch.sum(token[-4:], dim=0)
				if s[idx] == "CLS":
					idx += 1
					continue
				elif s[idx] == "SEP":
					break

				embeddings_dict[s[idx]] = sum_vec
				idx += 1

	print(len(embeddings_dict))

	with open("../data/bert_embeddings.pkl", "wb") as f:
		pkl.dump(embeddings_dict, f)















	# tokenized_text = [torch.tensor(tokenizer.encode(c, add_special_tokens=True, max_length=maxlen, pad_to_max_length=True)) for c in corpus]

	# ids = torch.stack(tokenized_text)
	# print(ids.shape)

	# attn_mask = (ids != 0).float()

	# model = BertModel.from_pretrained("bert-base-uncased")
	# model.eval()

	# with torch.no_grad() :
	# 	print(ids.shape, attn_mask.shape)
	# 	out = model(ids, attention_mask=attn_mask) # errors out here
	# 	embeddings = out[0]
	# 	print(embeddings.shape)


if __name__ == "__main__" :
	get_embeddings2()


