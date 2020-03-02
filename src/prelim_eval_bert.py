# prelim_eval_bert.py

from bert_embedding import BertEmbedding
from nltk.corpus import brown
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
	tokenized_text = [tokenizer.tokenize("[CLS] " + " ".join(c) + " [SEP]") for c in corpus]

	# print(tokenized_text[1])

	maxlen = 0
	for i in tokenized_text :
		if len(i) > maxlen :
			maxlen = len(i)

	padded_text = [i + ["[PAD]"] * (maxlen - len(i)) for i in tokenized_text]
	# print(padded_text[1])
	# attn_mask = [1 if word != "[PAD]" else 0 for sent in tokenized_text for word in sent]
	attn_mask = []
	for sent in padded_text :
		attn_mask.append([1 if word != "[PAD]" else 0 for word in sent])
	# print(attn_mask[1])

	# print(sum([len(padded_text[x]) != len(attn_mask[x]) for x in range(len(padded_text))]))

	# tokenized_text = tokenizer.tokenize("[CLS]" + " ".join(brown.sents(categories=["fiction"])[1]) + "[SEP]")
	indexed_text = [tokenizer.convert_tokens_to_ids(sent) for sent in padded_text]

	# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

	segments_ids = []
	for text in indexed_text :
		segments_ids.append([1] * len(text))

	input_ids = torch.tensor(indexed_text)
	attn_tensor = torch.tensor(attn_mask)
	segments_tensor = torch.tensor(segments_ids)

	# segments_ids = [1] * len(tokenized_text)

	# tokens_tensor = torch.tensor([indexed_tokens])
	# segments_tensor = torch.tensor([segments_ids])

	# print(tokenized_text)
	# print(tokens_tensor)

	# print(input_ids.shape)
	config = BertConfig()
	config.output_hidden_states = True
	# model = BertModel.from_pretrained("bert-base-uncased")
	model = BertModel(config)
	model.eval()


	with torch.no_grad() :
		# encoded_layers, _ = model(input_ids, attention_mask=attn_mask)
		# encoded_layers, _ = model(torch.tensor(tokenizer.convert_tokens_to_ids(['[CLS]', 'why', 'does', 'this', 'not', 'work', '[SEP]', '[PAD]'])))

		# sentence = "the red cube is at your left"
		# tokens = ["[CLS]"] + tokenizer.tokenize(sentence) + ["[SEP]"] 
		# input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))

		# encoded_layers, _ = model(input_ids[1].unsqueeze(0), attention_mask=attn_tensor[1].unsqueeze(0))
		# _, _, hidden_states = model(input_ids[1].unsqueeze(0), attn_tensor[1].unsqueeze(0))
		_ = model(input_ids[1].unsqueeze(0), attn_tensor[1].unsqueeze(0))
		print(model.get_input_embeddings().shape)


		# print(hidden_states[0].shape, hidden_states[1].shape) # number of examples, number of tokens, number of hidden states

		# embeddings = torch.stack(encoded_layers, dim=0)
		# embeddings = torch.squeeze(embeddings, dim=1)
		# embeddings = embeddings.permute(1, 0, 2)

	# vecs = []
	# embeddings = torch.squeeze(encoded_layers, dim=0)
	# print(embeddings.shape)


	# print(len(vecs), len(vecs[0]))


if __name__ == "__main__" :
	get_embeddings()
