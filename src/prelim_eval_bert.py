# prelim_eval_bert.py
from bert_embedding import BertEmbedding
from collections import defaultdict
from nltk.cluster import KMeansClusterer
from nltk.cluster.util import cosine_distance
from nltk.corpus import brown, wordnet
import numpy as np
import pickle as pkl
from sklearn.cluster import AgglomerativeClustering
import sys
from transformers import BertModel, BertTokenizer, BertConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def get_embeddings2():
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	print('corpus')
	corpus = brown.sents(categories=['fiction'])

	# max length of sequence for padding
	maxlen = 0
	for i in corpus :
		if len(i) > maxlen :
			maxlen = len(i)
	maxlen += 2
	
	# default is bert-base-uncased, edit config to return all layers
	config = BertConfig(output_hidden_states=True)
	model = BertModel(config)
	model.eval()

	# token : [vectors] (each vector is 1x768 and sum of last 4 layers, one vector for each appearance of token)
	embeddings_dict = defaultdict(list)

	# run one sentence at a time through encoder and bert
	for sent in corpus :
		print(sent)
		ids = tokenizer.encode(sent, add_special_tokens=True, max_length=maxlen, pad_to_max_length=True, return_tensors="pt")
		# ids = torch.stack(tokenized_text)
		attn_mask = (ids != 0).float()

		with torch.no_grad() :
			out = model(ids, attention_mask=attn_mask)
			token_embeddings = torch.squeeze(torch.stack(out[2][1:], dim=0), dim=1).permute(1, 0, 2)

			s = ["CLS"] + sent + ["SEP"] + ["PAD"] * (maxlen - len(sent) - 2)

			idx = 0
			for token in token_embeddings :
				sum_vec = torch.sum(token[-4:], dim=0)
				if s[idx] == "CLS":
					idx += 1
					continue
				elif s[idx] == "SEP":
					break

				embeddings_dict[s[idx]].append(sum_vec)
				idx += 1

	print(len(embeddings_dict))

	with open("../data/bert_embeddings.pkl", "wb") as f:
		pkl.dump(embeddings_dict, f)


def load_bert(path="../data/bert_embeddings.pkl"):
	with open(path, "rb") as f:
		return pkl.load(f)


# Get vocabulary from fiction set of Brown corpus
def get_brown_vocab() :
	return set(brown.words(categories=['fiction']))


def kmeans(vocab, k=900, r=25, file_num=0) :
	"""
	Use k-means clustering with cosine similarity as the distance metric to
	cluster the data into k groups.
	    
    Params:
        k (int): number of clusters
        vocab (list/dict): text vocabulary of dataset
        r (int): number of randominzed cluster trials (optional parameter for KMeansClusterer)
        
    Returns:
        cluster_dict (dict): cluster index (int) mapping to cluster (set)
        word_to_cluster (dict): vocab index mapping word (string) to cluster number (int)
	"""
	print("loading bert")
	bert = load_bert()

	# bert word embeddings
	print("berting vocab")
	embeds, words = [], []
	len_vocab = len(vocab)
	for v in vocab:
		embeds.append(torch.mean(torch.stack(bert[v]), dim=0))
		words.append(v)

	### CLUSTER ################################################################
	print("clustering")
	clusterer = KMeansClusterer(k, distance=cosine_distance, repeats=r)
	clusters = clusterer.cluster(embeds, assign_clusters=True)

	print("enumerating")
	cluster_dict = { i : [] for i in range(k) }
	word_to_cluster = {}

	for i, v in enumerate(words):
		cluster_dict[clusters[i]].append(v)
		word_to_cluster[v] = clusters[i]

	for c in cluster_dict :
		cluster_dict[c] = set(cluster_dict[c])

	print("pickling")
	with open("../data/bert_clusters_{}.pkl".format(file_num), "wb") as p :
		pkl.dump(cluster_dict, p)

	############################################################################

	# write individual precision and recall scores to text file
	f = open("../data/bert_{}.txt".format(file_num), "w")
	f.write("vocab\tprecision\trecall\n")

	precision, recall = [], [] # precision and recall for each word
	pre, rec = { i : [] for i in range(k)}, { i : [] for i in range(k)} # cluster to score mapping
	count = 0 # print for sanity check
	unknown = 0
	for w in words :
		p, r = 0.0, 0.0

		cluster = get_cluster(w, cluster_dict, word_to_cluster)

		# accumulate gold cluster for v with WordNet
		gold = []
		for syn in wordnet.synsets(w) :
			for l in syn.lemmas() :
				gold.append(l.name())
		gold = set(gold)
		if len(gold) == 0 :
			unknown += 1
			continue

		gold.add(w)

		intersection = cluster.intersection(gold) # true positive

		p = len(intersection) / (len(intersection) + len(cluster.difference(gold)))
		r = len(intersection) / (len(intersection) + len(gold.difference(cluster)))
		
		# try:
		# 	p = len(intersection) / (len(intersection) + len(cluster.difference(gold)))
		# except:
		# 	continue
		# try:
		# 	r = len(intersection) / (len(intersection) + len(gold.difference(cluster)))
		# except:
		# 	continue

		f.write("{}\t{}\t{}\n".format(w, p, r))

		count += 1
		if count % 10 == 0 :
			rint("{}/{}".format(count, len_vocab))
			print(p, r)


		precision.append(p)
		recall.append(r)
		pre[word_to_cluster[w]].append(p)
		rec[word_to_cluster[w]].append(r)

	p_bar, r_bar = np.mean(precision), np.mean(recall)

	f.write("\naverage\t{}\t{}\n".format(p_bar, r_bar))
	f.close()

	scores = open("../data/kmeans_scores_{}.txt".format(file_num), "w")
	scores.write("cluster\tprecision\trecall\n")
	for i in range(k) :
		scores.write("{}\t{}\t{}\n".format(i, np.mean(pre[i]), np.mean(rec[i])))

	scores.close()

	print(p_bar, r_bar)
	print(unknown)
	return p_bar, r_bar


def get_cluster(word, clusters, word2cluster):
    """
    Get the entire cluster associated with the given word (helper function for kmeans).
    
    Params:
        word (string): the word to find the cluster of
        clusters (dict): cluster index (int) to cluster (set/list) mapping
        word2cluster (dict): word (string) to cluster index (int) mapping
    
    Returns:
        cluster (set/list) or error message (if word not in vocab)
    """
    try:
        return clusters[word2cluster[word]]
    except KeyError:
        print("Word \"{}\" not seen in dataset".format(word))


def agglom(vocab, affinity="cosine", linkage="average", num_clusters=900, file_num=0) :
	print("loading bert")
	bert = load_bert()

	print("berting vocab")
	embeds, words  = [], []
	len_vocab = len(vocab)
	for v in vocab :
		embeds.append(torch.mean(torch.stack(bert[v]), dim=0))
		words.append(v)

	### CLUSTERING #############################################################
	print("clustering")
	clusters = AgglomerativeClustering(n_clusters=num_clusters, affinity=affinity, linkage=linkage).fit(embeds)

	print("enumerating")
	cluster_dict = { i : [] for i in range(num_clusters) }
	word_to_cluster = {}

	# for i, v in enumerate(words):
	# 	cluster_dict[clusters[i]].append(v)
	# 	word_to_cluster[v] = clusters[i]

	for x in range(len(embeds)) :
		cluster_dict[clusters.labels_[x]].append(words[x])
		word_to_cluster[words[x]] = clusters.labels_[x]

	for c in cluster_dict :
		cluster_dict[c] = set(cluster_dict[c])

	print("pickling")
	with open("../data/bert_agglom_clusters_{}.pkl".format(file_num), "wb") as p :
		pkl.dump(cluster_dict, p)

	############################################################################

	# write individual precision and recall scores to text file
	f = open("../data/bert_agglom_{}.txt".format(file_num), "w")
	f.write("vocab\tprecision\trecall\n")

	precision, recall = [], [] # precision and recall for each vocab
	pre, rec = { i : [] for i in range(num_clusters)}, { i : [] for i in range(num_clusters)} # cluster to score mapping
	count = 0 # print for sanity check
	unknown = 0

	for w in words :
		p, r = 0.0, 0.0

		cluster = get_cluster(w, cluster_dict, word_to_cluster)

		# accumulate gold cluster for v with WordNet
		gold = []
		for syn in wordnet.synsets(w) :
			for l in syn.lemmas() :
				gold.append(l.name())
		gold = set(gold)
		if len(gold) == 0 :
			unknown += 1
			continue

		# cluster.add(lemmatizer.lemmatize(w))
		gold.add(w)

		intersection = cluster.intersection(gold) # true positive


		p = len(intersection) / (len(intersection) + len(cluster.difference(gold)))
		r = len(intersection) / (len(intersection) + len(gold.difference(cluster)))

		# if p == 0 or r == 0:
		# 	print(w, lemmatizer.lemmatize(w))
		# 	print(cluster)
		# 	print(gold)

		
		# try:
		# 	p = len(intersection) / (len(intersection) + len(cluster.difference(gold)))
		# except:
		# 	continue
		# try:
		# 	r = len(intersection) / (len(intersection) + len(gold.difference(cluster)))
		# except:
		# 	continue

		f.write("{}\t{}\t{}\n".format(w, p, r))

		count += 1
		if count % 10 == 0 :
			print("{}/{}".format(count, len_vocab))
			print(p, r)


		precision.append(p)
		recall.append(r)
		pre[word_to_cluster[w]].append(p)
		rec[word_to_cluster[w]].append(r)

	p_bar, r_bar = np.mean(precision), np.mean(recall)

	f.write("\naverage\t{}\t{}\n".format(p_bar, r_bar))
	f.close()

	scores = open("../data/bert_agglom_scores_{}.txt".format(file_num), "w")
	scores.write("cluster\tprecision\trecall\n")
	for i in range(num_clusters) :
		scores.write("{}\t{}\t{}\n".format(i, np.mean(pre[i]), np.mean(rec[i])))

	scores.close()


	print(len_vocab)
	print(p_bar, r_bar)
	print(unknown)
	return p_bar, r_bar


if __name__ == "__main__" :
	# get_embeddings2()

	method, num = sys.argv[1], sys.argv[2]
	if method == "kmeans" :
		kmeans(get_brown_vocab(), k=900, r=25, file_num=num)
	elif method == "agglom" :
		agglom(get_brown_vocab(), num_clusters=900, file_num=num)


