# eval.py
import json
from nltk.cluster import KMeansClusterer
from nltk.cluster.util import cosine_distance
from nltk.corpus import wordnet
import numpy as np
import os
import pandas as pd
import pickle as pkl
from sklearn.cluster import AgglomerativeClustering
import sys

# datasets
_aae_vocab = None
_gv_vocab = None

# embeddings
_glove_50 = {}

_cn = {}

GLOVE_50_DIR = "../glove.twitter.27B/glove.twitter.27B.50d.txt"

# load datasets (GV and AAE)
def load_aae(path="../data/"):
	_aae_vocab = pkl.load(os.path.join(path, "aae_vocab.txt"))


def load_gv(path="../data/"):
	_gv_vocab = pkl.load(os.path.join(path, "gv_vocab.pkl"))


def load_cn(data, path="../data/"):
	with open(os.path.join(path, data)) as infile:
		next(infile)
		for line in infile:
			l = line.split("\t")
			_cn[l[0]] = json.loads(l[1])



# Load 50-dim pretrained GloVe embeddings from text file
def load_glove(dir=GLOVE_50_DIR) :
	with open(dir, "r") as glove_file:
		for line in glove_file:
			l = line.split()
			_glove_50[l[0]] = np.asarray(l[1:], dtype="float32")

# finetune GloVe
# idk


# get embeddings
def get_embeddings(vocab):
	print("loading glove")
	load_glove()

	# get GloVe word embeddings and number of missing words
	print("gloving vocab")
	embeds, words  = [], []
	missing = 0
	for v in vocab :
		try:
			embeds.append(_glove_50[v])
			words.append(v)
		except:
			missing += 1

	return embeds, words, missing


# cluster (kmeans)
def kmeans(vocab, data, k=900, r=25, file_num=0):
	"""
	Cluster glove embeddings with kmeans algorithm

	Params:
		vocab (set): set of all words in dataset
		data (string): dataset name for output file names
		k (int): number of clusters
		r (int): number of repeats
		file_num (int): number for output file names

	Returns:

	"""
	### CLUSTERING #############################################################
	print("clustering")
	embeds, words, missing = get_embeddings(vocab)
	print("missing from glove:", missing)

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
	with open("../data/kmeans_clusters_{}_{}.pkl".format(data, file_num), "wb") as p :
		pkl.dump(cluster_dict, p)


	############################################################################


def eval(path_to_cluster):
	words = []
	words_idx = []
	clusters = []

	with open(path_to_cluster, "rb") as infile:
		kmeans_clusters_cn = pkl.load(infile)

	for cluster_idx in kmeans_clusters_cn :
		precision_wn, recall_wn, precision_cn, recall_cn = [], [], [], []
		cluster = kmeans_clusters_cn[cluster_idx]
		for word in cluster :
			missing_from_wn, missing_from_cn = 0, 0

			gold_wn = get_gold_wn(word)
			try:
				gold_cn = _cn[word]
			except:
				gold_cn = set()
			gold_cn.add(word)

			missing_from_wn += len(gold_wn) == 1
			missing_from_cn += len(gold_cn) == 1
			
			true_positive_wn = len(cluster.intersection(gold_wn))
			false_positive_wn = len(cluster - gold_wn)
			false_negative_wn = len(gold_wn - cluster)
			p_wn = true_positive_wn / (true_positive_wn + false_positive_wn)
			r_wn = true_positive_wn / (true_positive_wn + false_negative_wn)
			precision_wn.append(p_wn)
			recall_wn.append(r_wn)
			
			true_positive_cn = len(cluster.intersection(gold_cn))
			false_positive_cn = len(cluster - gold_cn)
			false_negative_cn = len(gold_cn - cluster)
			p_cn = true_positive_cn / (true_positive_cn + false_positive_cn)
			r_cn = true_positive_cn / (true_positive_cn + false_negative_cn)
			precision_cn.append(p_cn)
			recall_cn.append(r_cn)
			
			words_idx.append(word)
			words.append({"precision_wn" : p_wn, "recall_wn" : r_wn,
						  "precision_cn" : p_cn, "recall_cn" : r_cn,
						  "missing_from_cn" : missing_from_cn,
						  "missing_from_wn" : missing_from_wn})
			
		clusters.append({"precision_wn" : np.mean(precision_wn),
						 "recall_wn" : np.mean(recall_wn),
						 "precision_cn" : np.mean(precision_cn),
						 "recall_cn" : np.mean(recall_cn)})

	pd.DataFrame(words, index=words_idx).to_csv("gv_words.csv")
	pd.DataFrame(clusters).to_csv("gv_clusters.csv")
	

'''
### UNECESSARY FUNCTIONS ######################################################
# eval with CN
def eval_cn(cluster):
	"""
	Given a cluster, compute precision and recall for each word and average for
	entire cluster. Return number of words not in concept net.

	Params:
		cluster (set): set of words
	Returns:
		scores (dict): ...
	"""
	pass

# eval with WN
def eval_wn(cluster):
	words = [] # dictionary of precision and recall values
	words_idx
	precision, recall = [], []

	for word in cluster:
		gold = get_gold_wn(word)

		tp = len(cluster.intersection(gold))
		fp = len(cluster - gold)
		fn = len(gold - cluster)

		precision.append(tp / (tp + fp))
		recall.append(tp / (tp + fn))


def get_gold_wn(word):
	gold = set()
	for syn in wordnet.synsets(word):
		for l in syn.lemmas():
			gold.add(l.name())
	gold.add(word)
	return gold
'''


if __name__ == "__main__":
	data, file_num = sys.argv[1], sys.argv[2]

	if data == "gv":
		print("loading gv")
		load_gv()
		k = len(_gv_vocab) / 10
		print(k)
		print("clustering")
		kmeans(_gv_vocab, data, k=k, file_num=file_num)
	elif data == "aae":
		print("loading aae")
		load_aae()
		k = len(_aae_vocab) / 10
		print(k)
		print("clustering")
		kmeans(_aae_vocab, data, k=k, file_num=file_num)

	print("evaluating")
	load_cn("gv_cn_gold.txt")
	eval("../data/kmeans_clusters_{}_{}.pkl".format(data, file_num))







