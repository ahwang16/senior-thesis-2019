# prelim_eval_w2v_cn.py

# prelim_eval_w2v.py

from gensim.models import Word2Vec

from nltk.cluster import KMeansClusterer
from nltk.cluster.util import cosine_distance

from nltk.corpus import brown
from nltk.corpus import wordnet

import numpy as np
import pickle as pkl
from random import randint
import sys

from sklearn.cluster import AgglomerativeClustering

with open("../data/brown_cn_new.pkl", "rb") as infile:
		_cn = pkl.load(infile)


def kmeans(embeds, vocab, k=900, r=25, file_num=0) :
	'''
	Cluster and evaluate kmeans algorithm.

	Params:
		embeds: word2vec
		vocab: vocab from corpus
	'''

	### CLUSTER ################################################################
	print("clustering")
	clusterer = KMeansClusterer(k, distance=cosine_distance, repeats=r)
	clusters = clusterer.cluster(embeds, assign_clusters=True)

	print("enumerating")
	cluster_dict = { i : [] for i in range(k) }
	word_to_cluster = {}

	for i, v in enumerate(vocab):
		cluster_dict[clusters[i]].append(v)
		word_to_cluster[v] = clusters[i]

	for c in cluster_dict :
		cluster_dict[c] = set(cluster_dict[c])

	print("pickling")
	with open("../data/kmeans_clusters_w2v_cn_{}.pkl".format(file_num), "wb") as p :
		pkl.dump(cluster_dict, p)

	############################################################################

	# write individual precision and recall scores to text file
	f = open("../data/kmeans_w2v_cn_{}.txt".format(file_num), "w")
	f.write("vocab\tprecision\trecall\n")

	precision, recall = [], [] # precision and recall for each word
	pre, rec = { i : [] for i in range(k)}, { i : [] for i in range(k)} # cluster to score mapping
	count = 0 # print for sanity check
	unknown = 0
	len_vocab = len(vocab)
	for w in vocab :
		p, r = 0.0, 0.0

		cluster = get_cluster(w, cluster_dict, word_to_cluster)

		# accumulate gold cluster for v with WordNet
		gold = set()
		try:
			gold = _cn[w]
		except:
			unknown += 1

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
			print("{}/{}".format(count, len_vocab))
			print(len(intersection), len(gold), len(cluster))
			print(gold)
			print(cluster)
			print(p, r)


		precision.append(p)
		recall.append(r)
		pre[word_to_cluster[w]].append(p)
		rec[word_to_cluster[w]].append(r)

	p_bar, r_bar = np.mean(precision), np.mean(recall)

	f.write("\naverage\t{}\t{}\n".format(p_bar, r_bar))
	f.close()

	scores = open("../data/kmeans_scores_w2v_cn_{}.txt".format(file_num), "w")
	scores.write("cluster\tprecision\trecall\n")
	for i in range(k) :
		scores.write("{}\t{}\t{}\n".format(i, np.mean(pre[i]), np.mean(rec[i])))

	scores.close()

	print(len_vocab)
	print(p_bar, r_bar)
	print(unknown)
	return p_bar, r_bar


# embeds: w2v, vocab: from corpus
def agglom(embeds, vocab, affinity="cosine", linkage="average", num_clusters=900, file_num=0) :
	### CLUSTERING #############################################################
	print("clustering")
	clusters = AgglomerativeClustering(n_clusters=num_clusters, affinity=affinity, linkage=linkage).fit(embeds)

	print("enumerating")
	cluster_dict = { i : [] for i in range(num_clusters) }
	word_to_cluster = {}

	for x in range(len(embeds)) :
		cluster_dict[clusters.labels_[x]].append(vocab[x])
		word_to_cluster[vocab[x]] = clusters.labels_[x]

	for c in cluster_dict :
		cluster_dict[c] = set(cluster_dict[c])

	print("pickling")
	with open("../data/agglom_clusters_w2v_cn_{}.pkl".format(file_num), "wb") as p :
		pkl.dump(cluster_dict, p)

	############################################################################

	# write individual precision and recall scores to text file
	f = open("../data/agglom_w2v_cn_{}.txt".format(file_num), "w")
	f.write("vocab\tprecision\trecall\n")

	precision, recall = [], [] # precision and recall for each vocab
	pre, rec = { i : [] for i in range(num_clusters)}, { i : [] for i in range(num_clusters)} # cluster to score mapping
	count = 0 # print for sanity check
	len_vocab = len(vocab)
	unknown = 0
	
	print("evaluating")

	for w in vocab :
		p, r = 0.0, 0.0

		cluster = get_cluster(w, cluster_dict, word_to_cluster)

		# accumulate gold cluster for v with WordNet
		gold = set()
		try:
			gold = _cn[w]
		except:
			unknown += 1

		gold.add(w)


		intersection = cluster.intersection(gold) # true positive

		p = len(intersection) / (len(intersection) + len(cluster.difference(gold)))
		r = len(intersection) / (len(intersection) + len(gold.difference(cluster)))

		f.write("{}\t{}\t{}\n".format(w, p, r))

		count += 1
		if count % 10 == 0 :
			print("{}/{}".format(count, len_vocab))
			print(len(intersection), len(gold), len(cluster))
			print(gold)
			print(cluster)
			print(p, r)


		precision.append(p)
		recall.append(r)
		pre[word_to_cluster[w]].append(p)
		rec[word_to_cluster[w]].append(r)

	p_bar, r_bar = np.mean(precision), np.mean(recall)

	f.write("\naverage\t{}\t{}\n".format(p_bar, r_bar))
	f.close()

	scores = open("../data/agglom_scores_w2v_cn_{}.txt".format(file_num), "w")
	scores.write("cluster\tprecision\trecall\n")
	for i in range(num_clusters) :
		scores.write("{}\t{}\t{}\n".format(i, np.mean(pre[i]), np.mean(rec[i])))

	scores.close()

	print(unknown)
	print(p_bar, r_bar)
	return p_bar, r_bar


def cluster(params, file_params) :
	pass


def eval() :
	pass


### HELPER METHODS #############################################################
def get_cluster(word, clusters, word2cluster) :
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


def load_w2v() :
	return Word2Vec(brown.sents(categories=['fiction']), min_count=1)


def get_brown_vocab() :
	return set(brown.words(categories=['fiction']))


if __name__ == "__main__" :
	w2v = load_w2v()
	vocab = list(get_brown_vocab())

	# arguments: clustering method, file_num
	# don't need to run browns or random again because they don't depend on embeddings
	method, num = sys.argv[1], sys.argv[2]

	if method == "kmeans" :
		kmeans(w2v.wv[w2v.wv.vocab], vocab, k=900, r=25, file_num=num)
	elif method == "agglom" :
		agglom(w2v.wv[w2v.wv.vocab], vocab, num_clusters=900, file_num=num)
	else :
		print("Usage: kmeans/agglom filenum")


