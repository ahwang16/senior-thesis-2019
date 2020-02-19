# File for preliminary evaluations of Brown's, agglomerative, and k-means
# clustering algorithms with bigrams and W2V, GloVe, and BERT embeddings.
# Evaluate with WordNet as gold standard.
# For word w in cluster c, check if every other word is in the synset for w.
# Report correctly clustered words / total words in cluster c for w.
# Also report correctly clustered words / total words in synset for w.
import sys
sys.path.append("../brown_clustering")

# from bert_embedding import BertEmbedding

from brownclustering import Corpus
from brownclustering import BrownClustering
from brownclustering.validation import BrownClusteringValidation

from nltk.cluster import KMeansClusterer
from nltk.cluster.util import cosine_distance

from nltk.corpus import brown
from nltk.corpus import wordnet

from collections import defaultdict
import numpy as np
import pickle as pkl
from random import randint

from sklearn.cluster import AgglomerativeClustering


# Seed vocabulary set
# 50 most frequent English adjectives https://www.ef.com/wwen/english-resources/english-vocabulary/top-50-adjectives/
# 25 most frequent English nouns https://www.englishclub.com/vocabulary/common-nouns-25.htm
# 65 common English nouns and adjectives
# SEED_VOCAB = [
# 	"able", "bad", "best", "better", "big", "certain", "clear",
# 	"different", "early", "easy", "economic", "free", "full", "good",
# 	"great", "hard", "high", "important", "large", "late", "little", "local",
# 	"long", "low", "major", "new", "old", "only", "other", "possible",
# 	"recent", "right", "small", "social", "special", "strong", "sure", "true",
# 	"whole", "young", "time", "person", "year", "way", "day", "thing",
# 	"man", "world", "life", "hand", "part", "child", "eye", "woman", "place",
# 	"work", "week", "case", "point", "government", "company", "number",
# 	"group", "problem", "fact"
# ]

# _full_vocab = set()

# _gold_clusters = []

_glove_50 = {} # all pretrained GloVe embeddings will be loaded here
# _glove_50_vocab = set() # GloVe embeddings for full vocab
GLOVE_50_DIR = "../glove.twitter.27B/glove.twitter.27B.50d.txt"


# **************************************************************************** #

# Generate full vocabulary set by adding members of synsets for each SEED_VOCAB
# Use the first synset for each word (since synsets are ordered by frequency
# of use)
# input: list/set of strings (words)
# output: set of all original vocab + members of first synsets
# def get_full_vocab(vocab=SEED_VOCAB) :
# 	"""
# 	Generate full vocabulary set by adding members of synsets for each
# 	SEED_VOCAB. Use the first synset for each word (since synsets are ordered
# 	by frequency of use)

# 	Params:
# 		vocab (list/set of strings): starting vocabulary

# 	Returns:
# 		v (set of strings): expanded vocabulary
# 	"""
# 	v = set()

# 	for word in SEED_VOCAB :
# 		for w in wordnet.synsets(word)[0].lemmas():
# 			if "_" not in w.name() :
# 				v.add(w.name())

# 	return v


# Load 50-dim pretrained GloVe embeddings from text file
def load_glove(dir=GLOVE_50_DIR) :
	with open(dir, "r") as glove_file:
	    for line in glove_file:
	        l = line.split()
	        _glove_50[l[0]] = np.asarray(l[1:], dtype="float32")


# Retrieve GloVe embeddings for given words
# def get_glove_vocab(vocab=_full_vocab) :
# 	for v in vocab :
# 		try :
# 			_glove_50_vocab.add(_glove_50[v])
# 		except :
# 			pass


# Get vocabulary from fiction set of Brown corpus
def get_brown_vocab() :
	return set(brown.words(categories=['fiction']))


def browns(debug=True, num_clusters=900, file_num=0) :
	"""
	Train Brown's clustering algorithm on Brown corpus (fiction) and
	calculate precision (number of correct synonyms / total in cluster). Write
	text file of precision and recall scores for each vocab word

	Params:
		num_clusters (int): number of clusters for Brown's algorithm return

	Returns:
		mean(precision) (float): average precision across all clusters
	"""

	# small sample dataset for debugging
	data = [d.split() for d in ["hungry thirsty hello"]]

	if not debug :
		data = brown.sents(categories=["fiction"])

	# Brown's clustering training
	corpus = Corpus(data)
	clustering = BrownClustering(corpus, num_clusters)
	clusters = clustering.train()

	precision, recall = [], [] # precision and recall for each vocab
	# pre, rec = defaultdict(list), defaultdict(list)

	# write individual precision and recall scores to text file
	f = open("../data/browns_{}.txt".format(file_num), "w")
	f.write("vocab\tprecision\trecall\n")
	scores = open("../data/browns_scores_{}.txt".format(file_num), "w")
	scores.write("cluster\tprecision\trecall\n")

	count = 0 # print for sanity check
	len_vocab = len(clustering.vocabulary)

	# iterate through vocabulary to find synonym sets through WordNet
	# for v in clustering.vocabulary :
	cluster_dict = defaultdict(list)

	for cluster in clusters :
		pre_, rec_ = [], []
		cluster_id = 0
		cluster = set(cluster)

		for word in cluster:
			p, r = 0.0, 0.0
			cluster_dict[cluster_id].append(word)

			# Brown's implementation gives clusters of (word, cluster_id) tuples
			# cluster = set(clustering.helper.get_cluster)

			# accumulate gold cluster for v with WordNet
			gold = []
			for syn in wordnet.synsets(word) :
				for l in syn.lemmas() :
					gold.append(l.name())
			gold = set(gold)

			intersection = cluster.intersection(gold) # true positive
			
			try:
				p = len(intersection) / (len(intersection) + len(cluster.difference(gold)))
			except:
				continue
			try:
				r = len(intersection) / (len(intersection) + len(gold.difference(cluster)))
			except:
				continue

			f.write("{}\t{}\t{}\n".format(word, p, r))

			count += 1
			if count % 10 == 0 :
				print("{}/{}".format(count, len_vocab))
				print(p, r)


			precision.append(p)
			recall.append(r)
			pre_.append(p)
			rec_.append(r)

		scores.write("{}\t{}\t{}".format(cluster_id, np.mean(pre_), np.mean(rec_)))
		cluster_id += 1

	pre, rec = np.mean(precision), np.mean(recall)

	f.write("\naverage\t{}\t{}\n".format(pre, rec))
	f.close()
	scores.close()

	with open("browns_clusters_{}.pkl".format(file_num), "wb") as fname :
		pkl.dump(cluster_dict, fname)

	return pre, rec


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
	print("loading glove")
	load_glove()

	# get GloVe word embeddings and number of missing words
	print("gloving vocab")
	embeds, words  = [], []
	missing = 0
	len_vocab = len(vocab)
	for v in vocab :
		try:
			embeds.append(_glove_50[v])
			words.append(v)
		except:
			missing += 1


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
	with open("../data/kmeans_clusters_{}.pkl".format(file_num), "wb") as p :
		pkl.dump(cluster_dict, p)

	############################################################################

	# write individual precision and recall scores to text file
	f = open("../data/kmeans_{}.txt".format(file_num), "w")
	f.write("vocab\tprecision\trecall\n")

	precision, recall = [], [] # precision and recall for each word
	pre, rec = { i : [] for i in range(k)}, { i : [] for i in range(k)} # cluster to score mapping
	count = 0 # print for sanity check
	for w in words :
		p, r = 0.0, 0.0

		cluster = get_cluster(w, cluster_dict, word_to_cluster)

		# accumulate gold cluster for v with WordNet
		gold = []
		for syn in wordnet.synsets(w) :
			for l in syn.lemmas() :
				gold.append(l.name())
		gold = set(gold)

		intersection = cluster.intersection(gold) # true positive
		
		try:
			p = len(intersection) / (len(intersection) + len(cluster.difference(gold)))
		except:
			continue
		try:
			r = len(intersection) / (len(intersection) + len(gold.difference(cluster)))
		except:
			continue

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

	scores = open("../data/kmeans_scores_{}.txt".format(file_num), "w")
	scores.write("cluster\tprecision\trecall\n")
	for i in range(k) :
		scores.write("{}\t{}\t{}\n".format(i, np.mean(pre[i]), np.mean(rec[i])))

	scores.close()


	print(missing, len_vocab)
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
	print("loading glove")
	load_glove()

	print("gloving vocab")
	embeds, words  = [], []
	missing = 0
	len_vocab = len(vocab)
	for v in vocab :
		try:
			embeds.append(_glove_50[v])
			words.append(v)
		except:
			missing += 1

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
	with open("../data/agglom_clusters_{}.pkl".format(file_num), "wb") as p :
		pkl.dump(cluster_dict, p)

	############################################################################

	# write individual precision and recall scores to text file
	f = open("../data/agglom_{}.txt".format(file_num), "w")
	f.write("vocab\tprecision\trecall\n")

	precision, recall = [], [] # precision and recall for each vocab
	pre, rec = { i : [] for i in range(num_clusters)}, { i : [] for i in range(num_clusters)} # cluster to score mapping
	count = 0 # print for sanity check

	for w in words :
		p, r = 0.0, 0.0

		cluster = get_cluster(w, cluster_dict, word_to_cluster)

		# accumulate gold cluster for v with WordNet
		gold = []
		for syn in wordnet.synsets(w) :
			for l in syn.lemmas() :
				gold.append(l.name())
		gold = set(gold)

		intersection = cluster.intersection(gold) # true positive
		
		try:
			p = len(intersection) / (len(intersection) + len(cluster.difference(gold)))
		except:
			continue
		try:
			r = len(intersection) / (len(intersection) + len(gold.difference(cluster)))
		except:
			continue

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

	scores = open("../data/agglom_scores_{}.txt".format(file_num), "w")
	scores.write("cluster\tprecision\trecall\n")
	for i in range(num_clusters) :
		scores.write("{}\t{}\t{}\n".format(i, np.mean(pre[i]), np.mean(rec[i])))

	scores.close()


	print(missing, len_vocab)
	return p_bar, r_bar


def random_cluster(vocab, num_clusters=900, file_num=0) :
	"""
	Random baseline for clustering precision and recall

	Params:
		vocab (list of strings): words to cluster together

	Returns:
		precision (float)
		recall (float)
	"""
	clusters = [[] for i in range(num_clusters)]
	len_vocab = len(vocab)

	print("clustering")
	for v in vocab :
		clusters[randint(0, num_clusters - 1)].append(v)

	print("setting clusters")
	for x in range(len(clusters)) :
		clusters[x] = set(clusters[x])

	precision, recall = [], []

	# write individual precision and recall scores to text file
	f = open("../data/random_{}.txt".format(file_num), "w")
	f.write("vocab\tprecision\trecall\n")

	scores = open("../data/random_scores_{}.txt".format(file_num), "w")
	scores.write("cluster\tprecision\trecall\n")

	count = 0
	for c in clusters :
		p_cluster, r_cluster = [], []
		for w in c :
			p, r = 0.0, 0.0

			# accumulate gold cluster for v with WordNet
			gold = []
			for syn in wordnet.synsets(w) :
				for l in syn.lemmas() :
					gold.append(l.name())
			gold = set(gold)

			intersection = c.intersection(gold) # true positive
		
			try:
				p = len(intersection) / (len(intersection) + len(c.difference(gold)))
			except:
				continue
			try:
				r = len(intersection) / (len(intersection) + len(gold.difference(c)))
			except:
				continue

			p_cluster.append(p)
			r_cluster.append(r)
			f.write("{}\t{}\t{}\n".format(w, p, r))

		scores.write("{}\t{}\t{}".format(count, np.mean(p_cluster), np.mean(r_cluster)))

		count += 1
		if count % 10 == 0 :
			print("{}/{}".format(count, len_vocab))
			print(p, r)

		precision.append(p)
		recall.append(r)

	pre, rec = np.mean(precision), np.mean(recall)

	f.write("\naverage\t{}\t{}\n".format(pre, rec))
	f.close()

	return pre, rec


	


def eval(clusters):
	pass



if __name__ == "__main__" :
	# arguments: clustering method, file_num
	method, num = sys.argv[1], sys.argv[2]
	if method == "random" :
		random_cluster(get_brown_vocab(), num_clusters=900, file_num=num)
	elif method == "browns" :
		browns(debug=False, num_clusters=900, file_num=num)
	elif method == "kmeans" :
		kmeans(get_brown_vocab(), k=900, r=25, file_num=num)
	elif method == "agglom" :
		agglom(get_brown_vocab(), num_clusters=900, file_num=num)

	# print(get_full_vocab(["big", "small", "good"]))
	# print(browns(debug=True, num_clusters=2))
	# load_glove()

	# vocab = get_brown_vocab()
	# print(kmeans(vocab, k=900, r=25))

	# print(kmeans(["hungry", "thirsty", "hello"], k=1, r=1, file_num="test"))

	# print(random_cluster(["hungry", "thirsty", "hello", "goodbye"], num_clusters=2, file_num="test"))
#	random_cluster(get_brown_vocab(), num_clusters=900)

	# print(agglom(["hungry", "thirsty", "hello", "goodbye"], num_clusters=2, file_num="test"))
	# print(agglom(["hungry", "thirsty", "hello", "goodbye"], num_clusters=2))

	# vocab = get_brown_vocab()
	# print(agglom(vocab))



