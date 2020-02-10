# File for preliminary evaluations of Brown's, agglomerative, and k-means
# clustering algorithms with bigrams and W2V, GloVe, and BERT embeddings.
# Evaluate with WordNet as gold standard.
# For word w in cluster c, check if every other word is in the synset for w.
# Report correctly clustered words / total words in cluster c for w.
# Also report correctly clustered words / total words in synset for w.
import sys
sys.path.append("../brown_clustering")

from bert_embedding import BertEmbedding

from brownclustering import Corpus
from brownclustering import BrownClustering
from brownclustering.validation import BrownClusteringValidation

from nltk.cluster import KMeansClusterer
from nltk.cluster.util import cosine_distance

from nltk.corpus import brown
from nltk.corpus import wordnet

import numpy as np
import pickle as pkl

from sklearn.cluster import AgglomerativeClustering


# Seed vocabulary set
# 50 most frequent English adjectives https://www.ef.com/wwen/english-resources/english-vocabulary/top-50-adjectives/
# 25 most frequent English nouns https://www.englishclub.com/vocabulary/common-nouns-25.htm
# 65 common English nouns and adjectives
SEED_VOCAB = [
	"able", "bad", "best", "better", "big", "certain", "clear",
	"different", "early", "easy", "economic", "free", "full", "good",
	"great", "hard", "high", "important", "large", "late", "little", "local",
	"long", "low", "major", "new", "old", "only", "other", "possible",
	"recent", "right", "small", "social", "special", "strong", "sure", "true",
	"whole", "young", "time", "person", "year", "way", "day", "thing",
	"man", "world", "life", "hand", "part", "child", "eye", "woman", "place",
	"work", "week", "case", "point", "government", "company", "number",
	"group", "problem", "fact"
]

_full_vocab = set()

# _gold_clusters = []

_glove_50 = {} # all pretrained GloVe embeddings will be loaded here
_glove_50_vocab = set() # GloVe embeddings for full vocab
GLOVE_50_DIR = "../data/glove.twitter.27B.50d.txt"


# **************************************************************************** #

# Generate full vocabulary set by adding members of synsets for each SEED_VOCAB
# Use the first synset for each word (since synsets are ordered by frequency
# of use)
# input: list/set of strings (words)
# output: set of all original vocab + members of first synsets
def get_full_vocab(vocab=SEED_VOCAB) :
	"""
	Generate full vocabulary set by adding members of synsets for each
	SEED_VOCAB. Use the first synset for each word (since synsets are ordered
	by frequency of use)

	Params:
		vocab (list/set of strings): starting vocabulary

	Returns:
		v (set of strings): expanded vocabulary
	"""
	v = set()

	for word in SEED_VOCAB :
		for w in wordnet.synsets(word)[0].lemmas():
			if "_" not in w.name() :
				v.add(w.name())

	return v


# Load 50-dim pretrained GloVe embeddings from text file
def load_glove(dir=GLOVE_50_DIR) :
	with open(dir, "r") as glove_file:
	    for line in glove_file:
	        l = line.split()
	        GLOVE_50[l[0]] = np.asarray(l[1:], dtype="float32")


# Retrieve GloVe embeddings for given words
def get_glove_vocab(vocab=_full_vocab) :
	for v in vocab :
		try :
			_glove_50_vocab.add(_glove_50[v])
		except :
			pass


def browns(debug=True, num_clusters=25) :
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
	clustering.train()

	precision, recall = [], [] # precision and recall for each vocab

	# write individual precision and recall scores to text file
	f = open("../data/browns.txt", "w")
	f.write("vocab\tprecision\trecall\n")

	# iterate through vocabulary to find synonym sets through WordNet
	for v in clustering.vocabulary :
		p, r = 0.0, 0.0

		# Brown's implementation gives clusters of (word, cluster_id) tuples
		cluster = set(c[0] for c in clustering.get_similar(v, cap=1000))

		# accumulate gold cluster for v with WordNet
		gold = []
		for syn in wordnet.synsets(v) :
			for l in syn.lemmas() :
				gold.append(l.name())
		gold = set(gold)

		intersection = cluster.intersection(gold) # true positive

		p = len(intersection) / (len(intersection) + len(cluster.difference(gold)))
		r = len(intersection) / (len(intersection) + len(gold.difference(cluster)))

		f.write("{}\t{}\t{}\n".format(v, p, r))


		precision.append(p)
		recall.append(r)


	pre, rec = np.mean(precision), np.mean(recall)

	f.write("\naverage\t{}\t{}\n".format(pre, rec))

	return pre, rec


def kmeans(embeds, vocab=_full_vocab, k=20, r=25) :
	"""
	Use k-means clustering with cosine similarity as the distance metric to
	cluster the data into k groups.
	    
    Params:
        k (int): number of clusters
        data (ndarray): word embeddings of vocabulary
        vocab (list/dict): text vocabulary of dataset
        r (int): number of randominzed cluster trials (optinal parameter for KMeansClusterer)
        
    Returns:
        cluster_dict (dict): cluster index (int) mapping to cluster (set)
        word_to_cluster (dict): vocab index mapping word (string) to cluster number (int)
	"""
	clusterer = KMeansClusterer(k, distance=cosine_distance, repeats=r)
	clusters = clusterer.cluster(data, assign_clusters=True)

	cluster_dict = { i : set() for i in range(k) }
	word_to_cluster = {}

	for i, v in enumerate(vocab):
		cluster_dict[clusters[i]].add(v)
		word_to_cluster[v] = clusters[i]

	pass


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


def agglom(embeds, linkage="ward", n_clusters=20) :
	e = list(embeds)
	clustering = AgglomerativeClustering().fit(e, linkage=linkage, n_clusters=n_clusters)

	precision, recall = 0.0, 0.0
	pass


def random_cluster(vocab) :
	"""
	Random baseline for clustering precision and recall

	Params:
		vocab (list of strings): words to cluster together

	Returns:
		precision (float): percen
	"""
	pass


def eval(clusters):
	pass



if __name__ == "__main__" :
	# print(get_full_vocab(["big", "small", "good"]))
	print(browns(debug=False, num_clusters=25))
	# load_glove()
