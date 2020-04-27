# eval.py

# datasets
_aae = None
_aae_vocab = None
_gv = None
_gv_vocab = None

# embeddings
_glove_50 = {}

GLOVE_50_DIR = "../glove.twitter.27B/glove.twitter.27B.50d.txt"

# load datasets (GV and AAE)
def load_aae(path="../data/"):
	_aae = pkl.load(os.join(path, "aae"))
	_aae_vocab = pkl.load(os.join(path, "aae"))


def load_gv(path="../data/"):
	_gv = pkl.load(os.join(path, "gang_violence"))
	_gv_vocab = pkl.load(os.join(path, "gv"))


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

	# ! DO SOMETHING WITH DATAFRAMES THAT I DON'T HAVE THE BRAINPOWER TO DEAL WITH RIGHT NOW

	precision, recall = [], [] # precision and recall for each word
	pre, rec = { i : [] for i in range(k)}, { i : [] for i in range(k)} # cluster to score mapping
	count = 0 # print for sanity check
	unknown = 0
	for w in words :
		p, r = 0.0, 0.0

		cluster = get_cluster(w, cluster_dict, word_to_cluster)

		something = eval_cn(cluster)

		count += 1
		if count % 10 == 0 :
			print(count)

		# ! DATAFRAMES DATAFRAMES DATAFRAMES

	p_bar, r_bar = np.mean(precision), np.mean(recall)


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


def eval_wn(cluster):
	pass

# eval with WN

# output precision, recall, unknown as csv

# pickle clusters

if __name__ == "__main__":
	pass