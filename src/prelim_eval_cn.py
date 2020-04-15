# prelim_eval_cn.py

import sys, os, requests, time, json
from IPython import embed
from nltk.corpus import brown

# CORPORA_PATH = '/Users/alyssahwang/Documents/workspace2/seniorthesis/data'
CORPORA_PATH = "../data/"
GLOVE_50_DIR = "../glove.twitter.27B/glove.twitter.27B.50d.txt"

_glove_50 = {}
_cn = {}


def check_and_sleep(start_time, curr_call_ct):
	if curr_call_ct % 120 == 0:  # every 120 sleep for a minute
		print('short: {}'.format(curr_call_ct))
		time.sleep(60)

	if curr_call_ct % 3600 == 0:
		print("long sleep: {}".format(curr_call_ct))
		curr_time = time.time()
		d = max(60 * 60 - (curr_time - start_time) + 2, 2)
		print("  d={}".format(d))
		time.sleep(d)
		start_time = time.time()
	return start_time


def query_conceptnet(w, rel, curr_call_ct, start_time):
	rel_nodes = set()
	next_pg = run_query(w, rel, rel_nodes, '/c/en/{}?offset=0&limit=1000'.format(w))
	curr_call_ct += 1

	while next_pg is not None:
		start_time = check_and_sleep(start_time, curr_call_ct)
		next_pg = run_query(w, rel, rel_nodes, next_pg)
		curr_call_ct += 1

	return rel_nodes, curr_call_ct, start_time


def run_query(w, rel, rel_nodes, page_info):
	obj = requests.get('http://api.conceptnet.io{}'.format(page_info))
	try:
		obj = obj.json()
	except:
		print(obj)
	edges = obj['edges']
	for e in edges:
		if e['rel']['label'] != rel: continue
		if e['start']['label'] != w and e['start']['@id'].startswith('/c/en/'):
			if " " not in e['start']['label'] :
				rel_nodes.add(e['start']['label'])
		elif e['end']['label'] != w and e['end']['@id'].startswith('/c/en/'):
			if " " not in e['end']['label'] :
				rel_nodes.add(e['end']['label'])

	if 'view' in obj.keys():
		return obj['view'].get('nextPage', None)
	return None


def get_related(vocab_name, outname):
	print("sleeping for 1 hour so starting fresh with request limits")
	time.sleep(60*60)

	print("starting requests")

	outf = open(os.path.join(CORPORA_PATH, outname), 'w', encoding='utf-8')
	outf.write('word,related_lst\n')
	# outf = open(os.path.join(CORPORA_PATH, outname), 'a', encoding='utf-8')

	# f = open(os.path.join(CORPORA_PATH, vocab_name), 'r', encoding='utf-8')
	# lines = f.readlines()
	# print("need {} words".format(len(lines)))
	start_time = time.time()
	i = 0
	count = 0
	for w in vocab_name:
		if count % 100 == 0 : print(count)
		count += 1

		start_time = check_and_sleep(start_time, i)

		# start_time = time.time()

		# w = l.strip()
		r, i, start_time = query_conceptnet(w, 'RelatedTo', i, start_time)
		outf.write('{},{}\n'.format(w, json.dumps(list(r))))


def load_bert(path="../data/bert_embeddings.pkl"):
	with open(path, "rb") as f:
		return pkl.load(f)


def load_cn():
	cn = {}
	with open("../data/brown_cn_gold_1.txt", "r") as infile:
		next(infile)
		for line in infile:
			l = line.split(',')
			print(l)
			if l[0] == "": continue
			cn[l[0]] = set(json.loads(", ".join(l[1:])))

	return cn


def load_glove(dir=GLOVE_50_DIR) :
	with open(dir, "r") as glove_file:
	    for line in glove_file:
	        l = line.split()
	        _glove_50[l[0]] = np.asarray(l[1:], dtype="float32")


def load_w2v() :
	return Word2Vec(brown.sents(categories=['fiction']), min_count=1)


def get_brown_vocab() :
	return set(brown.words(categories=['fiction']))


def kmeans(vocab, embed_type, k=900, r=25, file_num=0) :
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
	if embed_type == "glove":
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

	elif embed_type == "w2v":
		w2v = load_w2v()
		embeds = w2v.wv[w2v.wv.vocab]

	elif embed_type == "bert":
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
	with open("../data/kmeans_clusters_cn_{}_{}.pkl".format(embed_type, file_num), "wb") as p :
		pkl.dump(cluster_dict, p)

	############################################################################

	# write individual precision and recall scores to text file
	f = open("../data/kmeans_cn_{}_{}.txt".format(embed_type, file_num), "w")
	f.write("vocab\tprecision\trecall\n")

	precision, recall = [], [] # precision and recall for each word
	pre, rec = { i : [] for i in range(k)}, { i : [] for i in range(k)} # cluster to score mapping
	count = 0 # print for sanity check
	unknown = 0
	for w in words :
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
			rint("{}/{}".format(count, len_vocab))
			print(p, r)


		precision.append(p)
		recall.append(r)
		pre[word_to_cluster[w]].append(p)
		rec[word_to_cluster[w]].append(r)

	p_bar, r_bar = np.mean(precision), np.mean(recall)

	f.write("\naverage\t{}\t{}\n".format(p_bar, r_bar))
	f.close()

	scores = open("../data/kmeans_scores_cn_{}_{}.txt".format(embed_type, file_num), "w")
	scores.write("cluster\tprecision\trecall\n")
	for i in range(k) :
		scores.write("{}\t{}\t{}\n".format(i, np.mean(pre[i]), np.mean(rec[i])))

	scores.close()

	print(p_bar, r_bar)
	print(unknown)
	return p_bar, r_bar


def agglom(vocab, embed_type, affinity="cosine", linkage="average", num_clusters=900, file_num=0) :
	if embed_type == "glove":
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

	elif embed_type == "w2v":
		w2v = load_w2v()
		embeds = w2v.wv[w2v.wv.vocab]

	elif embed_type == "bert":
		print("loading bert")
		bert = load_bert()

		# bert word embeddings
		print("berting vocab")
		embeds, words = [], []
		len_vocab = len(vocab)
		for v in vocab:
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
	with open("../data/agglom_clusters_cn_{}_{}.pkl".format(embed_type, file_num), "wb") as p :
		pkl.dump(cluster_dict, p)

	############################################################################

	# write individual precision and recall scores to text file
	f = open("../data/agglom_cn_{}_{}.txt".format(embed_type, file_num), "w")
	f.write("vocab\tprecision\trecall\n")

	precision, recall = [], [] # precision and recall for each vocab
	pre, rec = { i : [] for i in range(num_clusters)}, { i : [] for i in range(num_clusters)} # cluster to score mapping
	count = 0 # print for sanity check
	unknown = 0

	for w in words :
		p, r = 0.0, 0.0

		cluster = get_cluster(w, cluster_dict, word_to_cluster)

		# accumulate gold cluster for v from ConceptNet dictionary
		gold = set()
		try:
			gold = _cn[w]
		except:
			unknown += 1

		# cluster.add(lemmatizer.lemmatize(w))
		gold.add(w)

		intersection = cluster.intersection(gold) # true positive


		p = len(intersection) / (len(intersection) + len(cluster.difference(gold)))
		r = len(intersection) / (len(intersection) + len(gold.difference(cluster)))

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

	scores = open("../data/agglom_scores_cn_{}_{}.txt".format(embed_type, file_num), "w")
	scores.write("cluster\tprecision\trecall\n")
	for i in range(num_clusters) :
		scores.write("{}\t{}\t{}\n".format(i, np.mean(pre[i]), np.mean(rec[i])))

	scores.close()


	print(missing, len_vocab)
	print(p_bar, r_bar)
	print(unknown)
	return p_bar, r_bar


if __name__ == "__main__":
	# vocab = set(brown.words(categories=["fiction"]))
	# print(len(vocab))
	# get_related(vocab, "brown_cn_gold_1.txt")

	_cn = load_cn()

	method, embed_type, num = sys.argv[1], sys.argv[2], sys.argv[3]
	if method == "kmeans" :
		kmeans(get_brown_vocab(), embed_type, k=900, r=25, file_num=num)
	elif method == "agglom" :
		agglom(get_brown_vocab(), embed_type, num_clusters=900, file_num=num)

