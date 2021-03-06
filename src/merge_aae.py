# merge_aae.py
import json
from nltk.corpus import wordnet
import numpy as np
import pandas as pd
import pickle as pkl

# glove = {}
# with open("../glove.twitter.27B/glove.twitter.27B.50d.txt") as infile:
# 	for line in infile:
# 		l = line.split('\t')
# 		glove[l[0]] = np.asarray(l[1:], dtype="float32")

# cn_gold = {}
# with open("../data/aae_cn_gold.txt", "r") as infile:
# 	next(infile)
# 	for line in infile:
# 		l = line.split('\t')
# 		if len(l) == 2:
# 			cn_gold[l[0]] = json.loads(l[1])


def get_gold_wn(word):
	gold = set()
	for syn in wordnet.synsets(word):
		for l in syn.lemmas():
			gold.add(l.name())
	gold.add(word)
	return gold


def create_missing_words_df(df):
	word_idx = []
	missing = []
	for word in df["Unnamed: 0"]:
		word_idx.append(word)
		missing_glove = int(word not in glove)
		try:
			missing_cn = int(len(cn_gold[word]) == 0)
		except:
			missing_cn = 1
		missing_wn = int(len(get_gold_wn(word)) == 1)
		
		missing.append({
			"missing_glove" : missing_glove,
			"missing_cn" : missing_cn,
			"missing_wn":  missing_wn
		})
		
	return pd.DataFrame(missing, index=word_idx)


if __name__ == "__main__":
	words_dfs = []
	for x in range(55):
		try:
			df = pd.read_csv("aae_words_glove_{}.csv".format(x))
			word_dfs.append(pd.merge(df, create_missing_words_df(df), left_on="Unnamed: 0", right_on="Unnamed: 0"))
			print(x, "works")
		except:
			print(x, "does not work")

	words = pd.concat(words_dfs, axis=0)
	words.to_csv("aae_words.csv")

	# clusters_dfs = []
	# for x in range(55):
	# 	try:
	# 		clusters_dfs.append(pd.read_csv("aae_clusters_glove_{}.csv".format(x)))
	# 		print(x, "works")
	# 	except:
	# 		print(x, "does not work")

	# clusters = pd.concat(clusters_dfs, axis=0)
	# clusters.to_csv("aae_clusters.csv")

