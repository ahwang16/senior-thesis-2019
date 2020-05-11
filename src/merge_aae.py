# merge_aae.py
import json
from nltk.corpus import wordnet
import numpy as np
import pandas as pd
import pickle as pkl

glove = {}
with open("../glove.twitter.27B/glove.twitter.27b.50d.txt") as infile:
    for line in infile:
        l = line.split('\t')
        glove[l[0]] = np.asarray(l[1:], dtype="float32")

cn_gold = {}
with open("../data/aae_cn_gold.txt", "r") as infile:
	next(infile)
	for line in infile:
		l = line.split('\t')
		if len(l) == 2:
			cn_gold[l[0]] = json.loads(l[1])


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
	    missing_cn = int(len(cn_gold[word]) == 0)
	    missing_wn = int(len(get_gold_wn(word)) == 1)
	    
	    missing.append({
	        "missing_glove" : missing_glove,
	        "missing_cn" : missing_cn,
	        "missing_wn":  missing_wn
	    })
	    
	brown_missing = pd.DataFrame(missing, index=word_idx)


if __name__ == "__main__":
	words_dfs = []
	for x in range(55):
		try:
			df = pd.read_csv("aae_words_glove_{}.csv".format(x))
			words_dfs.append(create_missing_words_df(df))
			print(x, "works")
		except:
			print(x, "does not work")

	words = pd.concat(words_dfs, axis=0)
	words.to_csv("aae_words.csv")

