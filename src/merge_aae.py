# merge_aae.py
import pandas as pd
import pickle as pkl

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
			with open("aae_words_glove_{}.csv".format(x), "rb") as infile:
				words = pkl.load(infile)
			words_dfs.append(create_missing_words_df(words))
			print(x, "works")
		except:
			print(x, "does not work")

	words = pd.concat(words_dfs, axis=0)
	words.to_csv("aae_words.csv")

	