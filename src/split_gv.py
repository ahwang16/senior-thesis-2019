# split_gv.py

import eval
import pickle
from multiprocessing import Process

if __name__ == "__main__":
	for x in range(55):
		print(x)
		with open("../data/aae_vocab_{}.pkl".format(x), "rb") as infile:
			aae_vocab = pkl.load(infile)
		p = Process(target=eval.kmeans, args=(aae_vocab, "aae", "glove", x))
		print("starting", x)
		p.start()