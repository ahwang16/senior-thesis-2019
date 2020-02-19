# import pickle as pkl
# from sklearn.cluster import KMeans
import sys

'''
with open("kmeans_clusters.pkl", "rb") as k:
	clusters = pkl.load(k)

print(clusters)
'''

if __name__ == "__main__" :
	filename = sys.argv[1]

	tweet = open(sys.argv[2], "w")
	vocab = open(sys.argv[3], "w")

	with open(filename, "r") as conll :
		t = []
		v = []	
		for line in conll :
			if line == "\n" :
				t.append("\n")
				tweet.write(" ".join(t))
				t = []
				continue
			word = line.split("\t")[1]
			t.append(word)
			v.append(word)

	v = set(v)
	for w in v :
		vocab.write(w)
		vocab.write("\n")

	tweet.close()
	vocab.close()
			
