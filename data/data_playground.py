# import pickle as pkl
# from sklearn.cluster import KMeans
import sys
import matplotlib.pyplot as plt
import pickle as pkl

'''
with open("kmeans_clusters.pkl", "rb") as k:
	clusters = pkl.load(k)

print(clusters)
'''

if __name__ == "__main__" :
	sizes = []
	for x in range(55):
		try:
			with open("kmeans_clusters_aae_{}.pkl".format(x), "rb") as infile:
				cluster_file = pkl.load(infile)
			for cluster in cluster_file:
				sizes.append(len(cluster_file[cluster]))
			print(x, "works")
		except:
			print(x, "does not work")

	plt.hist(sizes)
	plt.title("Size of Clusters (AAE)")
	plt.xlabel("Number of Words Per Cluster")
	plt.ylabel("Number of Clusters")
	plt.savefig("aae_hist.png")
		



'''
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
'''		
