import pickle as pkl
from sklearn.cluster import KMeans


with open("kmeans_clusters.pkl", "rb") as k:
	clusters = pkl.load(k)

print(clusters)
