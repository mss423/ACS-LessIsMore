from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import numpy as np
import pandas as pd
from vertex_embed import get_embeddings_task
from tqdm import tqdm

def kmeans_sample(data_df, Ks):
	embed_data = get_embeddings_task(data_df['sentence'])

	selected_samples = {}
	for K in tqdm(Ks, desc="Processing Ks..."):
		# Find kmeans centers
		kmeans = KMeans(n_clusters=K).fit(embed_data)

		# Loop over all clusters and find index of closest point to the cluster center and append to closest_pt_idx list.
		closest_pt_idx = []
		for iclust in range(kmeans.n_clusters):
			# get all indices of points assigned to this cluster:
			cluster_pts_indices = np.where(kmeans.labels_ == iclust)[0]

			# Handle empty clusters
			if len(cluster_pts_indices) == 0:
				# print(f"Warning: Cluster {iclust} is empty for K={K}. Skipping.")
				continue

			cluster_cen = kmeans.cluster_centers_[iclust]
			min_idx = np.argmin([euclidean(embed_data[idx], cluster_cen) for idx in cluster_pts_indices])
			closest_pt_idx.append(cluster_pts_indices[min_idx])
		selected_samples[K] = closest_pt_idx
		# print(len(selected_samples[K]))
	return selected_samples # data_df.iloc[closest_pt_idx]