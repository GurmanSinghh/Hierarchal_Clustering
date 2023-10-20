#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Step 1: Load the Olivetti Faces Dataset
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = data.data, data.target

# Step 2: Split Data into Training, Validation, and Test Sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Step 3: Train a Classifier with K-Fold Cross Validation
clf = SVC(kernel='linear')
K = 5  # Number of folds for cross-validation
scores = cross_val_score(clf, X_train, y_train, cv=K)

# Step 4: Hierarchical Clustering with Different Similarity Measures
max_clusters = 40
similarity_measures = ['euclidean', 'l2', 'cosine']

for similarity_measure in similarity_measures:
    linkage_method = 'ward' if similarity_measure == 'euclidean' else 'average'
    
    clustering = AgglomerativeClustering(n_clusters=max_clusters, linkage=linkage_method, metric=similarity_measure)
    cluster_assignments = clustering.fit_predict(X_train)

    # Step 5: Choose Number of Clusters with Silhouette Score
    silhouette_scores = []
    for num_clusters in range(2, max_clusters):
        clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_method, metric=similarity_measure)
        cluster_assignments = clustering.fit_predict(X_train)
        silhouette_scores.append(silhouette_score(X_train, cluster_assignments))

    optimal_num_clusters = np.argmax(silhouette_scores) + 2  # Adding 2 to account for the range starting from 2

    # Step 6: Train a Classifier on Clustered Data
    cluster_labels = AgglomerativeClustering(n_clusters=optimal_num_clusters, linkage=linkage_method, metric=similarity_measure).fit_predict(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = SVC(kernel='linear')
    cluster_scores = cross_val_score(clf, X_train_scaled, cluster_labels, cv=K)

    # Step 7: Discuss Discrepancies
    print(f'Similarity Measure: {similarity_measure}')
    print(f'Optimal Number of Clusters: {optimal_num_clusters}')
    print(f'Classifier Performance on Clustered Data (Mean Score): {np.mean(cluster_scores)}')


# In[12]:





# In[ ]:




