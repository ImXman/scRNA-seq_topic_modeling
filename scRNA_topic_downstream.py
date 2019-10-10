# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 03:37:16 2019

@author: Yang Xu

Topic modeling for analyzing single cell RNA-seq data
"""

import pandas as pd
import numpy as np

##topic modeling
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF#,LatentDirichletAllocation

##visualization
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

##clustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

##statistics
from scipy import stats

##plotting
import seaborn as sns
import matplotlib.pyplot as plt

##-----------------------------------------------------------------------------
##visualize genes after dimension reduction
'''
pca = PCA(n_components=2)
embedding = pca.fit_transform(new_w)
embedding = pd.DataFrame(embedding)
embedding.columns=['PC1','PC2']
f=sns.lmplot(x='PC1', y='PC2',data=embedding,
           fit_reg=False,legend=False,scatter_kws={"s": 5})
f.savefig("pca_gene",dpi=300)

embedding = TSNE(n_components=2).fit_transform(new_w)
embedding = pd.DataFrame(embedding)
embedding.columns=['tSNE1','tSNE2']
sns.lmplot(x='tSNE1', y='tSNE2',data=embedding,
           fit_reg=False,legend=False,scatter_kws={"s": 5,"color":"red"})
plt.savefig("tsne_gene",dpi=300)
'''

embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(new_w)
embedding = pd.DataFrame(embedding)
embedding.columns=['UMAP1','UMAP2']
sns.lmplot(x='UMAP1', y='UMAP2',data=embedding,
           fit_reg=False,legend=False,scatter_kws={"s": 5,"color":"green"})
plt.savefig("UMAP_gene",dpi=300)

##-----------------------------------------------------------------------------
##density based gene clustering
for k in range(1,100):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(embedding.iloc[:,:2])
    y_pred = kmeans.labels_
    cluster_size={}
    for i in list(set(y_pred)):
        cluster_size[i]=y_pred.tolist().count(i)
    if min(cluster_size.values()) <50:
        break

embedding["Cluster"]=y_pred
f=sns.lmplot(x='UMAP1', y='UMAP2',data=embedding,hue="Cluster",
           fit_reg=False,legend=False,scatter_kws={'s':5})
for i in list(set(y_pred)):
    plt.annotate(i, 
                 embedding.loc[embedding['Cluster']==i,['UMAP1','UMAP2']].mean(),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=15, weight='bold')
 
f.savefig("Gene_Cluster",dpi=450)

##-----------------------------------------------------------------------------
##Gene cluster topic
new_w=pd.DataFrame(new_w)
new_w['Cluster']=y_pred
new_w = new_w.sort_values("Cluster",ascending=True)
gene_cluster = new_w['Cluster']
gene_cluster_topic = new_w.groupby(['Cluster']).mean()

f=sns.clustermap(gene_cluster_topic.iloc[:,1:],method='ward', 
                 metric='euclidean',cmap="Blues",row_cluster=False)
f.savefig("gene_cluster_topic",dpi=450)