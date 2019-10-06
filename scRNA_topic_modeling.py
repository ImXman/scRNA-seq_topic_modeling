# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:13:06 2019

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
from sklearn import cluster
from sklearn.neighbors import kneighbors_graph

##statistics
from scipy import stats

##plotting
import seaborn as sns
import matplotlib.pyplot as plt

##self defined functions
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    
def get_top_topics(model, feature_names, n_top_words):
    message = []
    for topic_idx, topic in enumerate(model.components_):
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            message.append(feature_names[i])
    message = list(dict.fromkeys(message))
    return message

def topic_spec(m1,m2):
    m1.tolist()
    critical = sorted(m1.tolist())[int(len(m1)*0.05)]
    critical = (critical - np.mean(m2))/np.std(m2)
    return stats.norm.cdf(critical)

##-----------------------------------------------------------------------------
##topic modeling
spleen = pd.read_csv("pbmc_sc_umi.txt",header=0,index_col=0,sep="\t")
spl_matrix = spleen.T.values

n_features = 3000
n_top_words = 30

'''
##raw text for LDA
lda = LatentDirichletAllocation(n_components=n_components,
                                learning_method='online',
                                random_state=128)
lda = LatentDirichletAllocation(n_components=n_components, max_iter=10,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

lda.fit(spl_matrix)
topic_matrix = lda.transform(spl_matrix)
w=lda.components_
'''

##NMF
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(spl_matrix)

##test number of topics
difference={}
for i in range(51,101,1):
    n_components = i
    nmf = NMF(n_components=n_components, random_state=42,
              beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
              l1_ratio=.5).fit(tfidf)
    difference[i] = nmf.reconstruction_err_

difference = pd.DataFrame.from_dict(difference)
difference = difference.T
plt.plot(difference.index, difference.iloc[:,0], linestyle='dashed')
plt.xlabel("Number of topics")
plt.ylabel("beta-divergence")
plt.savefig("choice_of_topics",dpi=300)

n_components = 20
nmf = NMF(n_components=n_components, random_state=42,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)
topic_matrix = nmf.transform(tfidf)
w=nmf.components_

#train_topics=get_top_topics(nmf, spleen.index, n_top_words)
#print_top_words(nmf, spleen.index, n_top_words)

##-----------------------------------------------------------------------------
##visualization
pca = PCA(n_components=2)
embedding = pca.fit_transform(topic_matrix)
embedding = pd.DataFrame(embedding)
embedding.columns=['PC1','PC2']
f=sns.lmplot(x='PC1', y='PC2',data=embedding,
           fit_reg=False,legend=False,scatter_kws={"s": 5})
f.savefig("pca",dpi=300)

embedding = TSNE(n_components=2).fit_transform(topic_matrix)
embedding = pd.DataFrame(embedding)
embedding.columns=['tSNE1','tSNE2']
sns.lmplot(x='tSNE1', y='tSNE2',data=embedding,
           fit_reg=False,legend=False,scatter_kws={"s": 5,"color":"red"})
plt.savefig("tsne",dpi=300)

embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(topic_matrix)
embedding = pd.DataFrame(embedding)
embedding.columns=['UMAP1','UMAP2']
sns.lmplot(x='UMAP1', y='UMAP2',data=embedding,
           fit_reg=False,legend=False,scatter_kws={"s": 5,"color":"green"})
plt.savefig("UMAP",dpi=300)

##-----------------------------------------------------------------------------
##clustering
default_base = {'n_neighbors': 15,
                'n_clusters': 15}

params = default_base.copy()
    
connectivity = kneighbors_graph(topic_matrix, 
                                n_neighbors=params['n_neighbors'], 
                                include_self=False)
connectivity = 0.5 * (connectivity + connectivity.T)

ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], 
                                       linkage='ward',connectivity=connectivity)
ward.fit(topic_matrix)
y_pred = ward.labels_

cluster_size={}
for i in range(params['n_clusters']):
    cluster_size[i]=y_pred.tolist().count(i)

clusters = [i for i in cluster_size.keys() if cluster_size[i]>=20]    
#topic_matrix = topic_matrix[y_pred!=6]
#y_pred = y_pred[y_pred!=6]
embedding["Cluster"]=y_pred
f=sns.lmplot(x='UMAP1', y='UMAP2',data=embedding,hue="Cluster",
           fit_reg=False,legend=False,scatter_kws={'s':5})
for i in clusters:
    plt.annotate(i, 
                 embedding.loc[embedding['Cluster']==i,['UMAP1','UMAP2']].mean(),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=15, weight='bold')
 
f.savefig("Cluster",dpi=450)

embedding['topic3']=topic_matrix[:,3]
f=sns.lmplot(x='UMAP1', y='UMAP2',data=embedding,hue='topic3',palette="Purples",
             fit_reg=False,legend=False,scatter_kws={"s": 5})
f.savefig("topic3",dpi=450)
##-----------------------------------------------------------------------------
##topic specificity
##cdf
topic_matrix=pd.DataFrame(topic_matrix)
topic_matrix['Cluster']=y_pred
cluster_topic = topic_matrix.groupby(['Cluster']).mean()

f=sns.clustermap(cluster_topic,method='ward', metric='euclidean',cmap="Blues")
f.savefig("topic_cluster",dpi=450)
        
##Kolmogorov-Smirnov
topic_matrix = nmf.transform(tfidf)
spec_matrix=np.zeros((params['n_clusters'],n_components))
for i in clusters:
    m1 = topic_matrix[y_pred==i]
    m2 = topic_matrix[y_pred!=i]
    for j in range(n_components):
        c = stats.ks_2samp(m1[:,j],m2[:,j])[1]
        spec_matrix[i,j]=c
        
f=sns.clustermap(spec_matrix,method='ward', metric='euclidean',cmap="Blues")
f.savefig("topic_spec",dpi=450)

#topics=[]
#for i in range(params['n_clusters']):
#    topics.append(np.where(spec_matrix[i,:]==np.max(spec_matrix[i,:]))[0][0])
#topics = list(set(topics))

##-----------------------------------------------------------------------------
##gene (word) clustering
w = pd.DataFrame(w)
w = w.T
w.index = spleen.index

##top 300 genes
new_w = np.zeros((300*n_components,n_components))
new_w = pd.DataFrame(new_w)
genes = []
for i in range(n_components):
    a=w.sort_values(by=[w.columns[i]],ascending=False)
    genes.extend(a.index[:300])
    new_w.iloc[300*i:300*(i+1),:]=a.iloc[:300,:].values
new_w.index=genes
new_w = new_w.drop_duplicates()

cmap = sns.cubehelix_palette(light=1, as_cmap=True)
f=sns.clustermap(new_w,method='ward', metric='euclidean',cmap=cmap)
f.savefig("gene_topic",dpi=450)

##visualize genes after dimension reduction
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

embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(new_w)
embedding = pd.DataFrame(embedding)
embedding.columns=['UMAP1','UMAP2']
sns.lmplot(x='UMAP1', y='UMAP2',data=embedding,
           fit_reg=False,legend=False,scatter_kws={"s": 5,"color":"green"})
plt.savefig("UMAP_gene",dpi=300)
