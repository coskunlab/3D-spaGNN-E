
import sys

import scanpy as sc
import numpy as np

import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib import rc_context
from numpy.random import shuffle
from joblib import Parallel, delayed

from tqdm import tqdm

def gene_count_df(df, gene_name):
    counts = dict([(gene,[df.groupby(gene_name).get_group(gene).shape[0]]) for gene in df[gene_name].unique().tolist()])
    return pd.DataFrame(counts)

def counting(df, count_by, gene_name):
    counts_l = []
    groups = df.groupby(count_by)
    for item in np.unique(df[count_by].tolist()):
        group_counts = gene_count_df(groups.get_group(item), gene_name)
        counts_l.append(group_counts)
    exp = pd.concat(counts_l, axis=0, join="outer", ignore_index=True)
    exp = exp.fillna(0)
    return exp

def find_neighborhoods(cluster_rna_df, n_neighbors, row_name, col_name):
    
    r = cluster_rna_df[row_name].tolist()
    c = cluster_rna_df[col_name].tolist()
    r_c = list(zip(r,c))
    
    if n_neighbors>len(r_c):
        return None
    
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(r_c)
    dist, neighbors = neigh.kneighbors(r_c)
    return dist,neighbors

def count_neighbors(cluster_rna_df, neighbors, genes, markers_l):
    counts = np.zeros((cluster_rna_df.shape[0], len(markers_l)))
    for i in range(neighbors.shape[0]):
        for j in range(neighbors.shape[1]):
            neighbor = neighbors[i,j]
            idx = markers_l.index(genes[neighbor])
            counts[i,idx] = counts[i,idx] + 1
    df = pd.DataFrame(counts, columns=markers_l)
    return df

'''
def proximity_network(cluster_rna_df, n_neighbors, row_name, col_name, marker_names):
    r = cluster_rna_df[row_name].tolist()
    c = cluster_rna_df[col_name].tolist()
    r_c = list(zip(r,c))
    
    if n_neighbors > len(r_c):
        return None
    
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(r_c)
    dist, neighbors = neigh.kneighbors(r_c)
    
    genes = cluster_rna_df[marker_names].tolist()
    markers_l = list(np.unique(genes))
    df = count_neighbors(cluster_rna_df, neighbors, genes, markers_l)
    return df.corr(method='pearson')
'''

def count_interactions(neighbor_counts_df, marker_names):
    summed = neighbor_counts_df.groupby(marker_names).sum()
    return summed.values + summed.values.T

def permutation_z_score(interaction, interaction_l):
    arr = np.array(interaction_l)
    m = np.mean(arr, axis=0)
    s = np.std(arr, axis=0)
    with np.errstate(divide='ignore',invalid='ignore'):
        z = (interaction - m) / s
    return (z, m, s)

def permutation(cluster_rna_df, n_neighbors, row_name, col_name, marker_names, n_permutation):
    temp = find_neighborhoods(cluster_rna_df, n_neighbors, row_name, col_name)
    
    if temp is None:
        return None
    else:
        dist,neighbors = temp
    
    genes = cluster_rna_df[marker_names].tolist()
    markers_l = list(np.unique(genes))
    neighbor_counts_df = count_neighbors(cluster_rna_df, neighbors, genes, markers_l)
    neighbor_counts_df.insert(0,marker_names,cluster_rna_df[marker_names].tolist())
    
    interaction = count_interactions(neighbor_counts_df, marker_names)
    
    interaction_l = []

    for i in range(n_permutation):
        shuffle(genes)
        neighbor_counts_df = count_neighbors(cluster_rna_df, neighbors, genes, markers_l)
        neighbor_counts_df.insert(0,marker_names,genes)
        interaction_l.append(count_interactions(neighbor_counts_df,marker_names))
    
    (z_s, m, s) = permutation_z_score(interaction, interaction_l)
    
    z_df = pd.DataFrame(z_s, index=markers_l, columns=markers_l)
    z_df = z_df.fillna(0)
    return (z_df, m, s)




'''
def standard_df(data, x = 'x', y = 'y', cell = 'cell', gene = 'gene'):
    return data[[x,y,cell,gene]]
    # data: panda dataFrame
    # x,y,cell,gene: name of the parameters
'''

def clustered_cell_df(df, cell_n, r = 0.1, x = 'x', y = 'y', cell = 'cell' ):
    cell_df = df.groupby(cell).get_group(cell_n)
    l = np.vstack((cell_df[x], cell_df[y])).T
    adata = sc.AnnData(l)
    adata.var_name = ['x','y']
    adata.obsm['spatial'] = np.vstack((cell_df[y], cell_df[x])).T
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution = r)
    cell_df.insert(loc=cell_df.shape[1],column='patch',value=adata.obs['leiden'].tolist())
    return cell_df

def clustered_cell_df_all(df, r = 0.1, x = 'x', y = 'y', cell = 'cell'):
    cell_l = df[cell].unique().tolist()
    final_cell_df = pd.DataFrame()
    for c in tqdm(cell_l):
        cell_df = df.groupby(cell).get_group(c)
        l = np.vstack((cell_df[x], cell_df[y])).T
        adata = sc.AnnData(l)
        adata.var_name = ['x','y']
        adata.obsm['spatial'] = np.vstack((cell_df[y], cell_df[x])).T
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, resolution = r)
        cell_df.insert(loc=cell_df.shape[1],column='patch',value=adata.obs['leiden'].tolist())
        final_cell_df= pd.concat([final_cell_df, cell_df])
    return final_cell_df


def patch_correlations(df, cell_n, r = 0.1 , x = 'x', y = 'y', cell = 'cell', gene = 'gene'):
    df = clustered_cell_df_all(df,r = r, x = x, y = y, cell = cell)
    cell_df = df.groupby(cell).get_group(cell_n)
    patch_count = counting(cell_df,count_by='patch',gene_name= gene)
    corr = patch_count.corr(method='pearson')
    return corr

def patch_proximity(df, cell_n, patch_n, n_neighbors = 10, n_permutation= 1000,
                    r = 0.1 , x = 'x', y = 'y', cell = 'cell', gene = 'gene'):
    df = clustered_cell_df_all(df,r = r, x = x, y = y, cell = cell)
    cell_df = df.groupby(cell).get_group(cell_n)
    patch_df = cell_df.groupby('patch').get_group(patch_n)
    neighborhood_network = permutation(patch_df,n_neighbors=n_neighbors,n_permutation=n_permutation,
                                         row_name=y,col_name=x,
                                        marker_names=gene)
    return neighborhood_network

# tqdm
def all_patch_proximity(df, cell_n, n_neighbors = 10, n_permutation = 1000, r = 0.1,
                        x = 'x', y = 'y', cell = 'cell', gene = 'gene'):
    df = clustered_cell_df_all(df,r = r, x = x, y = y, cell = cell)
    cell_df = df.groupby(cell).get_group(cell_n)
    patch_l = np.sort(cell_df['patch'].unique())
    proximity_l = []
    for p in tqdm(patch_l):
        patch_df = cell_df.groupby('patch').get_group(p)
        neighborhood_network = permutation(patch_df, n_neighbors = n_neighbors,n_permutation=n_permutation,
                                         row_name = y,col_name = x, marker_names = gene)
        proximity_l.append(neighborhood_network)
    return proximity_l

def find_connected_pairs(neighbors,patch_df):
    connecting_pairs = []
    idx = patch_df.index.tolist()
    for i in range(neighbors.shape[0]):
        for j in range(neighbors.shape[1]):
            pair1 = (idx[i],idx[neighbors[i,j]])
            pair2 = (idx[neighbors[i,j]],idx[i])
            if (not pair1 in connecting_pairs) or (not pair2 in connecting_pairs):
                connecting_pairs.append(pair1)
    return connecting_pairs