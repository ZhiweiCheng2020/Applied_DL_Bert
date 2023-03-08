
import torch
import os
import sys
import random
import umap
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
curr_path = os.getcwd()
sys.path.append(curr_path)
sys.path.insert(0, os.path.join(curr_path, "src"))
import src.data.data_preprocess as preprocess
import src.models.bert as bert 
from src.models.utils import compute_loss 
import pickle

# dimension of the umap
plot_dim = 2

# load the embeddings of code0
with open(os.path.join(curr_path, "models", "bert_ebd.pkl"), "rb") as f:
    NSP_ebd, code0_int, code1_int = pickle.load(f)
    
# create umap
embedding = umap.UMAP(n_neighbors=100,
                      n_components=plot_dim, min_dist=0.2, 
                      random_state=42, 
                      verbose=True).fit_transform(NSP_ebd)
# plots
if plot_dim == 3:
    # Creating 3d plot
    # Creating figure
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = 0.3,
            alpha = 0.2)
    sctt = ax.scatter3D(embedding[:, 0], embedding[:, 1],embedding[:, 2], 
                        alpha = 0.8,
                        c=code0_int,)
    plt.title("Code 0 - Embedding projection by UMAP", fontsize=20)
    ax.set_xlabel('Dimension 1', fontweight ='bold')
    ax.set_ylabel('Dimension 2', fontweight ='bold')
    ax.set_zlabel('Dimension 3', fontweight ='bold')
    fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
    # show plot
    plt.show()
    plt.savefig(os.path.join(curr_path, "results", 
        "umap_3_dim.pdf"), dpi=150)


elif plot_dim == 2:
    # Creating 2d plot
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = plt.scatter(
        embedding[:, 0], embedding[:, 1],
        c=code0_int,
    )
    plt.xlabel("Dimension 1", size=18)
    plt.ylabel("Dimension 2", size=18)
    code0_names = ['A', 'B', 'C']
    plt.legend(handles=scatter.legend_elements()[0], 
            labels=code0_names, fontsize=18)
    print(scatter.legend_elements()[0])
    plt.title("Code 0 - Embedding projection by UMAP", fontsize=20)
    plt.show()
    plt.savefig(os.path.join(curr_path, "results", 
        "umap_2_dim.pdf"), dpi=150)
    
else:
    raise Exception("plot_dim can only be 2 or 3")
