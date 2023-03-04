
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


with open(os.path.join(curr_path, "models", "code0_ebd_best.pkl"), "rb") as f:
    code0_ebd, code0_int = pickle.load(f)
# umap
embedding = umap.UMAP(n_neighbors=50,
                      n_components=3, min_dist=0.5, 
                      random_state=42, 
                      verbose=True).fit_transform(code0_ebd)

plot_dim = 2

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
    
    plt.title("simple 3D scatter plot")
    ax.set_xlabel('X-axis', fontweight ='bold')
    ax.set_ylabel('Y-axis', fontweight ='bold')
    ax.set_zlabel('Z-axis', fontweight ='bold')
    fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
    
    # show plot
    plt.show()


if plot_dim == 2:
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