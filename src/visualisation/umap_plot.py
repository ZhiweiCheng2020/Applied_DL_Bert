import os
import sys
import random
import umap
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import data.data_preprocess as preprocess
import models.bert as bert 
from models.utils import compute_loss 
import pickle

class plot_umap():
    def __init__(self, bert_ebd_path):
        self.curr_path = os.path.dirname(os.getcwd())
        self.bert_ebd_path = bert_ebd_path
        
    def load_ebd(self):
        # load the embeddings
        with open(self.bert_ebd_path, "rb") as f:
            self.NSP_ebd, self.code0_int, self.code1_int = pickle.load(f)
            
    def create_umap(self, chosen_code, plot_dim):
        # create umap for specified code type and dimension
        if chosen_code == "code0":
            code_indicator = self.code0_int
        elif chosen_code == "code1":
            code_indicator = [100*(self.code0_int[i]+1)+self.code1_int[i] for i in range(len(self.code0_int))]
        else:
            raise Exception("please choose the correct code type: code0 or code1")
        # create umap
        embedding = umap.UMAP(n_neighbors=20,
                            n_components=plot_dim, min_dist=0.4, 
                            random_state=42, 
                            verbose=True).fit_transform(self.NSP_ebd)
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
            scatter = ax.scatter3D(embedding[:, 0], embedding[:, 1],embedding[:, 2], 
                                alpha = 0.6,
                                c=code_indicator,)
            plt.title(chosen_code+" - embedding projection by UMAP", fontsize=20)
            ax.set_xlabel('Dimension 1', fontweight ='bold')
            ax.set_ylabel('Dimension 2', fontweight ='bold')
            ax.set_zlabel('Dimension 3', fontweight ='bold')
            # show plot
            plt.savefig(os.path.join(self.curr_path, "results", 
                chosen_code+"_umap_3_dim.pdf"), dpi=150)
            # plt.show()

        elif plot_dim == 2:
            # Creating 2d plot
            fig, ax = plt.subplots(figsize=(12, 10))
            scatter = plt.scatter(
                embedding[:, 0], embedding[:, 1],
                c=code_indicator,
                alpha = 0.6
            )
            plt.xlabel("Dimension 1", size=18)
            plt.ylabel("Dimension 2", size=18)
            plt.title(chosen_code+" - embedding projection by UMAP", fontsize=20)
            plt.savefig(os.path.join(self.curr_path, "results", 
                chosen_code+"_umap_2_dim.pdf"), dpi=150)
            # plt.show()
            
        else:
            raise Exception("plot_dim can only be 2 or 3")
        
    def get_all_umap(self):
        self.load_ebd()
        self.create_umap(chosen_code="code0", plot_dim=2)
        self.create_umap(chosen_code="code0", plot_dim=3)
        self.create_umap(chosen_code="code1", plot_dim=2)
        self.create_umap(chosen_code="code1", plot_dim=3)
        print("----UMAP Plot - complete!-------")
