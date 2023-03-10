import click
from train import train
from visualisation.prepare_umap_data import umap_creator
from visualisation.umap_plot import plot_umap

@click.command()
@click.option('--len_all', default=13478, help='the number of sequences you would like to include in model training')
@click.option('--lr', default=0.001, help='the initial learning rate')
@click.option('--num_epochs', default=100, help='number of epochs')
@click.option('--batch_size', default=64, help='batch size')
@click.option('--num_layer', default=4, help='number of Encoder Layerse')
@click.option('--num_head', default=4, help='number of heads in Multi-Head Attention')
# @click.option('--verbose', default=True, help='if verbose')
def run_model(len_all, lr, num_epochs, num_layer, batch_size, num_head):
    # train and save model
    saved_model_path = train(    
        len_all = len_all, # the whole dataset
        lr=lr, #learning rate
        num_epochs = num_epochs,
        num_layer = num_layer, # number of Encoder Layers
        num_head = num_head, # number of heads in Multi-Head Attention
        batch_size = batch_size
                                ).run_model()
    # create embedding
    bert_ebd_path = umap_creator(
        model_path=saved_model_path,
        batch_size=batch_size
                                 ).prepare_umap()
    # UMAP projection
    plot_umap(bert_ebd_path).get_all_umap()

if __name__ == "__main__":
    run_model()