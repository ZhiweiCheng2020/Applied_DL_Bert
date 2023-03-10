import click
import train as train
import visualisation.prepare_umap_data as prepare
import visualisation.umap_plot as plot

@click.command()
@click.option('--len_all', default=13478, help='the number of sequences you would like to include in model training')
@click.option('--lr', default=0.001, help='the initial learning rate')
@click.option('--num_epochs', default=100, help='number of epochs')
@click.option('--batch_size', default=64, help='batch size')
@click.option('--num_layer', default=4, help='number of Encoder Layerse')
@click.option('--num_head', default=4, help='number of heads in Multi-Head Attention')
# @click.option('--verbose', default=True, help='if verbose')
def run_model(len_all, lr, num_epochs, batch_size, num_layer, num_head):
    train.train_model(    
        len_all = len_all, # the whole dataset
        lr=lr, #learning rate
        num_epochs = num_epochs,
        batch_size = batch_size,
        num_layer = 4, # number of Encoder Layers
        num_head = 4, # number of heads in Multi-Head Attention
        )
    prepare.prepare_umap()
    plot.get_all_umap()

if __name__ == "__main__":
    run_model()