import argparse
import os, random
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import network_factory as factory
import dataloader
import torch.nn as nn
import torch
import numpy as np

# Fix seed's for reproducibility
random.seed(42)
torch.manual_seed(42)

def main():
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to folder where models and stats will be saved')
    parser.add_argument('--batch', type=int, required=True,
                        help='Batch Size')
    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of epochs')
    parser.add_argument('--network_type', type=str, required=True,
                        help = 'Choose network type')
    parser.add_argument('--early_stop', type=int, required=True,
                        help='Number of epochs to activate early stop.')
    parser.add_argument('--fine_tunning_imagenet', type= bool, required=False, default=False,
                        help='set fine tunning on imagenet.')
    parser.add_argument('--feature_extract', type= bool, required=False, default=False,
                        help='Train just the classifier.')
    parser.add_argument('--bands', type= int, required=False, default=3,
                        help='Number of bands that images have.')
    parser.add_argument('--image_type', type= str, required=True,
                        help='Choose [aerial|sentinel|ground].')



    args = parser.parse_args()
    dataset_dir = args.dataset_path
    out_dir = args.output_path
    batch_size = args.batch
    epochs = args.epochs
    net_type = args.network_type
    fine_tunning = args.fine_tunning_imagenet
    early_stop = args.early_stop
    feature_extract = args.feature_extract
    total_classes = len(os.listdir(os.path.join(dataset_dir, 'train')))
    print (os.listdir(os.path.join(dataset_dir, 'train')))
    bands = args.bands
    image_type = args.image_type

    if (net_type == 'inception'):
        is_inception = True
    else:
        is_inception = False

    if (not os.path.exists(out_dir)):
        os.makedirs(out_dir)

    print ('.......Creating model.......')
    print('total classes: ', total_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size = factory.model_factory(net_type, total_classes, feature_extract, fine_tunning, bands)
    print (model)
    model = model.to(device)
    print ('......Model created.......')

    print ('......Creating dataloader......')
    if (image_type == 'aerial' and fine_tunning == False):
        #[0.359, 0.382, 0.326], [0.251, 0.25, 0.24]
        dataloaders_dict = dataloader.create_dataloader(dataset_dir, image_size, batch_size, [0.452, 0.442, 0.429], [0.256, 0.254, 0.253])
    if (image_type == 'ground' and fine_tunning == False):
        #[0.462, 0.49, 0.482], [0.26, 0.26, 0.267]
        dataloaders_dict = dataloader.create_dataloader(dataset_dir, image_size, batch_size, [0.492, 0.497, 0.482], [0.257, 0.257, 0.26])
    if (image_type == 'sentinel'):
        dataloaders_dict = dataloader.create_dataloader_sentinel(dataset_dir, image_size, batch_size)
    if (fine_tunning == True):
        dataloaders_dict = dataloader.create_dataloader(dataset_dir, image_size, batch_size, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    print ('......Dataloader created......')


    params_to_update = model.parameters()
    print("Params to learn:")

    #defining optimizer and loss
    # params = sum([np.prod(p.size()) for p in params_to_update])
    print (params_to_update)
    #print (params)
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    tensor_board = SummaryWriter(log_dir = out_dir)
    final_model, val_history = factory.train(model, dataloaders_dict, criterion, optimizer,
                                             epochs, early_stop, tensor_board, is_inception)
    print (out_dir)
    if fine_tunning:
        torch.save(final_model, os.path.join(out_dir, net_type + '_final_model_ft'))
        final_stats_file = open (os.path.join(out_dir, net_type + '_finalstats_ft.txt'), 'w')
    else:
        torch.save(final_model, os.path.join(out_dir, net_type + '_final_model'))
        final_stats_file = open (os.path.join(out_dir, net_type + '_finalstats.txt'), 'w')
    csv_file = open(os.path.join(out_dir, net_type + '_results.csv'), 'w')

    factory.final_eval(final_model, dataloaders_dict, csv_file, final_stats_file, is_inception)

if __name__ == '__main__':
    main()
