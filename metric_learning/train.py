import os
import argparse
import dataloader
from dataloader import DataLoader
import network_factory
import criterion as c
import torch
import numpy as np
import statistics as stat



def train(model, dataloader, optimizer, batch, epochs, output_path):
    sat_global_matrix = np.zeros([dataloader.get_val_dataset_size(), 2048])
    grd_global_matrix = np.zeros([dataloader.get_val_dataset_size(), 2048])
    ids_global_matrix = np.empty([dataloader.get_val_dataset_size()], dtype="S100")
    ids_global_matrix2 = np.empty([dataloader.get_val_dataset_size()], dtype="S100") #ground


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)


    for i in range(epochs):
        print ("Epoch " + str(i))
        counter = 0
        loss_value = 0
        #training
        model.train()
        while True:
            batch_aer, batch_grd = dataloader.next_train_batch_scan(batch)
            if batch_aer is None:
                print ('Finished epoch or error reading image.')
                break
            counter += 1
            optimizer.zero_grad()
            batch_aer = batch_aer.to(device)
            batch_grd = batch_grd.to(device)
            dist_matrix, y_a, y_g = model(batch_aer, batch_grd)
            loss = c.weighted_soft_margin_triple_loss(dist_matrix, batch)
            #print ("ITERATION " + str(counter) + ": " + str(loss.data.cpu().numpy()))
            loss_value += loss.data.cpu().numpy()
            loss.backward()
            optimizer.step()
        print ("LOSS = " + str(loss_value/counter))
        torch.save(model, os.path.join(output_path, "model_" + str(i)))
        

        # #validation
        model.eval()
        loss_value = 0
        val_i = 0
        while True:
            batch_aer, batch_grd, batch_ids, batch_ids2 = dataloader.next_val_batch_scan(batch)
            if (batch_aer is None):
                break
            batch_aer = batch_aer.to(device)
            batch_grd = batch_grd.to(device)
            dist_matrix, y_a, y_g = model(batch_aer, batch_grd)
            loss = c.weighted_soft_margin_triple_loss(dist_matrix, batch)
            
            #saving results

            sat_global_matrix[val_i : val_i + y_a.shape[0], :] = y_a.data.cpu().numpy()
            grd_global_matrix[val_i : val_i + y_g.shape[0], :] = y_g.data.cpu().numpy()
            ids_global_matrix[val_i:val_i + y_a.shape[0]] = batch_ids
            ids_global_matrix2[val_i:val_i + y_g.shape[0]] = batch_ids2

            val_i += y_a.shape[0]


        distances = 2 - 2 * np.matmul(grd_global_matrix, np.transpose(sat_global_matrix))
        stat.calculate_mAP(distances, ids_global_matrix, ids_global_matrix2)


def main():
    parser = argparse.ArgumentParser(description='Metric Learning for multi-view data completion.')
    parser.add_argument('--aerial_path', type=str, required=True,
                        help='Path to aerial dataset.')
    parser.add_argument('--ground_path', type=str, required=True,
                        help='Path to ground dataset.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output folder.')
    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of epochs.')
    

    
    args = parser.parse_args()
    aerial_path = args.aerial_path
    ground_path = args.ground_path
    output_path = args.output
    epochs = args.epochs
    batch = len(os.listdir(os.path.join(aerial_path, 'train')))
    learning_rate = 1e-5

    if (not os.path.exists(output_path)):
    	os.makedirs(output_path)

    input_data = DataLoader(aerial_path, ground_path, batch)
    model = network_factory.SwAVResnet(batch)
    parameters_to_learn = model.parameters()
    optimizer = torch.optim.Adam(params = parameters_to_learn, lr = learning_rate, betas = (0.9, 0.999))

    train(model, input_data, optimizer, batch, epochs, output_path)

if __name__ == '__main__':
    main()
