import os
import argparse
import dataloader
from dataloader import DataLoader
import network_factory
import criterion as c
import torch
import numpy as np
import statistics as stat



def test(model, dataloader, batch, output_path, query_type):
    sat_global_matrix = np.zeros([dataloader.get_test_dataset_size(), 2048])
    grd_global_matrix = np.zeros([dataloader.get_test_dataset_size(), 2048])
    ids_global_matrix = np.empty([dataloader.get_test_dataset_size()], dtype="S100")
    ids_global_matrix2 = np.empty([dataloader.get_test_dataset_size()], dtype="S100") #ground


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_i = 0

    while True:
        batch_aer, batch_grd, batch_ids, batch_ids2 = dataloader.next_test_batch_scan(batch)
        if batch_aer is None:
            print ('Finished feature extraction.')
            break
        batch_aer = batch_aer.to(device)
        batch_grd = batch_grd.to(device)
        dist_matrix, y_a, y_g = model(batch_aer, batch_grd)
        loss = c.weighted_soft_margin_triple_loss(dist_matrix, batch)

        sat_global_matrix[test_i : test_i + y_a.shape[0], :] = y_a.data.cpu().numpy()
        grd_global_matrix[test_i : test_i + y_g.shape[0], :] = y_g.data.cpu().numpy()
        ids_global_matrix[test_i : test_i + y_a.shape[0]] = batch_ids
        ids_global_matrix2[test_i : test_i + y_g.shape[0]] = batch_ids2
        test_i += y_a.shape[0]

    distances = 2 - 2 * np.matmul(grd_global_matrix, np.transpose(sat_global_matrix))
    stat.calculate_mAP(distances, ids_global_matrix, ids_global_matrix2)

    #saving inference results
    np.save(os.path.join(output_path, query_type + '_grd_descriptor.npy'), grd_global_matrix)
    np.save(os.path.join(output_path, query_type + '_sat_descriptor.npy'), sat_global_matrix)
    np.save(os.path.join(output_path, query_type + '_ids_matrix.npy'), ids_global_matrix)
    np.save(os.path.join(output_path, query_type + '_ids_matrix2.npy'), ids_global_matrix2)

    f1 = open(os.path.join(output_path, query_type + '_list_for_classification.txt'), 'w')
    stat.generate_retrieval_list(distances, 1, ids_global_matrix, ids_global_matrix2, f1, query_type)
    f2 = open(os.path.join(output_path, query_type + '_list_top5.txt'), 'w')
    stat.generate_retrieval_list(distances, 5, ids_global_matrix, ids_global_matrix2, f2, query_type)
    f3 = open(os.path.join(output_path, query_type + '_list_top10.txt'), 'w')
    stat.generate_retrieval_list(distances, 10, ids_global_matrix, ids_global_matrix2, f3, query_type)
    f4 = open(os.path.join(output_path, query_type + '_list_top50.txt'), 'w')
    stat.generate_retrieval_list(distances, 50, ids_global_matrix, ids_global_matrix2, f4, query_type)
    f5 = open(os.path.join(output_path, query_type + '_list_top100.txt'), 'w')
    stat.generate_retrieval_list(distances, 100, ids_global_matrix, ids_global_matrix2, f5, query_type)





def main():
    parser = argparse.ArgumentParser(description='Metric Learning for multi-view data completion.')
    parser.add_argument('--aerial_path', type=str, required=True,
                        help='Path to aerial dataset.')
    parser.add_argument('--ground_path', type=str, required=True,
                        help='Path to ground dataset.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output folder.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model.')
    parser.add_argument('--use_as_query', type=str, required=True,
    					help='Select [ground, aerial]')


    args = parser.parse_args()
    aerial_path = args.aerial_path
    ground_path = args.ground_path
    output_path = args.output
    model_path = args.model_path
    query_type = args.use_as_query
    batch = 10


    input_data = DataLoader(aerial_path, ground_path, batch)
    model = torch.load(model_path)
    test(model, input_data, batch, output_path, query_type)


if __name__ == '__main__':
    main()
