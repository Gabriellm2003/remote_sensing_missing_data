import argparse
import torch
import numpy as np
import os
from collections import OrderedDict
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, cohen_kappa_score, f1_score
import sys
import torch
from torchvision import transforms
from PIL import Image
import math



np.set_printoptions(threshold=sys.maxsize)

def infer (model, image_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    im = Image.open(image_path).convert('RGB')
    im_t = transform(im)
    batch_t = torch.unsqueeze(im_t,0)
    batch_t = batch_t.to(device)
    with torch.set_grad_enabled(False):
        output = model(batch_t)
        #print (output.data.cpu().numpy())
        return softmax(output.data.cpu().numpy())[0]

def calculate_metrics(preds, labels, file = None):
    cm = confusion_matrix(np.asarray(labels), np.asarray(preds))
    b_acc = balanced_accuracy_score(np.asarray(labels), np.asarray(preds))
    acc = accuracy_score(np.asarray(labels), np.asarray(preds))
    kappa = cohen_kappa_score(np.asarray(labels), np.asarray(preds))
    f1 = f1_score(np.asarray(labels), np.asarray(preds), average = 'weighted')    
    if file is not None:
        file.write("Accuracy: " + str(acc) + "\n")
        file.write("Balanced_Accuracy: " + str(b_acc) + "\n")
        file.write("Kappa: " + str(kappa) + "\n")
        file.write("F1: " + str(f1) + "\n")
        file.write("Confusion Matrix: \n" + str(cm) + "\n\n\n\n")
        file.write("Labels\n{}\n".format(np.asarray(labels)))
        file.write("Predictions\n{}".format(np.asarray(preds)))
        
    else:
        print ("\nAccuracy: " + str(acc))
        print ("Balanced_Accuracy: " + str(b_acc))
        print ("Kappa: " + str(kappa))
        print ("F1: " + str(f1))
        print (cm)

def get_images_list_top1(file, aerial_data_path, ground_data_path, query):
    counter = 0
    if (query == 'ground'):
        aerial = {}
        street = []
        for l in file:
            l = l.replace("'", "")
            aerial[str(counter)] = os.path.join(aerial_data_path, l.split(' : ')[1].split('___')[0], l.split(' : ')[1].split('___')[1].replace('\n', '') + '.png')
            # aerial.append(os.path.join(aerial_data_path, l.split(' : ')[1].split('___')[0], l.split(' : ')[1].split('___')[1].replace('\n', '') + '.png'))
            street.append(os.path.join(ground_data_path, l.split(' : ')[0].split('___')[0], l.split(' : ')[0].split('___')[1].replace('\n', '') + '.png'))
            counter += 1
    else:
        aerial = []
        street = {}
        for l in file:
            l = l.replace("'", "")
            #street.append(os.path.join(ground_data_path, l.split(' : ')[1].split('___')[0], l.split(' : ')[1].split('___')[1].replace('\n', '') + '.png'))
            street[str(counter)] = os.path.join(ground_data_path, l.split(' : ')[1].split('___')[0], l.split(' : ')[1].split('___')[1].replace('\n', '') + '.png') 
            aerial.append(os.path.join(aerial_data_path, l.split(' : ')[0].split('___')[0], l.split(' : ')[0].split('___')[1].replace('\n', '') + '.png'))
            counter += 1
    return aerial, street

def get_images_list_topK(file, aerial_data_path, ground_data_path, query, topK):
    counter = 0
    if (query == 'ground'):
        aerial = {}
        street = []
        for l in file:
            l_aux = l.split(' : ')[1]
            if ('6aW___77ZPTn9VCA' in l_aux):
                continue
            aerial[str(counter)] = []
            aerial[str(counter)].append(os.path.join(aerial_data_path, l_aux.split('___')[0], l_aux.split('___')[1].split(',')[0].replace("'", "") + '.png'))
            for i in range (topK-1):
                aerial[str(counter)].append(os.path.join(aerial_data_path, l_aux.split('___')[i+1].split(',')[1], l_aux.split('___')[i+2].split(',')[0].replace("'", "").replace('\n', '') + '.png'))
            counter += 1
            street.append(os.path.join(ground_data_path, l.split(' : ')[0].split('___')[0], l.split(' : ')[0].split('___')[1].replace('\n', '').replace("'", "") + '.png'))
    else:
        street = {}
        aerial = []
        for l in file:
            l_aux = l.split(' : ')[1]
            if ('6aW___77ZPTn9VCA' in l_aux):
                continue
            street[str(counter)] = []
            street[str(counter)].append(os.path.join(ground_data_path, l_aux.split('___')[0], l_aux.split('___')[1].split(',')[0].replace("'", "") + '.png'))
            for i in range (topK-1):
                street[str(counter)].append(os.path.join(ground_data_path, l_aux.split('___')[i+1].split(',')[1], l_aux.split('___')[i+2].split(',')[0].replace("'", "").replace('\n', '') + '.png'))
            counter += 1
            aerial.append(os.path.join(aerial_data_path, l.split(' : ')[0].split('___')[0], l.split(' : ')[0].split('___')[1].replace('\n', '').replace("'", "") + '.png'))
    return aerial, street


def main():
    parser = argparse.ArgumentParser(description='Image classification with retrieval file.')
    parser.add_argument('--aerial_model', type=str, required=True,
                        help='Path to aerial network model file.')
    parser.add_argument('--ground_model', type=str, required=True,
                        help='Path to ground network model file.')
    parser.add_argument('--net_type', type=str, required=True,
                        help='Choose[densenet,sknet,vgg]')
    parser.add_argument('--aerial_data_path', type=str, required=True,
                        help='Path to aerial data.')
    parser.add_argument('--ground_data_path', type=str, required=True,
                        help='Path to ground data.')
    parser.add_argument('--ranking_file_path', type=str, required=True,
                        help='path for file containing ranking.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='path for saving logs')
    parser.add_argument('--query_type', type=str, required=True,
                        help='Choose between [aerial/ground]')
    parser.add_argument('--ranking', type=int, required=True,
                        help='Ranking size')
    


    args = parser.parse_args()
    aerial_model = args.aerial_model
    ground_model = args.ground_model
    net_type = args.net_type
    aerial_data_path = args.aerial_data_path
    ground_data_path = args.ground_data_path
    ranking_file_path = args.ranking_file_path
    output_path = args.output_path
    query_type = args.query_type
    ranking = args.ranking

    if (not os.path.exists(output_path)):
        os.makedirs(output_path)

    if ('airound' in aerial_model):
        num_classes = 11
    else:
        num_classes = 7

    if (ranking == 1):
        file = open (os.path.join(ranking_file_path, query_type + '_list_for_classification.txt'), 'r')
        aerial, street = get_images_list_top1(file, aerial_data_path, ground_data_path, query_type)
         
    else:
        file = open (os.path.join(ranking_file_path, query_type + '_list_top' + str(ranking) + '.txt'))
        aerial, street = get_images_list_topK(file, aerial_data_path, ground_data_path, query_type, ranking)
        

    labels_dict = {'airport': 0, 'bridge': 1, 'church': 2, 'forest': 3, 'lake': 4, 
                   'park': 5, 'river': 6, 'skyscraper': 7, 'stadium': 8, 'statue': 9, 'tower': 10,
                   'apartment':0, 'apartament': 0, 'hospital': 1, 'house': 1, 'industrial': 2, 
                   'parking_lot': 3, 'religious': 4, 'school': 5, 'store': 6, 'vacant_lot': 8}
    
    a_model = torch.load(aerial_model)
    a_model.eval()
    g_model = torch.load(ground_model)
    g_model.eval()
    labels = []
    predictions = []
    stats_file2 = open(os.path.join(output_path, query_type + '_' + net_type + '_preds_top' + str(ranking) + '.csv'), 'w') 
    stats_file2.write('name_q;pred_g;pred_a;pred_f\n')
    
    if (query_type == 'ground'):
        
        for i in range (len(street)):
            print ("IMAGE " + str(i) + '/' +str(len(street)))
            if ('6aW.png' in aerial[str(i)]):
                continue
            label = labels_dict[street[i].split('/')[-2]]
            soft_a = np.zeros(num_classes)
            if (ranking != 1):
                for j in range(ranking):
                    soft_a += infer(a_model, aerial[str(i)][j])
            else:
                soft_a = infer(a_model, aerial[str(i)])
            soft_a = soft_a/ranking
            soft_b = infer(g_model, street[i])
            pred_a = np.argmax(soft_a)
            pred_b = np.argmax(soft_b)
            softmax = [np.prod(x) for x in zip(soft_a, soft_b)]
            labels.append(label)
            predictions.append(np.argmax(softmax))
            stats_file2.write(street[i] + ';' + str(pred_b) + ';' + str(pred_a) + ';' + str(np.argmax(softmax)) + '\n')
        stats_file = open(os.path.join(output_path, query_type + '_' + net_type + '_top' + str(ranking) + '.txt'), 'w')
        calculate_metrics(predictions, labels, stats_file)
    
    if (query_type == 'aerial'):
        
        for i in range (len(aerial)):
            print ("IMAGE " + str(i) + '/' +str(len(street)))
            if ('6aW.png' in street[str(i)]):
                continue
            label = labels_dict[aerial[i].split('/')[-2]]
            soft_a = np.zeros(num_classes)
            if (ranking != 1):
                for j in range(ranking):
                    soft_a += infer(g_model, street[str(i)][j])
            else:
                soft_a = infer(g_model, street[str(i)])
            soft_a = soft_a/ranking
            soft_b = infer(a_model, aerial[i])
            pred_a = np.argmax(soft_a)
            pred_b = np.argmax(soft_b)
            softmax = [np.prod(x) for x in zip(soft_a, soft_b)]
            labels.append(label)
            predictions.append(np.argmax(softmax))
            stats_file2.write(aerial[i] + ';' + str(pred_a) + ';' + str(pred_b) + ';' + str(np.argmax(softmax)) + '\n')
        stats_file = open(os.path.join(output_path, query_type + '_' + net_type + '_top' + str(ranking) + '.txt'), 'w')
        calculate_metrics(predictions, labels, stats_file)

if __name__ == '__main__':
    main()
