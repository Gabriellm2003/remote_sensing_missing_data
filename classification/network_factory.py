import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models
import statistics
import time
import copy
from tqdm import tqdm
from SENets.se_resnet import se_resnet50
import sknet
from collections import OrderedDict

def model_factory(model_name, num_classes, feature_extract=False, use_pretrained=True, bands=3):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        if (not use_pretrained):
            model_ft.conv1 = nn.Conv2d(bands, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        if (not use_pretrained):
            model_ft.features[0] = nn.Conv2d(bands, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "densenet":
        """ Densenet169
        """
        model_ft = models.densenet169(pretrained=use_pretrained)
        if (not use_pretrained):
            model_ft.features.conv0 = nn.Conv2d(bands, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        if (not use_pretrained):
            model_ft.features[0] = nn.Conv2d(bands, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == 'squeezenet':
        """ squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        if (not use_pretrained):
            model_ft.features[0] = nn.Conv2d(bands, 96, kernel_size=(7, 7), stride=(2, 2))
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == 'inception':
        """ inception v3
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        if (not use_pretrained):
            model_ft.Conv2d_1a_3x3.conv = nn.Conv2d(bands, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
    elif model_name == 'seresnet':
        """ squeeze and exciting resnet50
        """
        model_ft = se_resnet50(pretrained =use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        if (not use_pretrained):
            model_ft.conv1 = nn.Conv2d(bands, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_fltrs = 2048
        model_ft.fc = nn.Linear(in_features=num_fltrs, out_features=num_classes, bias = True)
        input_size = 224
    elif model_name == 'sknet':
        """selective kernels network
        """
        model_ft = sknet.sk_resnet101()
        if (use_pretrained):
            state_dict = torch.load('/mnt/DADOS_PONTOISE_1/gabriel/Mestrado/my_repo/sk_resnet101.pth.tar')['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model_ft.load_state_dict(new_state_dict)
        else:
            model_ft.conv1 = nn.Conv2d(bands, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
        model_ft.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True) 
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def train(model, dataloaders, criterion, optimizer, num_epochs, epochs_early_stop, tensor_board, is_inception):
    counter_early_stop_epochs = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    #counter = 0
    val_acc_history = []
    total_time = 0

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = 9999999.99

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        predictions = []
        labels_list = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs[0].to(device)
                labels = labels[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if (phase == 'train'):
                        time1 = time.time()

                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        # Get model outputs and calculate loss
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    if (phase == 'train'):
                        total_time += (time.time() - time1)

                # statistics
                for p in preds.data.cpu().numpy(): 
                    predictions.append(p)
                for l in labels.data.cpu().numpy(): 
                    labels_list.append(l)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if (phase == 'train'):
                tensor_board.add_scalar('Loss/train', epoch_loss, epoch)
                tensor_board.add_scalar('Accuracy/train', epoch_acc, epoch)
            else:
                tensor_board.add_scalar('Loss/val', epoch_loss, epoch)
                tensor_board.add_scalar('Accuracy/val', epoch_acc, epoch)

            # deep copy the model
            if phase == 'validation':
               counter_early_stop_epochs += 1
               val_acc_history.append(epoch_acc)
            if phase == 'validation' and epoch_loss < best_val_loss:
                counter_early_stop_epochs = 0
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            statistics.calculate_metrics(predictions, labels_list)
            
            predictions = []
            labels_list = []
        print ('Epoch ' + str(epoch) + ' - Time Spent ' + str(total_time))
        if (counter_early_stop_epochs >= epochs_early_stop):
            print ('Stopping training because validation loss did not improve in ' + str(epochs_early_stop) + ' consecutive epochs.')
            break
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_val_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def softmax(A):
    e = np.exp(A)
    return  e / e.sum(axis=0).reshape((-1,1))

def final_eval (model, dataloaders, csv_file, stats_file, is_inception):
    print ("Begining final eval.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    csv_file.write('Image;Labels;Predictions;Softmax\n')
    predictions = []
    labels_list = []
    softmax_values = []
    image_names = []
    #model.eval()
    for inputs, labels in dataloaders['validation']:
        #saving name of images
        for names in labels[0]: 
            image_names.append(names)
        inputs = inputs[0].to(device)
        labels = labels[1].to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for p in preds.data.cpu().numpy(): 
            predictions.append(p)
        for l in labels.data.cpu().numpy(): 
            labels_list.append(l)
        for s in range(len(preds)):
            for o in softmax(outputs.data.cpu().numpy()[s]):
                softmax_values.append(o)
    for i in range(len(predictions)):
        csv_file.write(str(image_names[i]) + ';' + str(labels_list[i]) + ';' + str(predictions[i]) + ';' + str(softmax_values[i]) + '\n')
    statistics.calculate_metrics(predictions, labels_list, stats_file)