from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, cohen_kappa_score, f1_score
import sys
import numpy as np

# Allow print matrix without truncation
np.set_printoptions(threshold=sys.maxsize)



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


