"""
Transformer for EEG classification
"""
import csv
import pandas as pd
from openpyxl import load_workbook
import os

import numpy as np
import torch

from torch import nn
from torch.backends import cudnn
from sklearn.metrics import confusion_matrix, classification_report

# Import model
from model.transformer_fl import ViT

from natsort import natsorted

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

cudnn.benchmark = False
cudnn.deterministic = True

# Band: choose in ["delta", "theta", "alpha", "beta"]
band = "EEG_Processed_new"

# Class mapping to indices
classes = ["B", "C"]
classIdx = {"B": 0, "C": 1}

n_epochs = 250
num_folds = 5

# Training metrics
fold_tr_labels = dict()
fold_tr_preds = dict()
fold_tr_scores = dict()

fold_tr_pz_label = dict()
fold_tr_pz_preds = dict()

# Validation metrics
fold_val_labels = dict()
fold_val_preds = dict()
fold_val_scores = dict()

fold_val_pz_label = dict()
fold_val_pz_preds = dict()

# Test metrics
fold_te_labels = dict()
fold_te_preds = dict()
fold_te_scores = dict()
fold_te_aucs = dict()

fold_te_pz_label = dict()
fold_te_pz_preds = dict()


# Loop over folds
for fold in range(num_folds):
    print("Current fold: " + str(fold))
    mainDir = "D:\\Antonio\\Rev_Results\\2class_5cv_64_30sec\\" + band + "\\" + str(fold+1) + "\\"
    res_path = mainDir + "res_" + band + "\\"

    # Model init
    #model = ViT(n_classes=2).cuda()
    #model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])
    #model = model.cuda()

    states = os.listdir(res_path)
    states = natsorted(states)
    epochs = len(states)

    # Load models and get the epoch with highest auc on validation set
    val_aucs = []
    row = []
    #with open('D:\\Sibilano\\OneDrive - Politecnico di Bari\\EEG_Project\\Nuovo_Lavoro\\Results\\Test1_'+ str(fold) +'.csv', 'w',
                #encoding='UTF8') as f:
        #writer = csv.DictWriter(f, fieldnames=['epoch','Train accuracy', 'Validation accuracy', 'Test accuracy', 'Validation AUC', 'Test AUC'], delimiter=',', lineterminator='\r')
       # writer.writeheader()

    df = pd.DataFrame()
    #path = "D:\\Sibilano\\OneDrive - Politecnico di Bari\\EEG_Project\\Nuovo_Lavoro\\risultati_che_avevamo\\Test_25.xlsx"
    #writer = pd.ExcelWriter(path, mode='a', if_sheet_exists='overlay')
    for epoch in range(epochs):
        state = res_path + states[epoch]
        perf = torch.load(state)
        res = {key: perf[key] for key in perf.keys()
            & {'epoch','Train accuracy', 'Validation accuracy', 'Test accuracy', 'Validation AUC', 'Test AUC'}}

        #df = pd.concat([df, pd.DataFrame(data=res, index=[0])], ignore_index=True)
        #df.to_excel(writer, sheet_name= str(fold), index=False)

        #writer.writerows([res])

        val_aucs.append(perf['Validation AUC'])
        #f.close()
    #writer.close()

    idx_max = np.argmax(val_aucs)
    print(idx_max)
    print(max(val_aucs))
    auc_max = val_aucs[idx_max]

    state = res_path + states[idx_max]
    perf = torch.load(state)

    val_pz_label = []
    val_pz_preds = []

    tr_pz_label = []
    tr_pz_preds = []

    te_pz_label = []
    te_pz_preds = []

    # Training performance
    tr_preds = perf['Train predictions']
    tr_labels = perf['Train labels']
    tr_scores = perf['Train scores']

    pz_list = torch.unique(perf['Train patients'])
    for patient in pz_list:
        label = torch.unique(perf['Train labels'][perf['Train patients'] == patient]).item()
        pred = torch.mode(perf['Train predictions'][perf['Train patients'] == patient], 0).values.item()
        tr_pz_label.append(label)
        tr_pz_preds.append(pred)

    # Validation performance
    val_preds = perf['Validation predictions']
    val_labels = perf['Validation labels']
    val_scores = perf['Validation scores']

    pz_list = torch.unique(perf['Validation patients'])
    for patient in pz_list:
        label = torch.unique(perf['Validation labels'][perf['Validation patients'] == patient]).item()
        pred = torch.mode(perf['Validation predictions'][perf['Validation patients'] == patient], 0).values.item()
        val_pz_label.append(label)
        val_pz_preds.append(pred)

    # Test performance
    te_preds = perf['Test predictions']
    te_labels = perf['Test labels']
    te_scores = perf['Test scores']
    print('Test accuracy', perf['Test accuracy'])
    print('Test AUC', perf['Test AUC'])
    fold_te_aucs[fold] = perf['Test AUC']

    pz_list = torch.unique(perf['Test patients'])
    for patient in pz_list:
        label = torch.unique(perf['Test labels'][perf['Test patients'] == patient]).item()
        pred = torch.mode(perf['Test predictions'][perf['Test patients'] == patient], 0).values.item()
        te_pz_label.append(label)
        te_pz_preds.append(pred)

    fold_tr_labels[fold] = tr_labels
    fold_tr_preds[fold] = tr_preds
    fold_tr_scores[fold] = tr_scores

    fold_tr_pz_label[fold] = tr_pz_label
    fold_tr_pz_preds[fold] = tr_pz_preds

    fold_val_labels[fold] = val_labels
    fold_val_preds[fold] = val_preds
    fold_val_scores[fold] = val_scores
    fold_val_pz_label[fold] = val_pz_label
    fold_val_pz_preds[fold] = val_pz_preds

    fold_te_labels[fold] = te_labels
    fold_te_preds[fold] = te_preds
    fold_te_scores[fold] = te_scores

    fold_te_pz_label[fold] = te_pz_label
    fold_te_pz_preds[fold] = te_pz_preds

ep_accs = []
ep_senss = []
ep_specs = []
ep_aucs = []
ep_f1 = []
ep_prec = []
pz_accs = []
pz_senss = []
pz_specs = []
pz_f1 = []
pz_prec = []

for fold in range(num_folds):
    # print("Fold: " + str(fold))
    # print("Training Epochs")
    # print(confusion_matrix(np.asarray(fold_tr_labels[fold].cpu()), np.asarray(fold_tr_preds[fold].cpu())))
    # print("-----------------")
    # print("Validation Epochs")
    # print(confusion_matrix(np.asarray(fold_val_labels[fold].cpu()), np.asarray(fold_val_preds[fold].cpu())))
    # print("Test Epochs")
    print(confusion_matrix(np.asarray(fold_te_labels[fold].cpu()), np.asarray(fold_te_preds[fold].cpu())))
    #print(classification_report(np.asarray(fold_te_labels[fold].cpu()), np.asarray(fold_te_preds[fold].cpu())))
    tn, fp, fn, tp = confusion_matrix(np.asarray(fold_te_labels[fold].cpu()), np.asarray(fold_te_preds[fold].cpu())).ravel()
    ep_accs.append((tp + tn)/(tp +tn + fp + fn))
    ep_senss.append(tp/(tp + fn))
    ep_specs.append(tn/(tn + fp))
    ep_f1.append(tp/(tp+ 0.5*(fp+fn)))
    ep_prec.append(tp/(tp+fp))
    ep_aucs.append(fold_te_aucs[fold])


    # print("-----------------")
    #
    # print("Training Patients")
    #print(confusion_matrix(np.asarray(fold_tr_pz_label[fold]), np.asarray(fold_tr_pz_preds[fold])))
    # print("-----------------")
    # print("Validation Patient")
    #print(confusion_matrix(np.asarray(fold_val_pz_label[fold]), np.asarray(fold_val_pz_preds[fold])))
    # print("Test Patients")
    cm = confusion_matrix(np.asarray(fold_te_pz_label[fold]), np.asarray(fold_te_pz_preds[fold]))
    #print(classification_report(np.asarray(fold_te_labels[fold].cpu()), np.asarray(fold_te_preds[fold].cpu())))
    print(cm)
    # print("Test Patients Accuracy")
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    pz_accs.append(acc)
    # print(acc)
    # print("-----------------")

    tn, fp, fn, tp = confusion_matrix(np.asarray(fold_te_pz_label[fold]), np.asarray(fold_te_pz_preds[fold])).ravel()
    pz_accs.append((tp + tn) / (tp + tn + fp + fn))
    pz_senss.append(tp / (tp + fn))
    pz_specs.append(tn / (tn + fp))
    pz_f1.append(tp / (tp + 0.5 * (fp + fn)))
    pz_prec.append(tp/(tp+fp))

print(np.mean(ep_accs), np.std(ep_accs))
print(np.mean(ep_senss), np.std(ep_senss))
print(np.mean(ep_specs), np.std(ep_specs))
print("f1_ep",np.mean(ep_f1), np.std(ep_f1))
print(np.mean(ep_prec), np.std(ep_prec))
print(np.mean(ep_aucs), np.std(ep_aucs))
print(np.mean(pz_accs), np.std(pz_accs))
print(np.mean(pz_senss), np.std(pz_senss))
print(np.mean(pz_specs), np.std(pz_specs))
print("f1_pat",np.mean(pz_f1), np.std(pz_f1))
print(np.mean(pz_prec), np.std(pz_prec))