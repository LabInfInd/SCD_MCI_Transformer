"""
EEG Conformer

Convolutional Transformer for EEG decoding

Couple CNN and Transformer in a concise manner with amazing results
"""
import csv
# remember to change paths

import os

from sklearn import metrics
from sklearn.metrics import confusion_matrix

from util.dataset_loader_fl import EEGDatasetTrain2

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import time
import datetime

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch

from torch import nn
from torch.backends import cudnn
from sklearn.model_selection import StratifiedKFold, train_test_split

from model.transformer_fl import ViT
import matplotlib.pyplot as plt

cudnn.benchmark = False
cudnn.deterministic = True

def main():
    #model = ViT(num_heads=8, n_classes=2).cuda()
    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Paths
    main_path = "D:\\Sibilano\\OneDrive - Politecnico di Bari\\EEG_Project\\Occhi_Chiusi\\"
    label_path_file = "D:\\Sibilano\\OneDrive - Politecnico di Bari\\EEG_Project\\Labels_new.csv"

    # Band: choose in ["delta", "theta", "alpha", "beta"]
    band = "EEG_Processed_new"

    # Path with EEG data
    data_path = main_path + band + "\\"

    # Class mapping to indices
    class_idx = {"B": 0, "C": 1}

    # Subject labels
    with open(label_path_file, encoding='utf-8-sig') as file:
        dataLabels = csv.reader(file)
        dataLabels1 = list(dataLabels)

    # Data organization
    indices = [i for i, x in enumerate(dataLabels1, start=1)]
    elements = [(ele, dataLabels1[ele - 1][0]) for ele in indices if dataLabels1[ele - 1] not in [["NA"], ["A"]]]

    case_index = [element[0] for element in elements]
    case_labels = [element[1] for element in elements]

    cases = np.asarray(case_index)
    labels = np.asarray(case_labels)

    train_data, test_data, train_label, test_label = train_test_split(cases, labels, test_size=0.2,
                                                                      random_state=42,
                                                                      shuffle=True
                                                                      )

    train_cases = train_data
    train_labels = train_label

    test_cases = test_data
    test_labels = test_label

    batch_size = 16
    n_epochs = 250

    Tensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor

    criterion_l1 = torch.nn.L1Loss().cuda()
    criterion_l2 = torch.nn.MSELoss().cuda()
    criterion_cls = torch.nn.CrossEntropyLoss().cuda()

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    fold_index = 0
    for train_index, val_index in kf.split(train_cases, train_labels):
        fold_index = fold_index + 1
        mainDir = "D:\\Antonio\\Rev_Results\\2class_5cv_64_30sec_8heads_1depth\\" + band + "\\" + str(fold_index) + "\\"
        #ch_path = mainDir + "ch_" + band + "\\"
        res_path = mainDir + "res_" + band + "\\"
        models_path = mainDir + "mod_" + band + "\\"
        #os.makedirs(ch_path, exist_ok=True)
        os.makedirs(res_path, exist_ok=True)
        os.makedirs(models_path, exist_ok=True)

        # Dataset and Dataloader: training
        train_dataset = EEGDatasetTrain2(classIndices=class_idx, ids=train_cases[train_index], root=data_path,
                                         normalize=True, labels=train_labels[train_index], len_sec=30, fc=512,
                                         balanceData=True, classes=2)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # Dataset and Dataloader: validation
        val_dataset = EEGDatasetTrain2(classIndices=class_idx, ids=train_cases[val_index], root=data_path,
                                       normalize=True, labels=train_labels[val_index], len_sec=30, fc=512,
                                       balanceData=True, classes=2)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        # Dataset and Dataloader: validation
        te_dataset = EEGDatasetTrain2(classIndices=class_idx, ids=test_cases, root=data_path, normalize=True,
                                      labels=test_labels, len_sec=30, fc=512, balanceData=False, classes=2)
        te_dataloader = DataLoader(dataset=te_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        starttime = datetime.datetime.now()

        model = ViT(num_heads= 8, n_classes=2).cuda()
        #print(sum(p.numel() for p in model.parameters() if p.requires_grad()))


        # Optimizers
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        for epoch in range(n_epochs):
            print("ciao")
            tr_loss = 0
            train_acc = 0
            pz_train_acc = 0

            # test process
            va_loss = 0
            te_loss = 0
            va_acc = 0
            te_acc = 0
            pz_va_acc = 0
            pz_te_acc = 0

            # Arrays for training and validation AUC computation
            tr_labels = torch.tensor([], dtype=torch.long).cuda()
            tr_preds = torch.tensor([], dtype=torch.float).cuda()
            tr_scores = torch.tensor([], dtype=torch.float).cuda()
            tr_pz_list = torch.tensor([], dtype=torch.long).cuda()

            # Arrays for training and validation AUC computation
            val_labels = torch.tensor([], dtype=torch.long).cuda()
            val_preds = torch.tensor([], dtype=torch.float).cuda()
            val_scores = torch.tensor([], dtype=torch.float).cuda()
            val_pz_list = torch.tensor([], dtype=torch.long).cuda()

            # Arrays for training and validation AUC computation
            te_labels = torch.tensor([], dtype=torch.long).cuda()
            te_preds = torch.tensor([], dtype=torch.float).cuda()
            te_scores = torch.tensor([], dtype=torch.float).cuda()
            te_pz_list = torch.tensor([], dtype=torch.long).cuda()

            # training set
            model.train()
            for img, _, label, pz in train_dataloader:
                img = Variable(img.cuda().type(Tensor))
                label = Variable(label.cuda().type(LongTensor))
                pz = Variable(pz.cuda().type(LongTensor))
                tr_pz_list = torch.cat((tr_pz_list, pz), 0)
                optimizer.zero_grad()  # Reset optimizer

                tok, outputs = model(img)
                soft_out = nn.Softmax(dim=1)(outputs)

                #loss = criterion_cls(outputs, label)
                loss = criterion_cls(outputs,label)
                tr_loss += loss.item()

                loss.backward()
                optimizer.step()

                train_pred = torch.max(outputs, 1)[1]

                # Predictions and Labels append to array
                tr_labels = torch.cat((tr_labels, label), 0)
                tr_preds = torch.cat((tr_preds, train_pred), 0)
                tr_scores = torch.cat((tr_scores, soft_out), 0)

                train_acc += float((train_pred == label).cpu().numpy().astype(int).sum())

            fpr, tpr, thresholds = metrics.roc_curve(tr_labels.detach().cpu().numpy(), tr_scores[:, 1].detach().cpu().numpy())
            tr_auc = metrics.auc(fpr, tpr)

            pz_list = torch.unique(tr_pz_list)
            tr_pz_label = []
            tr_pz_preds = []
            for patient in pz_list:
                label = torch.unique(tr_labels[tr_pz_list == patient]).item()
                pred = torch.mode(tr_preds[tr_pz_list == patient]).values.item()
                tr_pz_label.append(label)
                tr_pz_preds.append(pred)

            pz_train_acc += float((np.asarray(tr_pz_preds) == np.asarray(tr_pz_label)).astype(int).sum())
            pz_train_acc = pz_train_acc / len(pz_list)

            model.eval()
            with torch.no_grad():
                for img, _, label, pz in val_dataloader:
                    img = Variable(img.cuda().type(Tensor))
                    label = Variable(label.cuda().type(LongTensor))
                    pz = Variable(pz.cuda().type(LongTensor))
                    val_pz_list = torch.cat((val_pz_list, pz), 0)

                    Tok, Cls = model(img)
                    soft_out = nn.Softmax(dim=1)(Cls)

                    #loss = criterion_cls(Cls, label)
                    loss = criterion_cls(Cls, label)
                    va_loss += loss.item()

                    y_pred = torch.max(Cls, 1)[1]

                    # Predictions and Labels append to array
                    val_labels = torch.cat((val_labels, label), 0)
                    val_preds = torch.cat((val_preds, y_pred), 0)
                    val_scores = torch.cat((val_scores, soft_out), 0)

                    va_acc += float((y_pred == label).cpu().numpy().astype(int).sum())

                fpr, tpr, thresholds = metrics.roc_curve(val_labels.detach().cpu().numpy(), val_scores[:, 1].detach().cpu().numpy())
                val_auc = metrics.auc(fpr, tpr)

                pz_list = torch.unique(val_pz_list)
                val_pz_label = []
                val_pz_preds = []
                for patient in pz_list:
                    label = torch.unique(val_labels[val_pz_list == patient]).item()
                    pred = torch.mode(val_preds[val_pz_list == patient]).values.item()
                    val_pz_label.append(label)
                    val_pz_preds.append(pred)

                pz_va_acc += float((np.asarray(val_pz_preds) == np.asarray(val_pz_label)).astype(int).sum())
                pz_va_acc = pz_va_acc / len(pz_list)

                for img, _, label, pz in te_dataloader:
                    img = Variable(img.cuda().type(Tensor))
                    label = Variable(label.cuda().type(LongTensor))
                    pz = Variable(pz.cuda().type(LongTensor))
                    te_pz_list = torch.cat((te_pz_list, pz), 0)

                    Tok, Cls = model(img)
                    soft_out = nn.Softmax(dim=1)(Cls)

                    #loss = criterion_cls(Cls, label)
                    loss = criterion_cls(Cls, label)
                    te_loss += loss.item()

                    y_pred = torch.max(Cls, 1)[1]

                    # Predictions and Labels append to array
                    te_labels = torch.cat((te_labels, label), 0)
                    te_preds = torch.cat((te_preds, y_pred), 0)
                    te_scores = torch.cat((te_scores, soft_out), 0)

                    te_acc += float((y_pred == label).cpu().numpy().astype(int).sum())

                fpr, tpr, thresholds = metrics.roc_curve(te_labels.detach().cpu().numpy(), te_scores[:, 1].detach().cpu().numpy())
                te_auc = metrics.auc(fpr, tpr)

                pz_list = torch.unique(te_pz_list)
                te_pz_label = []
                te_pz_preds = []
                for patient in pz_list:
                    label = torch.unique(te_labels[te_pz_list == patient]).item()
                    pred = torch.mode(te_preds[te_pz_list == patient]).values.item()
                    te_pz_label.append(label)
                    te_pz_preds.append(pred)

                pz_te_acc += float((np.asarray(te_pz_preds) == np.asarray(te_pz_label)).astype(int).sum())
                pz_te_acc = pz_te_acc / len(pz_list)

            print('Epoch:', epoch,
                  '; Training AUC:', tr_auc,
                  '; Validation AUC:', val_auc,
                  '; Test AUC:', te_auc,
                  '; Train loss:', tr_loss / len(train_dataloader),
                  '; Validation loss:', va_loss / len(val_dataloader),
                  '; Test loss:', te_loss / len(te_dataloader),
                  '; EP_Train accuracy:', train_acc / len(train_dataloader.dataset),
                  '; EP_Validation accuracy:', va_acc / len(val_dataloader.dataset),
                  '; EP_Test accuracy:', te_acc / len(te_dataloader.dataset),
                  '; PZ_Train accuracy:', pz_train_acc,
                  '; PZ_Validation accuracy:', pz_va_acc,
                  '; PZ_Test accuracy:', pz_te_acc,
                  )

            cm_file = mainDir + "cm_file.txt"
            with open(cm_file, 'a') as f:
                print("-----", epoch, "-----", file=f)
                print("Epoch Training CM", file=f)
                print(confusion_matrix(np.asarray(tr_labels.cpu()), np.asarray(tr_preds.cpu())), file=f)
                print("-----------------", file=f)
                print("Epoch Validation CM", file=f)
                print(confusion_matrix(np.asarray(val_labels.cpu()), np.asarray(val_preds.cpu())), file=f)
                print("Epoch Test CM", file=f)
                print(confusion_matrix(np.asarray(te_labels.cpu()), np.asarray(te_preds.cpu())), file=f)
                print("-----------------", file=f)
                print("Patient Training CM", file=f)
                print(confusion_matrix(np.asarray(tr_pz_label), np.asarray(tr_pz_preds)), file=f)
                print("-----------------", file=f)
                print("Patient Validation CM", file=f)
                print(confusion_matrix(np.asarray(val_pz_label), np.asarray(val_pz_preds)), file=f)
                print("Patient Test CM", file=f)
                print(confusion_matrix(np.asarray(te_pz_label), np.asarray(te_pz_preds)), file=f)
                print("-----------------", file=f)

            # Save epoch information
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'Train loss': tr_loss / len(train_dataloader),
                'Validation loss': va_loss / len(val_dataloader),
                'Test loss': te_loss / len(te_dataloader),
                'Train accuracy': train_acc / len(train_dataloader.dataset),
                'Validation accuracy': va_acc / len(val_dataloader.dataset),
                'Test accuracy': te_acc / len(te_dataloader.dataset),
                'Train predictions': tr_preds,
                'Validation predictions': val_preds,
                'Test predictions': te_preds,
                'Train labels': tr_labels,
                'Validation labels': val_labels,
                'Test labels': te_labels,
                'Train patients': tr_pz_list,
                'Validation patients': val_pz_list,
                'Test patients': te_pz_list,
                'Train scores': tr_scores,
                'Validation scores': val_scores,
                'Test scores': te_scores,
                'Train AUC': tr_auc,
                'Validation AUC': val_auc,
                'Test AUC': te_auc
            }, res_path + str(epoch) + '.pt')
            torch.save(model.state_dict(), models_path + str(epoch) + '.pt')  # save the model



        endtime = datetime.datetime.now()
        print(endtime - starttime)


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))