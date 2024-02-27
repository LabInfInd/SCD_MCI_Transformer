import csv
import math
import mne
import numpy as np
import torch
import pybv
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import LeaveOneOut
# Import model
from model.transformer_baseline import ViT
from util.dataset_loader_fl import EEGDatasetTrain2
from interpret import plot_interpret
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.autograd import Variable

# Set the device for training
device = torch.device("cuda")  # use cuda as alternative

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

#
# skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

all_pred_train = dict.fromkeys(band, [])
all_pred_validation = dict.fromkeys(band, [])
all_pred_test = dict.fromkeys(band, [])

p_pred_train = dict.fromkeys(band, [])
p_pred_validation = dict.fromkeys(band, [])
p_pred_test = dict.fromkeys(band, [])

tr_Y_true = dict.fromkeys(band, [])
tr_Y_pred = dict.fromkeys(band, [])

val_Y_true = dict.fromkeys(band, [])
val_Y_pred = dict.fromkeys(band, [])

# te_Y_true = dict.fromkeys(band, [])
# te_Y_pred = dict.fromkeys(band, [])
#
# # Compute ROC curve and ROC area for each class
# tr_fpr = dict.fromkeys(band, [])
# tr_tpr = dict.fromkeys(band, [])
# tr_roc_auc = dict.fromkeys(band, [])
# tr_thresholds = dict.fromkeys(band, [])
tr_cm = dict.fromkeys(band, [])
#
# val_fpr = dict.fromkeys(band, [])
# val_tpr = dict.fromkeys(band, [])
# val_roc_auc = dict.fromkeys(band, [])
# val_thresholds = dict.fromkeys(band, [])
val_cm = dict.fromkeys(band, [])
#
# te_fpr = dict.fromkeys(band, [])
# te_tpr = dict.fromkeys(band, [])
# te_roc_auc = dict.fromkeys(band, [])
# te_thresholds = dict.fromkeys(band, [])
# te_cm = dict.fromkeys(band, [])

opt_thresholds = dict.fromkeys(band, [])

accuracies = {}

#model = ViT().to(device)

batch_size = 1
n_epochs = 1
k_size = 64 #1024
# Path with EEG data
dataPath = main_path + band + "/"
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

fold_index = 5 #1
#for train_index, val_index in kf.split(train_cases, train_labels):
print("Current fold: " + str(fold_index))
mainDir =  "D:\\Antonio\\Rev_Results\\2class_5cv_64_30sec\\" + band + "\\" + str(fold_index) + "\\"
res_path = mainDir + "res_" + band + "\\"
models_path = mainDir + "mod_" + band + "\\"

# # Dataset and Dataloader: training
# train_dataset = EEGDatasetTrain2(classIndices=class_idx, ids=train_cases[train_index], root=data_path,
#                                  normalize=True, labels=train_labels[train_index], len_sec=30, fc=512,
#                                  balanceData=True, classes=2)
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#
# # Dataset and Dataloader: validation
# val_dataset = EEGDatasetTrain2(classIndices=class_idx, ids=train_cases[val_index], root=data_path,
#                                normalize=True, labels=train_labels[val_index], len_sec=30, fc=512,
#                                balanceData=True, classes=2)
# val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# Dataset and Dataloader: validation
te_dataset = EEGDatasetTrain2(classIndices=class_idx, ids=cases, root=data_path, normalize=True,
                              labels=labels, len_sec=30, fc=512, balanceData=False, classes=2)
te_dataloader = DataLoader(dataset=te_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

##testtr_Y_true[band] = [None] * len(train_dataloader)
#testtr_Y_pred[band] = [None] * len(train_dataloader)

# val_Y_true[band] = [None] * len(validation_dataloader)
# val_Y_pred[band] = [None] * len(validation_dataloader)

# te_Y_true[band] = [None] * len(test_dataloader)
# te_Y_pred[band] = [None] * len(test_dataloader)
gpus = [0]
model = ViT(num_heads=1 ,n_classes=2).cuda()
#model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])
model = model.cuda()
# Aprire i res per ogni epoca
val_acc = []
tr_auc = []
val_auc = []
#for epoch in range(epochs):
   # perf = torch.load(res_path + str(epoch) + ".pt")
    #tr_auc.append(perf['Train accuracy'])
    #val_auc.append(perf['Validation accuracy'])

# Get max of auc on validation with the highest auc on training
#val_auc = np.asarray(val_auc)
#max_val_auc_idx = np.argmax(val_auc)

epoch = 42#58 #11 #157
a = torch.load(res_path + '%d.pt' % epoch)
model.load_state_dict(a['model_state_dict'])
#model.load_state_dict(torch.load(models_path + '%d.pt' % epoch))

# params = sum(p.numel() for p in model.parameters() if p.requires_grad)
model.eval()
te_labels = torch.tensor([], dtype=torch.long).cuda()
te_pz_list = torch.tensor([], dtype=torch.long)
te_eeg_pz = torch.tensor([], dtype= torch.float).cuda()
montage = mne.channels.read_custom_montage("chan_loc.loc", head_size=0.095, coord_frame=None)
n_channels = len(montage.ch_names)
fake_info = mne.create_info(ch_names=montage.ch_names, sfreq=512., ch_types='eeg')
#att = np.mean(att.detach().cpu().numpy(),axis=1)
with torch.no_grad():
    # train
    ii = 0
    for i, (img, img2, label, pz) in enumerate(te_dataloader):
        img = img.to(device).type(torch.float)
        img2 = img2.to(device).type(torch.float)
        label = label.to(device).type(torch.long)
        #te_labels = torch.cat((te_labels, label), 0)

        #te_pz_list = torch.cat((te_pz_list, pz), 0)
        Tok, Cls = model(img)
        #attention relativa a input
        att = torch.load("D:\\SCD_MCI_ENV\\Projects\\git\\SCD_MCI_Classification\\Project\\interpretability\\att1.pt")
        #RIMETTERE MEDIA
        att = np.mean(att.detach().cpu().numpy(), axis=1)
        #att = np.sum(att, axis= 1)
        feat_eeg = np.zeros((15360,))

        for k in range(240):
            feat_eeg[k * k_size:(k + 1) * k_size] = att[0, 0, k+1]#.detach().cpu().numpy() #att[0, 7, k+1]# #ATTENZIONE SINGOLE HEAD
        #t = np.arange(0, 1, 1 / 512)
        t1 = np.arange(0, 30, 1 / 512)
        eeg = img[:, :, 0, :].squeeze().detach().cpu().numpy()
        eeg2 = img2[:, :, 0, :].squeeze().detach().cpu().numpy()
        eeg3 = img.squeeze().detach().cpu().numpy()
        times = np.argmax(feat_eeg)
        max_time = t1[times]
        #plot_interpret(t, eeg, eeg2, eeg3, feat_eeg, times, ii, label=pz.item(),
         #              save_path='D:\\SCD_MCI_ENV\\Projects\\git\\SCD_MCI_Classification\\Project\\interpretability\\figs\\')
        #ii += 1
        #estraggo finestra segnale in corrispondenza dei pesi di attention maggiori
        #if  (times + 2* k_size) < img2.shape[3] and (times-k_size) > 0 :
        if (times + k_size + 224) < img2.shape[3] and (times - 224) > 0:
            #eeg_window = img2[:, :, :, times - k_size: times + 2*k_size]
            #t = t1[times - 224: times + k_size + 224]
            # t = t1[5120:7680]
            # eeg_window3 = img2[:, :, :, 5120:7680].squeeze().detach().cpu().numpy()
            # eeg_window = img2[:, :, 3, 5120:7680].squeeze().detach().cpu().numpy()
            # eeg_window_norm = img[:, :, 3, 5120:7680].squeeze().detach().cpu().numpy()
        #eeg_window = img2[:, :, :, :]

            #plot_interpret(t, eeg_window_norm, eeg_window, eeg3, feat_eeg[5120:7680], times, ii, max_time, label=pz.item(),
                       #save_path='D:\\SCD_MCI_ENV\\Projects\\git\\SCD_MCI_Classification\\Project\\interpretability\\figs\\5s_new2\\')
            ii += 1
            eeg_window = img2[:, :, :, times-224 : times + k_size+224]
            te_eeg_pz = torch.cat((te_eeg_pz, eeg_window), 0)
            te_labels = torch.cat((te_labels, label), 0)
            te_pz_list = torch.cat((te_pz_list, pz), 0)
    #
    #
    #     #plot


    pz_list = torch.unique(te_pz_list)
    for patient in pz_list:
        label = torch.unique(te_labels[te_pz_list == patient]).item()
        eeg_att = te_eeg_pz[te_pz_list == patient].detach().cpu().numpy()
        epoch_list = []
        for epoch in eeg_att:
            ep = mne.io.RawArray(epoch[0], fake_info)
            epoch_list.append(ep)
        eeg_epochs = mne.concatenate_raws(epoch_list)

        #eeg_att = mne.io.RawArray(eeg_epochs, fake_info)
        if label == 0:
            #eeg_epochs.save('D:\\Sibilano\\OneDrive - Politecnico di Bari\\EEG_Project\\Nuovo_Lavoro\\ep_att_512_30s\\'+ str(label) + '\\SCD_eeg_att_' + str(patient.item()) + '_raw.fif',  fmt = 'single')
            pybv.write_brainvision(data= eeg_epochs[:][0], sfreq= 512, unit= 'V', ch_names= montage.ch_names, fname_base= 'SCD_eeg_att_' + str(patient.item()), folder_out= 'D:\\Sibilano\\OneDrive - Politecnico di Bari\\EEG_Project\\Nuovo_Lavoro\\Final\\[all]ep_att_self\\depth1\\SCD\\')
            #eeg_epochs.export('D:\\Sibilano\\OneDrive - Politecnico di Bari\\EEG_Project\\Nuovo_Lavoro\\ep_att_512_30s\\'+ str(label) + '\\SCD_eeg_att_' + str(patient.item()),  fmt = 'eeglab')
        else:
            pybv.write_brainvision(data= eeg_epochs[:][0], sfreq= 512, unit= 'V', ch_names= montage.ch_names, fname_base= 'MCI_eeg_att_' + str(patient.item()), folder_out= 'D:\\Sibilano\\OneDrive - Politecnico di Bari\\EEG_Project\\Nuovo_Lavoro\\Final\\[all]ep_att_self\\depth1\\MCI\\')

            # eeg_epochs.export('D:\\Sibilano\\OneDrive - Politecnico di Bari\\EEG_Project\\Nuovo_Lavoro\\ep_att_512_30s\\'+ str(label)  + '\\MCI_eeg_att_' + str(patient.item()), fmt = 'eeglab')
            # eeg_epochs.save('D:\\Sibilano\\OneDrive - Politecnico di Bari\\EEG_Project\\Nuovo_Lavoro\\ep_att_64_30sec\\'+ str(label) + '\\eeg_att_' + str(patient.item()) + '.fif', fmt = 'double')
            # torch.save((te_eeg_pz[te_pz_list == patient]).values(), "eeg_att" + str(patient) + ".")
            # pred = torch.mode(te_eeg_pz[te_pz_list == patient]).values.item()

            #train_y_pred = nn.Softmax(dim=1)(Cls)[0][1]
            #train_y_pred = torch.argmax(nn.Softmax(dim=1)(Cls), dim=1) #(nn.Softmax(dim=1)(Cls))[:, 1]

            #tr_Y_true[band][i] = label.cpu().detach().numpy()
            #tr_Y_pred[band][i] = (train_y_pred.cpu().detach().numpy(), patient)

        # validation
        # for i, (img, label, patient) in enumerate(validation_dataloader):
        #     img = img.to(device).type(torch.float)
        #     label = label.to(device).type(torch.long)
        #     Tok, Cls = model(img)
        #
        #     #validation_y_pred = nn.Softmax(dim=1)(Cls)[0][1]
        #     validation_y_pred = torch.argmax(nn.Softmax(dim=1)(Cls), dim=1)#(nn.Softmax(dim=1)(Cls))[:, 1]
        #
        #     val_Y_true[band][i] = label.cpu().detach().numpy()
        #     val_Y_pred[band][i] = (validation_y_pred.cpu().detach().numpy(), patient)

        # for i, (img, label, patient) in enumerate(test_dataloader):
        #     img = img.to(device).type(torch.float)
        #     label = label.to(device).type(torch.long)
        #     Tok, Cls = model(img)
        #
        #     test_y_pred = nn.Softmax(dim=1)(Cls)[0][1]
        #
        #     te_Y_true[band][i] = label.cpu().detach().numpy()
        #     te_Y_pred[band][i] = (test_y_pred.cpu().detach().numpy(), patient)

