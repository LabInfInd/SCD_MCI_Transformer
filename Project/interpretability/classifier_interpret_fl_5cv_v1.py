import csv
import mne
import numpy as np
import torch
import pybv
import os
from natsort import natsorted
from torch.utils.data import DataLoader
# Import model
from Project.model.transformer_baseline import ViT
from Project.util.dataset_loader_fl import EEGDatasetTrain2
from interpret import plot_interpret

# Set the device for training
device = torch.device("cuda")

main_path = "D:\\Sibilano\\OneDrive - Politecnico di Bari\\EEG_Project\\Occhi_Chiusi\\"
label_path_file = "D:\\Sibilano\\OneDrive - Politecnico di Bari\\EEG_Project\\Labels_new.csv"

band = "EEG_Processed_new"

# Path with EEG data
data_path = main_path + band + "\\"

# Class mapping to indices B: SCD, C: MCI
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



#best model configuration: L = 30 s, k = 64, H = 8, depth = 2

batch_size = 1
k_size = 64
num_folds = 5
depths = 2
# Path with EEG data
dataPath = main_path + band + "/"

best_epochs = []
best_aucs = []

for fold in range(num_folds):
    print("Current fold: " + str(fold))
    mainDir = "D:\\Antonio\\Rev_Results\\2class_5cv_64_30sec\\" + band + "\\" + str(fold + 1) + "\\"
    res_path = mainDir + "res_" + band + "\\"

    states = os.listdir(res_path)
    states = natsorted(states)
    epochs = len(states)

    # Load models and get the epoch with highest auc on validation set
    val_aucs = []
    row = []
    for epoch in range(epochs):
        state = res_path + states[epoch]
        perf = torch.load(state)
        res = {key: perf[key] for key in perf.keys()
            & {'epoch','Train accuracy', 'Validation accuracy', 'Test accuracy', 'Validation AUC', 'Test AUC'}}

        val_aucs.append(perf['Validation AUC'])

    idx_max = np.argmax(val_aucs)
    auc_max = val_aucs[idx_max]
    print("Validation AUC: " + str(auc_max))

    best_epochs.append(idx_max)
    best_aucs.append(auc_max)


fold_index = np.argmax(best_aucs)
epoch = max(best_epochs)

print("Best fold: " + str(fold_index + 1))
print("Best epoch: " + str(epoch))

mainDir =  "D:\\Antonio\\Rev_Results\\2class_5cv_64_30sec\\" + band + "\\" + str(fold_index + 1) + "\\"
res_path = mainDir + "res_" + band + "\\"

# Dataset and Dataloader: all subjects
te_dataset = EEGDatasetTrain2(classIndices=class_idx, ids=cases, root=data_path, normalize=True,
                              labels=labels, len_sec=30, fc=512, balanceData=False, classes=2)
te_dataloader = DataLoader(dataset=te_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


gpus = [0]
model = ViT(num_heads = 8 ,n_classes=2).cuda()
#load the trained model
best_model = torch.load(res_path + '%d.pt' % epoch)
model.load_state_dict(best_model['model_state_dict'])

model = model.cuda()
model.eval()
te_labels = torch.tensor([], dtype=torch.long).cuda()
te_pz_list = torch.tensor([], dtype=torch.long)
te_eeg_pz = torch.tensor([], dtype= torch.float).cuda()

#create a mne montage based on the EEGLAB location file
montage = mne.channels.read_custom_montage("chan_loc.loc", head_size=0.095, coord_frame=None)
n_channels = len(montage.ch_names)
#this is required to create .eeg files from raw binary data and import them in Letswave7
fake_info = mne.create_info(ch_names=montage.ch_names, sfreq=512., ch_types='eeg')
save_path = 'custom_path\\'


with torch.no_grad():
    # inference
    for depth in range(depths):
        ii = 0
        for i, (img, img2, label, pz) in enumerate(te_dataloader):
            img = img.to(device).type(torch.float)
            img2 = img2.to(device).type(torch.float)
            label = label.to(device).type(torch.long)
            Tok, Cls = model(img)
            #the model saves trained attention matrices for each block
            att = torch.load(".\\att" + str(depth) +".pt")

            #avrage attention weights along attention heads
            att = np.mean(att.detach().cpu().numpy(), axis=1)

            feat_eeg = np.zeros((15360,))
            n_patches = len(feat_eeg) / k_size
            for k in range(int(n_patches)):
                #get attention scores corresponding to the first row (CLS*)
                feat_eeg[k * k_size:(k + 1) * k_size] = att[0, 0, k+1]#.detach().cpu().numpy() #batch, CLS*, weights
            #t = np.arange(0, 1, 1 / 512)
            t1 = np.arange(0, 30, 1 / 512)
            eeg = img[:, :, 0, :].squeeze().detach().cpu().numpy()
            eeg2 = img2[:, :, 0, :].squeeze().detach().cpu().numpy() #non-normalized signal input
            eeg3 = img.squeeze().detach().cpu().numpy()
            #get the patch with highest attention scores
            times = np.argmax(feat_eeg)
            #get the corresponding time in the EEG epoch
            max_time = t1[times]

            #estraggo finestra segnale in corrispondenza dei pesi di attention maggiori
            if (times + k_size + 224) < img2.shape[3] and (times - 224) > 0:

                ###uncomment this section for sample plots of attention scores over 5-s window (Fig. 3 in the paper)
                # w_start = 5120
                # w_length = 5
                # t = t1[w_start:w_length*512]
                #
                # eeg_window_raw = img2[:, :, 3, w_start:w_length*512].squeeze().detach().cpu().numpy() #one channel
                # eeg_window_norm = img[:, :, 3, w_start:w_length*512].squeeze().detach().cpu().numpy()
                #
                # plot_interpret(t, eeg_window_norm, eeg_window_raw, feat_eeg[w_start:w_length*512], ii, label=pz.item(),
                #            save_path='.\\')

                ii += 1
                #extract the 1-s long window centered on the patch
                eeg_window = img2[:, :, :, times-224 : times + k_size+224]
                te_eeg_pz = torch.cat((te_eeg_pz, eeg_window), 0)
                te_labels = torch.cat((te_labels, label), 0)
                te_pz_list = torch.cat((te_pz_list, pz), 0)



        #concatenate extracted windows for each patient to create one time series
        pz_list = torch.unique(te_pz_list)
        for patient in pz_list:
            label = torch.unique(te_labels[te_pz_list == patient]).item()
            eeg_att = te_eeg_pz[te_pz_list == patient].detach().cpu().numpy()
            epoch_list = []
            for epoch in eeg_att:
                ep = mne.io.RawArray(epoch[0], fake_info)
                epoch_list.append(ep)
            eeg_epochs = mne.concatenate_raws(epoch_list)


            #save files for SCD and MCI
            if label == 0:
                pybv.write_brainvision(data= eeg_epochs[:][0], sfreq= 512, unit= 'V', ch_names= montage.ch_names, fname_base= 'SCD_eeg_att_' + str(patient.item()), folder_out= save_path + 'depth' + str(depth) +'\\SCD\\')
            else:
                pybv.write_brainvision(data= eeg_epochs[:][0], sfreq= 512, unit= 'V', ch_names= montage.ch_names, fname_base= 'MCI_eeg_att_' + str(patient.item()), folder_out= save_path + 'depth' + str(depth) + '\\MCI\\')
