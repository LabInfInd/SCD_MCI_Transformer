from torch import Tensor
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from matplotlib.collections import LineCollection
import mne






def plot_interpret(x,y, y2, dydx, i, signal_type = "EEG", label = 0, save_path = None):


    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    points2 = np.array([x, y2]).T.reshape(-1, 1, 2)
    segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)

    fig = plt.figure(figsize = (35,25))

    ax0 = plt.subplot(211)
    ax1 = plt.subplot(212)
    plt.subplots_adjust(hspace=0.5)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='YlOrRd', norm=norm)
    lc2 = LineCollection(segments2, cmap='YlOrRd', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(4)
    line = ax0.add_collection(lc)

    lc2.set_array(dydx)
    lc2.set_linewidth(4)
    line2 = ax1.add_collection(lc2)
    cbar1 = fig.colorbar(line, ax=ax0)
    cbar2 = fig.colorbar(line2, ax=ax1)
    cbar1.ax.tick_params(labelsize = 44)
    cbar2.ax.tick_params(labelsize=44)

    ax0.set_title(f"Attention scores over z-score normalized signal", fontsize = 58, pad = 30)
    ax1.set_title(f"Attention scores over raw signal", fontsize = 58, pad = 30)

    ax0.set_xlim(x.min(), x.max())
    ax0.set_ylim(-3,3)
    ax0.set_xlabel('Time (s)', fontsize = 48)
    ax0.set_ylabel('Amplitude (μV)', fontsize = 48)
    ax0.tick_params(labelsize=44, length= 5, pad = 20, top = True)
    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(-15,15)
    ax1.set_xlabel('Time (s)', fontsize = 48)
    ax1.set_ylabel('Amplitude (μV)', fontsize = 48)
    ax1.tick_params(labelsize=44,length= 5, pad = 20, top = True)

    #plt.show()
    if save_path:
        fig.savefig(os.path.join(save_path,f"{signal_type}_{label}_{i}.png"))



