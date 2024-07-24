# Library
import pickle
import os
import numpy as np
import torch
import mne
from mne.time_frequency import psd_array_welch
from mne_connectivity import spectral_connectivity_epochs
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Dataset
target_freqs = np.arange(1, 46, 1)


def node_feature(raw, sfreq=500):
    psds, freqs = psd_array_welch(raw._data, sfreq)
    idxs = [np.argmin(np.abs(freqs - tf)) for tf in target_freqs]
    psd_value = torch.tensor(psds[:, idxs], dtype=torch.long)

    return psd_value  # PSD


def epoching_fc(raw):
    eeg_info = mne.create_info(ch_names=raw.ch_names, sfreq=raw.info['sfreq'], ch_types='eeg')
    data = mne.io.RawArray(raw._data, info=eeg_info)
    epoched = mne.make_fixed_length_epochs(raw=data, duration=2)
    epoch_len = epoched.get_data().shape[0]
    epoched_data = epoched.get_data()[np.random.choice(epoch_len, 50), :, :int(2 * raw.info['sfreq'])]
    con = spectral_connectivity_epochs(data=epoched_data, names=raw.ch_names, method='coh', sfreq=raw.info['sfreq'],
                                       fmin=0.5, fmax=45, faverage=True, mt_adaptive=True, n_jobs=1)
    conn = con.get_data(output='dense')[:, :, 0]
    adj_matrix = (conn + conn.T)
    edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)  # FC
    edge_attr = torch.tensor(adj_matrix)
    # edge_weight = torch.tensor(adj_matrix[edge_index[0], edge_index[1]])
    # return edge_index, edge_weight

    return edge_index, edge_attr


# Data
data_list = []
for sub_num in range(1, 89):
    fname = "./venv/data/derivatives/{}/eeg/{}_task-eyesclosed_eeg.set".format(f"sub-0{sub_num:02d}", f"sub-0{sub_num:02d}")
    raw = mne.io.read_raw_eeglab(fname, preload=False)

    x = node_feature(raw)
    edge_index, edge_attr = epoching_fc(raw)

    if sub_num <= 36:
        y = torch.tensor([0], dtype=torch.long)  # 0 = AD
    elif sub_num <= 65:
        y = torch.tensor([1], dtype=torch.long)  # 1 = CN
    else:
        y = torch.tensor([2], dtype=torch.long)  # 2 = FTD

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data_list.append(data)


# Save data_list
dir = os.getcwd()
save_path = os.path.join(dir, 'data_list.pkl')

with open(save_path, 'wb') as f:
    pickle.dump(data_list, f)



