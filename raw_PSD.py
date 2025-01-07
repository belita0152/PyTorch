# Library
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_welch
import pandas as pd



# Exporting raw PSD data

# psd 출력하는 함수 정의
def total_psd( fname ):
    # freq band idxs
    delta_idx = np.array(np.where(np.logical_and(freqs >= 0.5, freqs <= 4))).ravel()
    delta_idx = delta_idx.tolist()  # delta_idx 그냥 뽑아내면, tuple 형으로 나와 뒤에서 slicing 불가. 또한 [[1 2]] 형태로 출력되기 때문에 indexing 불가.
    theta_idx = np.array(np.where(np.logical_and(freqs >= 4, freqs <= 8))).ravel()
    theta_idx = theta_idx.tolist()
    alpha_idx = np.array(np.where(np.logical_and(freqs >= 8, freqs <= 13))).ravel()
    alpha_idx = alpha_idx.tolist()
    beta_idx = np.array(np.where(np.logical_and(freqs >= 13, freqs <= 25))).ravel()
    beta_idx = beta_idx.tolist()
    gamma_idx = np.array(np.where(np.logical_and(freqs >= 25, freqs <= 45))).ravel()
    gamma_idx = gamma_idx.tolist()

    # 전체 psds에서 electrode 별로 psd 추출
    # 전체 psds에서 electrode 별로 추출
    col_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    row_names = ['delta_{}'.format(sub_num), 'theta_{}'.format(sub_num), 'alpha_{}'.format(sub_num),
                 'beta_{}'.format(sub_num), 'gamma_{}'.format(sub_num)]
    total_psd = pd.DataFrame(index=row_names, columns=col_names)

    for i in range(19):
        electrode_psd = psds[i]
        # freq band 별로 psd 값 추출.
        electrode_delta_values = electrode_psd[delta_idx[0]:delta_idx[-1] + 1]
        electrode_theta_values = electrode_psd[theta_idx[0]:theta_idx[-1] + 1]
        electrode_alpha_values = electrode_psd[alpha_idx[0]:alpha_idx[-1] + 1]
        electrode_beta_values = electrode_psd[beta_idx[0]:beta_idx[-1] + 1]
        electrode_gamma_values = electrode_psd[gamma_idx[0]:gamma_idx[-1] + 1]

        # psd 값 sum
        electrode_delta_psd = np.sum(electrode_delta_values, axis=0)
        electrode_theta_psd = np.sum(electrode_theta_values, axis=0)
        electrode_alpha_psd = np.sum(electrode_alpha_values, axis=0)
        electrode_beta_psd = np.sum(electrode_beta_values, axis=0)
        electrode_gamma_psd = np.sum(electrode_gamma_values, axis=0)

        # 마지막 출력값을 df에 저장
        output = [electrode_delta_psd, electrode_theta_psd, electrode_alpha_psd, electrode_beta_psd,
                  electrode_gamma_psd]
        total_psd[col_names[i]] = output

    return total_psd

# Exporting raw AD-PSD data

# sub001- ~ sub-009 에 대해 함수 이용해서 출력.
for sub_num in range(1, 10):
    fname = "./data/derivatives/sub-00{}/eeg/sub-00{}_task-eyesclosed_eeg.set".format(sub_num, sub_num)
    der_sub = mne.io.read_raw_eeglab(fname, preload=False)
    psds, freqs = psd_array_welch(der_sub._data, sfreq=500)

    # print(total_psd(fname))
    result_under10 = total_psd(fname)
    # result_under10.to_csv("AD_sub001-009_using function.csv", mode='a')

# sub010- ~ sub-036 에 대해 함수 이용해서 출력.
for sub_num in range(10, 37):
    path = "./data/derivatives/sub-0{}/eeg/sub-0{}_task-eyesclosed_eeg.set".format(sub_num, sub_num)
    der_sub = mne.io.read_raw_eeglab(path, preload=False)
    psds, freqs = psd_array_welch(der_sub._data, sfreq=500)

    # print(total_psd(path))
    result_over10 = total_psd(path)
    # result_over10.to_csv("AD_sub010-036_using function.csv", mode='a')

psd_under10 = pd.read_csv("AD_sub001-009_using function.csv")
psd_over10 = pd.read_csv("AD_sub010-036_using function.csv")
total_AD = pd.concat([psd_under10, psd_over10], axis=0)  # for문 돌리면서 바로 하나의 df로 저장하는 방법을 모르겠음. 따로 csv로 저장한 후 합침.
print(total_AD.shape)  # 214 x 20  # 불필요한 df의 col names 행이 추가되어 df가 복잡해짐. col names 행을 제거해야 함.

# df 정리

idx = [x for x in range(214) if x % 6 == 5]  # 디버깅->View as DataFrame으로 확인해본 결과, index=6n-1일 때 col names 행이 존재. 이를 제거함.
mod_total_AD = total_AD.drop(index=idx, axis=0)

mod_total_AD.to_csv("AD group_psd_using functions.csv")



# Exporting raw FTD-PSD data

# 전체 FTD group (sub066- ~ sub-088) 에 대해, 함수 이용해서 출력. CN도 sub 번호만 변경하여 동일하게 진행.
for sub_num in range(66, 89):
    fname = "./data/derivatives/sub-0{}/eeg/sub-0{}_task-eyesclosed_eeg.set".format(
        sub_num, sub_num)
    der_sub = mne.io.read_raw_eeglab(fname, preload=False)
    psds, freqs = psd_array_welch(der_sub._data, sfreq=500)

    print(total_psd(fname))
    result = total_psd(fname)
    result.to_csv("FTD_sub066-088_using function.csv", mode='a')  # for문 돌리면서 바로 df로 저장하는 방법 모르겠음..

total_FTD = pd.read_csv("FTD_sub066-088_using function.csv")
print(total_FTD.shape)

# df 정리

idx = [x for x in range(137) if x % 6 == 5]  # 디버깅->View as DataFrame으로 확인해본 결과, index=6n-1일 때 col names 행이 존재. 이를 제거함.
mod_total_FTD = total_FTD.drop(index=idx, axis=0)

mod_total_FTD.to_csv("FTD group_psd_using functions.csv")  # index 번호 바꿔야 하는데 잘 모르게쑴… 그리고 unnamed col을 index로 바꾸고 싶음.



# Exporting raw CN-PSD data

# 전체 CN group (sub037- ~ sub-065) 에 대해, 함수 이용해서 출력
for sub_num in range(37, 66):
    fname = "./data/derivatives/sub-0{}/eeg/sub-0{}_task-eyesclosed_eeg.set".format(
        sub_num, sub_num)
    der_sub = mne.io.read_raw_eeglab(fname, preload=False)
    psds, freqs = psd_array_welch(der_sub._data, sfreq=500)

    print(total_psd(fname))
    result = total_psd(fname)
    result.to_csv("CN_sub036-065_using function.csv", mode='a')  # for문 돌리면서 바로 df로 저장하는 방법 모르겠음..

total_CN = pd.read_csv("CN_sub036-065_using function.csv")
print(total_CN.shape)

# df 정리

idx = [x for x in range(len(total_CN[:, 0])) if x % 6 == 5]
mod_total_CN = total_CN.drop(index=idx, axis=0)

mod_total_CN.to_csv("CN group_psd_using functions.csv")






# raw PSD topomap

# Library
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_welch
import pandas as pd


# Data - 3 groups
fname = "./data/derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
der_sub_001 = mne.io.read_raw_eeglab(fname, preload=False)

mod_total_AD = pd.read_csv("AD group_psd.csv", index_col=0)
mod_total_FTD = pd.read_csv("FTD group_psd_using functions.csv", index_col=0)
mod_total_FTD.reset_index()
mod_total_CN = pd.read_csv("CN group_psd_using functions.csv", index_col=0)

# PSD topomap
mod_total_AD = mod_total_AD.to_numpy()
mod_total_AD = np.delete(mod_total_AD, 0, axis=1)  # freq band 나타내는 열이 [0]에 존재. 이를 제거.
mod_total_FTD = mod_total_FTD.to_numpy()
mod_total_FTD = np.delete(mod_total_FTD, 0, axis=1)
mod_total_CN = mod_total_CN.to_numpy()
mod_total_CN = np.delete(mod_total_CN, 0, axis=1)



fig, axes = plt.subplots(5, 3)
for remainder in range(0, 5):
    # index가 5n, +1, +2, +3, +4 일 때 각각 D, Th, A, B, G freq band에 해당.
    idx_AD = [x for x in range(180) if x % 5 == remainder]  # AD = 전체 180개 행. 디버깅으로 확인.
    AD_freq_band = np.sum(mod_total_AD[idx_AD], axis=0)  # 모든 sub에 대해 각 band일 때의 값을 sum. 이때 열 별로 => axis=0.

    idx_FTD = [x for x in range(115) if x % 5 == remainder]  # FTD = 전체 115개 행. 디버깅으로 확인.
    FTD_freq_band = np.sum(mod_total_FTD[idx_FTD], axis=0)

    idx_CN = [x for x in range(145) if x % 5 == remainder]  # CN = 전체 145개 행. 디버깅으로 확인.
    CN_freq_band = np.sum(mod_total_CN[idx_CN], axis=0)

    mne.viz.plot_topomap(AD_freq_band, der_sub_001.info, ch_type='eeg', axes=axes[remainder, 0], show=False)
    mne.viz.plot_topomap(CN_freq_band, der_sub_001.info, ch_type='eeg', axes=axes[remainder, 1], show=False)
    mne.viz.plot_topomap(FTD_freq_band, der_sub_001.info, ch_type='eeg', axes=axes[remainder, 2], show=False)

plt.show()