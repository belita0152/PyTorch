from torch.utils.data import Dataset
import numpy as np
import torch
from utils.utils import base_path, total_path


class TorchDataset(Dataset):  # Parsing + ToTensor
    def __init__(self, mode, transform=None):
        self.ch_names = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']
        self.event_name = ['Sleep stage W',
                         'Sleep stage 1',
                         'Sleep stage 2',
                         'Sleep stage 3',
                         'Sleep stage 4',
                         'Sleep stage R']
        self.second = 30  # duration
        self.sfreq = 100  # sampling rate

        self.train_paths, self.val_paths, self.eval_paths = total_path['train_paths'], total_path['val_paths'], total_path['eval_paths']

        if mode == 'train':
            paths = self.train_paths  # x : (57850, 3, 3000)
        elif mode == 'val':
            paths = self.val_paths  # x : (43808, 3, 3000)
        elif mode == 'eval':
            paths = self.eval_paths  # x : (27365, 3, 3000)
        elif mode == 'all':
            paths = base_path

        self.data_x, self.data_y = self.parser(paths)

    def parser(self, paths_):
        total_x, total_y = [], []
        for path in paths_:
            data = np.load(path)
            x, y = data['x'], data['y']
            total_x.append(x)
            total_y.append(y)
        total_x, total_y = np.concatenate(total_x), np.concatenate(total_y)

        return total_x, total_y

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, item):  # 샘플마다 데이터 로드
        x = torch.tensor(self.data_x[item], dtype=torch.float32)
        y = torch.tensor(self.data_y[item], dtype=torch.long)  # class label -> 정수

        return x, y


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = TorchDataset('all')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for batch in dataloader:
        x, y = batch
        print(x.shape, y.shape)
