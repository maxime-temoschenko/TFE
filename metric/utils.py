from torch.utils.data import Dataset, DataLoader, ConcatDataset
from TFE.utils import *
from pathlib import Path

class ClassificationDataset(Dataset):
    def __init__(self, dataset, label: int):
        self.dataset = dataset
        self.label = label
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        sample, _ = self.dataset[idx]
        return sample, self.label

def getDataLoader(process_path : str, samples_path : str, batch_size = 32):
    process_file = Path(process_path)
    samples_file = Path(samples_path)
    samples_dataset = BatchDataset(file=samples_file, data_keyword='samples')
    process_dataset =  SequenceDataset(file=process_file, window=12, flatten=True, slicer=slice(len(samples_dataset)))
    train_datasets = [ClassificationDataset(process_dataset, 0), ClassificationDataset(samples_dataset, 1)]
    dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader





