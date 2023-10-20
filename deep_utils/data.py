import numpy as np
from typing import List, Callable
from torch.utils.data import Dataset, DataLoader


class SubDataset(Dataset):
    def __init__(self, original_dataset: Dataset, indices: List[int], randomize_factor: int = 1):
        indices = list(indices)
        self.original_dataset = original_dataset
        self.randomize_factor = randomize_factor
        self.indices = indices * randomize_factor
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        return self.original_dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset: Dataset, portion: float=0.9):
    dataset_size = len(dataset)
    random_indices = np.random.permutation(dataset_size)
    return \
        SubDataset(dataset, random_indices[:round(dataset_size * portion)]), \
        SubDataset(dataset, random_indices[round(dataset_size * portion):])



class IdentityDataset(Dataset):
    def __init__(self, original_dataset: Dataset):
        self.original_dataset = original_dataset

    def __getitem__(self, index):
        x, y = self.original_dataset[index]
        return x, x

    def __len__(self):
        return len(self.original_dataset)


class TransformationDataset(Dataset):
    def __init__(self, original_dataset: Dataset, transformation: Callable):
        self.original_dataset = original_dataset
        self.transformation = transformation

    def __getitem__(self, index: int):
        data_batch = self.original_dataset[index]
        return self.transformation(data_batch)

    def __len__(self):
        return len(self.original_dataset)
