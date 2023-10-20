from typing import Callable


from torch.utils.data import Dataset


class SplitXDataset(Dataset):
    def __init__(self, original_dataset: Dataset, spliter: Callable = None):
        self.original_dataset = original_dataset
        spliter = spliter or (lambda x: [x])
        self.spliter = spliter

    def __getitem__(self, index):
        x, y = self.original_dataset[index]
        xs = self.spliter(x)
        return xs, y

    def __len__(self):
        return len(self.original_dataset)


if __name__ == '__main__':
    from deep_utils.datasets import Mnist

    from torch.utils.data import DataLoader

    mnist_train, mnist_test = Mnist.get()
    split_mnist = SplitXDataset(mnist_train, lambda x: [x[:, 0], x[:, 1:]])
    mnist_loader = DataLoader(split_mnist, batch_size=32)
    batch = next(iter(mnist_loader))
    print(batch[0].shape, batch[1].shape)
