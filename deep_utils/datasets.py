from typing import Tuple

import pickle

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import MNIST as _MNIST, \
    CIFAR10 as _CIFAR10, CIFAR100 as _CIFAR100, \
    FashionMNIST as _FashionMNIST, ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, Resize, CenterCrop


from deep_utils.data import TransformationDataset


data_root = Path("/home/zf/projects/data")


class Mnist:
    @staticmethod
    def get(txs: list = None, tys: list = None) -> (Dataset, Dataset):
        """
        :param txs: xs transforms: Default: Converting to tensors within range [0, 1]!!!
        :param tys: ys transforms
        :return:
        """
        txs = txs or []
        tys = tys or []
        mnist_train = _MNIST((data_root / "mnist").as_posix(),
                             transform=Compose([ToTensor()] + txs),
                             target_transform=Compose(tys), download=True)
        mnist_test = _MNIST((data_root / "mnist").as_posix(), train=False,
                            transform=Compose([ToTensor()] + txs),
                            target_transform=Compose(tys), download=True)
        return mnist_train, mnist_test

    @staticmethod
    def tx_flatten(x: torch.Tensor):
        return x.view(28 * 28)

    @staticmethod
    def ty_onehot(y: torch.Tensor):
        return torch.eye(10)[y]


class FashionMnist:
    @staticmethod
    def get(txs: list = None, tys: list = None) -> (Dataset, Dataset):
        """
        :param txs: xs transforms: Default: Converting to tensors within range [0, 1]!!!
        :param tys: ys transforms
        :return:
        """
        txs = txs or []
        tys = tys or []
        mnist_train = _FashionMNIST((data_root / "fashion-mnist").as_posix(),
                                    transform=Compose([ToTensor()] + txs),
                                    target_transform=Compose(tys), download=True)
        mnist_test = _FashionMNIST((data_root / "fashion-mnist").as_posix(), train=False,
                                    transform=Compose([ToTensor()] + txs),
                                    target_transform=Compose(tys), download=True)
        return mnist_train, mnist_test

    @staticmethod
    def tx_flatten(x: torch.Tensor):
        return x.view(28 * 28)

    @staticmethod
    def ty_onehot(y: torch.Tensor):
        return torch.eye(10)[y]


class Cifar10:
    @staticmethod
    def get(txs_train: list = None, tys: list = None) -> (Dataset, Dataset):
        """
        :param txs_train: xs transforms: Default: Converting to tensors within range [0, 1]!!!
        :param tys: ys transforms
        :return:
        """
        txs_train = txs_train or []
        tys = tys or []
        cifar10_train = _CIFAR10((data_root / "cifar10").as_posix(),
                                 transform=Compose([ToTensor()] + txs_train),
                                 target_transform=Compose(tys))
        cifar10_test = _CIFAR10((data_root / "cifar10").as_posix(), train=False,
                                transform=Compose([ToTensor()]),
                                target_transform=Compose(tys))
        return cifar10_train, cifar10_test

    @staticmethod
    def tx_flatten(x: torch.Tensor):
        return x.view(3 * 32 * 32)

    @staticmethod
    def get_txs_random_transform():
        return [RandomHorizontalFlip(p=0.5), RandomCrop(32, padding=4)]

    @staticmethod
    def ty_onehot(y: torch.Tensor):
        return torch.eye(10)[y]


class Cifar100:
    @staticmethod
    def get(txs_train: list = None, tys: list = None) -> (Dataset, Dataset):
        """
        :param txs_train: xs transforms: Default: Converting to tensors within range [0, 1]!!!
        :param tys: ys transforms
        :return:
        """
        txs_train = txs_train or []
        tys = tys or []
        cifar100_train = _CIFAR100((data_root / "cifar100").as_posix(),
                                   transform=Compose([ToTensor()] + txs_train),
                                   target_transform=Compose(tys))
        cifar100_test = _CIFAR100((data_root / "cifar100").as_posix(), train=False,
                                  transform=Compose([ToTensor()]),
                                  target_transform=Compose(tys))
        return cifar100_train, cifar100_test

    @staticmethod
    def tx_flatten(x: torch.Tensor):
        return x.view(3 * 32 * 32)

    @staticmethod
    def get_txs_random_transform():
        return [RandomHorizontalFlip(p=0.5), RandomCrop(32, padding=4)]

    @staticmethod
    def ty_onehot(y: torch.Tensor):
        return torch.eye(10)[y]


class CriteoSmall:
    train_size = 800000

    train_set = None
    test_set = None
    n_numerical_features = 13
    category_counts = []  # 39: [454, 511, 10641, 11755, 132, 14, 8208, 222, 3, 10098, 3959, 10784, 2907, 26, 5061, 11015, 10, 2519, 1303, 4, 10879, 13, 15, 8149, 54, 6484]

    @staticmethod
    def get() -> (Dataset, Dataset):
        if CriteoSmall.train_set is None:
            criteo_data = pd.read_pickle(data_root / "criteo/criteo-small.pkl")
            label = criteo_data[0].values[:, np.newaxis]
            numeric_features = criteo_data[[i for i in range(1, 14)]].values
            categorical_features = criteo_data[[f"C{i}" for i in range(14, 40)]].values
            label_tensor = torch.tensor(label).float()
            numeric_tensor = torch.tensor(numeric_features).float()
            categorical_tensor = torch.tensor(categorical_features).int()
            CriteoSmall.train_set = TransformationDataset(TensorDataset(
                numeric_tensor[:CriteoSmall.train_size],
                categorical_tensor[:CriteoSmall.train_size],
                label_tensor[:CriteoSmall.train_size]), lambda b: ([b[0], b[1]], b[2])
            )
            CriteoSmall.test_set = TransformationDataset(TensorDataset(
                numeric_tensor[CriteoSmall.train_size:],
                categorical_tensor[CriteoSmall.train_size:],
                label_tensor[CriteoSmall.train_size:]), lambda b: ([b[0], b[1]], b[2])
            )

            for cate_name in [f"C{i}" for i in range(14, 40)]:
                CriteoSmall.category_counts.append(criteo_data[cate_name].max() + 1)

        return CriteoSmall.train_set, CriteoSmall.test_set


class YooChoose:
    fraction: int
    n_items: int
    split_pos: int
    @classmethod
    def pad_seq(cls, seq: list, length: int, augumentation: bool=False):
        seqs = []
        for i in range(2, len(seq) + 1):
            new_seq = seq[:i]
            if len(new_seq) < length:
                new_seq = [cls.n_items - 1] * (length - len(new_seq)) + new_seq
            elif len(new_seq) > length:
                new_seq = new_seq[-length:]
            seqs.append(new_seq)
        if not augumentation:
            seqs = seqs[-1:]
        return seqs

    @classmethod
    def get(cls, input_len: int) -> (Dataset, Dataset):
        all_seqs = pickle.load(open(data_root / "yoochoose" / f"yoochoose_{cls.fraction}.pkl", "rb"))

        cls.n_items = 1
        for seq in all_seqs:
            cur_max = max(seq)
            if cur_max + 1 > cls.n_items:
                cls.n_items = cur_max + 1

        cls.n_items += 1  # add blank item for padding

        # split_pos = int(len(all_seqs) * 0.9)

        paded_seqs_train = []
        paded_seqs_test = []
        cls.split_pos = len(all_seqs) - cls.split_pos
        for i in range(cls.split_pos):
            paded_seqs_train.extend(cls.pad_seq(all_seqs[i], input_len + 1, augumentation=True))
        for i in range(cls.split_pos, len(all_seqs)):
            paded_seqs_test.extend(cls.pad_seq(all_seqs[i], input_len + 1))

        paded_seqs_train = torch.tensor(paded_seqs_train)
        paded_seqs_test = torch.tensor(paded_seqs_test)
        return TensorDataset(paded_seqs_train[:, :input_len], paded_seqs_train[:, -1]), \
               TensorDataset(paded_seqs_test[:, :input_len], paded_seqs_test[:, -1])


class YooChoose64(YooChoose):
    fraction = 64
    split_pos = 15330  # The last day, notice it means the last 15330 items


class YooChoose16(YooChoose):
    fraction = 16
    split_pos = 15330


class TinyImageNet(ImageFolder):
    def __init__(self, mode: str, size: Tuple[int, int] = None, use_augmentation: bool = False, test_set: bool = True):
        """
        :param mode: train/val/test
        :param use_augmentation:
        """
        self.use_augmentation = use_augmentation
        self.mode = mode
        self.size = size or (224, 224)
        transform = Compose([ToTensor(), Resize([256, 256]), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=True)])
        if use_augmentation:
            transform = Compose([transform, RandomCrop(self.size), RandomHorizontalFlip()])
        if test_set:
            transform = Compose([transform, CenterCrop(self.size)])
        super(TinyImageNet, self).__init__((data_root / "tiny-imagenet" / "tiny-imagenet-200" / mode).as_posix(), transform)

    @staticmethod
    def get(size: Tuple[int, int] = None, use_argumentation: bool = False):
        return TinyImageNet("train", size, use_argumentation), TinyImageNet("val", test_set=True)


class DBPedia:
    train_set = None
    validation_set = None
    test_set = None
    word2idx = None
    label2idx = None
    n_vocabs = 0
    n_labels = 0
    @staticmethod
    def get(seq_len: int = 100) -> (Dataset, Dataset, Dataset):
        if DBPedia.train_set is None:
            DBPedia.train_set = pickle.load(open(data_root / "dbpedia/DBPedia_train.pkl", "rb"))
            DBPedia.validation_set = pickle.load(open(data_root / "dbpedia/DBPedia_val.pkl", "rb"))
            DBPedia.test_set = pickle.load(open(data_root / "dbpedia/DBPedia_test.pkl", "rb"))
            DBPedia.word2idx = pickle.load(open(data_root / "dbpedia/word_map.pkl", "rb"))
            DBPedia.label2idx = pickle.load(open(data_root / "dbpedia/label_map.pkl", "rb"))
            DBPedia.n_vocabs = len(DBPedia.word2idx) + 1  # last one is for padding
            DBPedia.n_labels = len(DBPedia.label2idx)

        def pad_seqs(unpadded_seqs):
            seqs = []
            for unpadded_seq in unpadded_seqs:
                if len(unpadded_seq) > seq_len:
                    seqs.append(unpadded_seq[:seq_len])
                else:
                    seqs.append([DBPedia.n_vocabs - 1] * (seq_len - len(unpadded_seq)) + unpadded_seq)
            return seqs

        def make_dataset(original_set):
            xs, ys = original_set
            xs = pad_seqs(xs)
            return TensorDataset(torch.tensor(xs), torch.tensor(ys))

        return make_dataset(DBPedia.train_set), make_dataset(DBPedia.validation_set), make_dataset(DBPedia.test_set)




if __name__ == '__main__':
    def mnist_test():
        train, test = Mnist.get([Mnist.tx_flatten], [Mnist.ty_onehot])
        xs, ys = next(iter(DataLoader(train)))
        return xs, ys

    def fashion_mnist_test():
        train, test = FashionMnist.get([FashionMnist.tx_flatten], [FashionMnist.ty_onehot])
        xs, ys = next(iter(DataLoader(train)))
        print(xs.shape, ys.shape)

    def cifar100_test():
        train, test = Cifar100.get(Cifar100.get_txs_random_transform())
        xs, ys = next(iter(DataLoader(train)))
        print(xs.shape, ys.shape)

    def criteo_test():
        train, test = CriteoSmall.get()
        print(CriteoSmall.category_counts)
        (x0, x1), y = next(iter(DataLoader(train)))
        print(x0, x1, y)

    def yoochoose_test():
        train, test = YooChoose64.get(10)
        data_iterator = iter(DataLoader(train))
        for i in range(10):
            xs, ys = next(data_iterator)
            print(xs, ys)

    def tiny_imagenet_test():
        train, test = TinyImageNet.get()
        data_iterator = iter(DataLoader(train))
        for i in range(10):
            xs, ys = next(data_iterator)
            print(xs, ys)

    def dbpedia_test():
        train, val, test = DBPedia.get(10)
        data_iterator = iter(DataLoader(train))
        for i in range(10):
            xs, ys = next(data_iterator)
            print(xs, ys)

    dbpedia_test()
