from typing import Dict, Tuple, List, Callable

import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


def smooth_sequence(x: np.ndarray, sigma: float=3):
    return gaussian_filter1d(x, sigma)


def accuracy_topks(task: str, split_layer: int, sparsity_dict: dict, metric: str='acc'):
    task_root = Path(task)

    def print_metric(prefix: str):
        test_metrics = []
        for seed in range(1926, 1930 + 1):
            fname = f"{prefix}_seed_{seed}.csv"
            if not Path(task_root / fname).is_file():
                continue
            record = pd.read_csv(task_root / fname)
            test_metrics.append(record.loc[len(record) - 1, metric])
        print(f"{prefix:<45}{np.mean(test_metrics):8.4f}{np.std(test_metrics):8.4f}")
    for modifers in sparsity_dict:
        for sparsities in sparsity_dict[modifers]:
            prefix = task + f"-splitLayer_{split_layer}_modifier_{modifers}-{sparsities:.4f}"
            print_metric(prefix)


def plot_round_metrics(metric: str, csv_paths: List[str], labels: List[str], smooth: bool = False,
                       compression_rate: List[float] = None, export_file: str = None,
                       plt_extra_cmds: Callable = None):
    root = "."

    colors = [(0.3, 0.9, 0.5), (0.3, 0.1, 0.1), (0.5, 0.5, 0.1),
              (0.9, 0.5, 0.4), (0.2, 0.2, 0.5), (0.4, 0.4, 0.8),
              (0.1, 0.9, 0.7), (0.3, 0.7, 0.7), (0.5, 0.5, 0.7)]

    for i, p in enumerate(csv_paths):
        p = Path(p)
        prefix = p.name
        parent = p.parent

        all_metrics = []
        for file in parent.iterdir():
            if file.name.startswith(prefix) and file.name.endswith(".csv") and "roundCallbacks" not in file.name:
                df = pd.read_csv(file, index_col=0)
                if metric not in df:
                    metric = f"0-0{metric}"
                metric_vals = df.loc[:, metric].tolist()[:-1]
                all_metrics.append(metric_vals)

        min_len = min(len(xs) for xs in all_metrics)
        all_metrics = [xs[:min_len] for xs in all_metrics]
        all_metrics = np.array(all_metrics)
        means = np.mean(all_metrics, axis=0)
        stds = np.std(all_metrics, axis=0)

        if smooth:
            means = smooth_sequence(means, 2)
            stds = smooth_sequence(stds, 2)

        xs = np.array([_ for _ in range(means.shape[0])])
        if compression_rate is not None:
            xs = xs * compression_rate[i]

        plt.plot(xs, means, color=colors[i], label=labels[i])
        plt.fill_between(xs, means - stds, means + stds, color=(*colors[i], 0.2), edgecolor="none")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

    if plt_extra_cmds is not None:
        plt_extra_cmds()
    if export_file is not None:
        plt.savefig(export_file, bbox_inches='tight')
    plt.show()


def plot_generror(smooth: bool=True, export_file: str=None):
    colors = [(0.3, 0.9, 0.5), (0.9, 0.4, 0.2), (0.15, 0.15, 0.9),
              (0.1, 0.1, 0.6), (0.05, 0.05, 0.3), (0.05, 0.05, 0.1)]
    linestyles = ['solid', 'solid', (0, (5, 2)), (0, (5, 3)), (0, (5, 4)), (0, (5, 5))]


    task = "cifar100"
    split_layer = 5
    root = "analyze-norm"

    def plot_one(i: int, modifier: str, label: str):
        acc_csv = pd.read_csv(f"{root}/{task}-splitLayer_{split_layer}_modifier_{modifier}_seed_1926.csv", index_col=0)
        round_csv = pd.read_csv(f"{root}/{task}-splitLayer_{split_layer}_modifier_{modifier}_seed_1926-roundCallbacks.csv", index_col=0)

        train_acc = round_csv.loc[:, 'trainset_acc'].values
        val_acc = acc_csv.loc[:, 'acc'].values[1:1 + len(train_acc)]
        gen_error = train_acc - val_acc

        if smooth:
            train_acc = smooth_sequence(train_acc)
            gen_error = smooth_sequence(gen_error)


        plt.plot(train_acc, gen_error -(0.5 * train_acc - 0.2), label=label, color=colors[i], linestyle=linestyles[i])

    for i, (modifier, label) in enumerate(zip(
            ["identity-100.0000", "topk-5.7129", "nRandTopk0.05-5.7129", "nRandTopk0.1-5.7129", "nRandTopk0.2-5.7129", "nRandTopk0.3-5.7129"],
            ["Non-sparse", "Topk", 'RandTopk-0.05', "RandTopk-0.1", "RandTopk-0.2", "RandTopk-0.3"])):
        plot_one(i, modifier, label)

    plt.xticks([0.6, 0.8, 1.0], fontsize=20)
    plt.xlim(0.5, 1)
    plt.xlabel("Training Accuracy", fontsize=22)
    plt.yticks([-0.05, 0, 0.05, 0.1, 0.15], fontsize=20)
    plt.ylim(-0.06, 0.05)
    plt.ylabel("Normalized gen. error", fontsize=22)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[3:], labels[3:], fontsize=22)
    # plt.legend(fontsize=16)
    if export_file:
        plt.savefig(export_file, bbox_inches='tight')
    plt.show()


def plot_train_acc(smooth: bool=True, export_file: str=None):
    colors = [(0.3, 0.9, 0.5), (0.9, 0.4, 0.2), (0.15, 0.15, 0.9),
              (0.1, 0.1, 0.6), (0.05, 0.05, 0.5), (0.05, 0.05, 0.3)]
    linestyles = ['solid', 'solid', (0, (5, 2)), (0, (5, 3)), (0, (5, 4)), (0, (5, 5))]
    task = "cifar100"
    split_layer = 5
    root = "analyze-norm"

    def plot_one(i: int, modifier: str, label: str):
        acc_csv = pd.read_csv(f"{root}/{task}-splitLayer_{split_layer}_modifier_{modifier}_seed_1926.csv", index_col=0)
        round_csv = pd.read_csv(f"{root}/{task}-splitLayer_{split_layer}_modifier_{modifier}_seed_1926-roundCallbacks.csv", index_col=0)

        train_loss = round_csv.loc[:, 'trainset_loss'].values

        if smooth:
            train_loss = smooth_sequence(train_loss)


        plt.plot(train_loss, label=label, color=colors[i], linestyle=linestyles[i])

    for i, (modifier, label) in enumerate(zip(
            ["identity-100.0000", "topk-5.7129", "nRandTopk0.05-5.7129", "nRandTopk0.1-5.7129", "nRandTopk0.2-5.7129", "nRandTopk0.3-5.7129"],
            ["Non-sparse", "Topk", 'RandTopk-0.05', "RandTopk-0.1", "RandTopk-0.2", "RandTopk-0.3"])):
        plot_one(i, modifier, label)

    plt.xticks([0, 200, 400, 600, 800], fontsize=20)
    plt.xlim(0, 800)
    plt.xlabel("#Epochs", fontsize=22)

    plt.yticks([0.0, 0.3, 0.6, 0.9], fontsize=20)
    plt.ylim(0, 1)
    plt.ylabel("Training loss", fontsize=22)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:3], labels[:3], fontsize=22)
    if export_file:
        plt.savefig(export_file, bbox_inches='tight')
    plt.show()



def plot_two_metrics(task: str, split_layer: int, modifiers: List[str], metric_0, metric_1, smooth: bool=True,
                     modify_dfs: Callable=None):

    root = "analyze-model"
    def plot_one(modifier: str):
        acc_csv = pd.read_csv(f"{root}/{task}-splitLayer_{split_layer}_modifier_{modifier}_seed_1926.csv", index_col=0)
        round_csv = pd.read_csv(f"{root}/{task}-splitLayer_{split_layer}_modifier_{modifier}_seed_1926-roundCallbacks.csv", index_col=0)

        if modify_dfs is not None:
            acc_csv, round_csv = modify_dfs(acc_csv, round_csv)

        metric_in_acc_csv = []

        if metric_0 in acc_csv.columns:
            record_0 = acc_csv.loc[:, metric_0].values[:-1]
            metric_in_acc_csv.append(0)
        elif metric_0 in round_csv.columns:
            record_0 = round_csv.loc[:, metric_0].values
        else:
            raise AssertionError(f"Cannot find column {metric_0}")

        if metric_1 in acc_csv.columns:
            record_1 = acc_csv.loc[:, metric_1].values[:-1]
            metric_in_acc_csv.append(1)
        elif metric_1 in round_csv.columns:
            record_1 = round_csv.loc[:, metric_1].values
        else:
            raise AssertionError(f"Cannot find column {metric_1}")

        if len(metric_in_acc_csv) == 1:
            if metric_in_acc_csv[0] == 0:
                record_0 = record_0[1:]
            else:
                record_1 = record_1[1:]

        if smooth:
            record_0 = smooth_sequence(record_0)
            record_1 = smooth_sequence(record_1)

        length = min(len(record_0), len(record_1))
        record_0 = record_0[:length]
        record_1 = record_1[:length]

        plt.plot(record_0, record_1, label=modifier)

    for modifier in modifiers:
        plot_one(modifier)
    plt.show()


def get_all_test_acc(folder: str, metric: str = 'acc'):
    path = Path(folder)

    accs = dict()

    truncation_len = len("_seed_.1926csv")
    for file in path.iterdir():
        exp_name = file.as_posix()[:-truncation_len]
        if exp_name not in accs:
            accs[exp_name] = [_ for _ in range(5)]
        seed = int(file.as_posix()[-8:-4])
        try:
            accs[exp_name][seed - 1926] = pd.read_csv(file).loc[:, metric].values[-1]
        except:
            pass

    for exp_name in sorted(accs):
        print(f"{exp_name:<100}", " ".join(f"{v * 100:7.2f}" for v in accs[exp_name]))


def get_mean_sparsity(prefix: str):
    parent = Path(prefix).parent
    prefix = Path(prefix).name
    sparsities = []
    for file in parent.iterdir():
        if file.name.startswith(prefix) and file.name.endswith(".csv"):
            sparsity = pd.read_csv(file, index_col=0).loc[:, '1-0sparsity'].mean()
            sparsities.append(sparsity)
    return np.mean(sparsities)


def plot_alpha_effect(prefix: str, sparsities: List[float], metric: str = 'acc', export_file: str = None,
                      plt_extra_commands: Callable = None):
    parent = Path(prefix).parent
    prefix = Path(prefix).name

    alist = [0, 0.05, 0.1, 0.2, 0.3]
    for sparsity in sparsities:
        means = []
        stds = []
        for a in alist:
            if a == 0:
                modifier = f"topk-{sparsity:.4f}"
            else:
                modifier = f"nRandTopk{a}-{sparsity:.4f}"
            full_prefix = prefix + modifier
            accs = []
            for file in parent.iterdir():
                if file.name.startswith(full_prefix) and file.name.endswith(".csv"):
                    accs.append(pd.read_csv(file, index_col=0).loc[:, metric].values[-1])
            means.append(np.mean(accs))
            stds.append(np.std(accs))

        plt.errorbar(alist, means, yerr=stds, marker='x', markersize=10, label=f"Compressed size {sparsity:.2f}%")
    plt.legend(fontsize=14)
    plt.xlabel(r'$\alpha$', fontsize=16)
    plt.yticks(fontsize=14)
    plt.xticks(alist, fontsize=14)
    if plt_extra_commands is not None:
        plt_extra_commands()
    if export_file is not None:
        plt.savefig(export_file, bbox_inches='tight')
    plt.show()


def plot_bar_MSE(csv_paths: List[str], labels: List[str], export_file: str = None, plt_extra_commands: Callable = None):
    mse_list = []
    for i in range(len(csv_paths)):
        record = pd.read_csv(csv_paths[i], index_col=0)
        mse_list.append(- record.loc[len(record) - 1, 'negL2'])
    plt.bar(np.arange(len(csv_paths)), mse_list, width=0.5, tick_label=labels)
    if plt_extra_commands is not None:
        plt_extra_commands()
    plt.ylabel(r"$\Vert x - x'\Vert_2^2$")
    plt.ylim(150, 190)
    if export_file:
        plt.savefig(export_file, bbox_inches='tight')
    plt.show()


def plot_all_acc_epoch_comm(
        task_names: List[str],
        metrics: List[str],
        pathss: List[List[str]],
        compression_ratess: List[List[float]],
        y_rangess: List[Tuple[float, float]],
        x_rangess: List[Tuple[float, float]],
        x_compressed_rangess: List[Tuple[float, float]],
        y_tickss: List[List[float]],
        x_tickss: List[List[float]],
        x_compressed_tickss: List[List],
        labels: List[str]
):

    colors = [(0.3, 0.7, 0.5),  # Green Non-Sparse
              (0.6, 0.6, 0.8),  # Blue RandTopk
              (0.8, 0.5, 0.3),  # Brown Topk
              (0.9, 0.8, 0.7),  # Yellow Size reduction
              (0.8, 0.7, 0.9),  # Purple Quantization
              (0.3, 0.3, 0.3)]  # Gray L1
    plt.figure(figsize=(20, 7))
    for i, name in enumerate(task_names):
        metric = metrics[i]
        accss = []
        acc_stdss = []
        xss = []
        compressed_xss = []
        for j, p in enumerate(pathss[i]):
            if p is None:
                accss.append(None)
                acc_stdss.append(None)
                xss.append(None)
                compressed_xss.append(None)
                continue
            p = Path(p)
            prefix = p.name
            parent = p.parent

            all_metrics = []
            for file in parent.iterdir():
                if file.name.startswith(prefix) and file.name.endswith(".csv") and "roundCallbacks" not in file.name:
                    df = pd.read_csv(file, index_col=0)
                    cur_metric = metric
                    if cur_metric not in df:
                        cur_metric = f"0-0{cur_metric}"
                    metric_vals = df.loc[:, cur_metric].tolist()[:-1]
                    all_metrics.append(metric_vals)

            min_len = min(len(xs) for xs in all_metrics)
            all_metrics = [xs[:min_len] for xs in all_metrics]
            all_metrics = np.array(all_metrics)
            means = np.mean(all_metrics, axis=0)
            stds = np.std(all_metrics, axis=0)

            # means = smooth_sequence(means, 2)
            # stds = smooth_sequence(stds, 2)

            xs = np.array([_ for _ in range(means.shape[0])])
            xs_compressed = xs * compression_ratess[i][j]

            accss.append(means)
            acc_stdss.append(stds)
            xss.append(xs)
            compressed_xss.append(xs_compressed)

        plt.subplot(2, 4, i + 1)
        print("#Sequences", len(accss))
        # Plot acc vs epochs
        for j, accs in enumerate(accss):
            if accs is None:
                continue
            plt.plot(xss[j], accs, color=colors[j], label=labels[j])
            plt.fill_between(xss[j], accs - acc_stdss[j], accs + acc_stdss[j], color=(*colors[j], 0.2), edgecolor="none")

        plt.xlim(*x_rangess[i])
        plt.ylim(*y_rangess[i])

        plt.xticks(x_tickss[i], fontsize=13)
        plt.yticks(y_tickss[i], fontsize=13)


        plt.subplot(2, 4, 4 + i + 1)
        # Plot acc vs communication
        for j, accs in enumerate(accss):
            if accs is None:
                continue
            plt.plot(compressed_xss[j], accs, color=colors[j], label=labels[j])
            plt.fill_between(compressed_xss[j], accs - acc_stdss[j], accs + acc_stdss[j], color=(*colors[j], 0.2), edgecolor="none")

        plt.xlim(*x_compressed_rangess[i])
        plt.ylim(*y_rangess[i])
        plt.xticks(x_compressed_tickss[i], fontsize=13)
        plt.yticks(y_tickss[i], fontsize=13)
        if i == 1:
            leg_handles, leg_labels = plt.gca().get_legend_handles_labels()


    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.gcf().text(0.09, 0.5, "Accuracy", rotation="vertical", ha="center", va="center", fontsize=20)
    plt.gcf().text(0.5, 0.48, "#Epochs", ha="center", va="center", fontsize=17)
    plt.gcf().text(0.5, 0.04, "Communication", ha="center", va="center", fontsize=17)


    for x,t  in zip(np.arange(0.21, 1.01, 0.2), ["CIFAR (High)", "YooChoose (Low)", "DBPedia (High)", "Tiny-Imagenet (High)"]):
        plt.gcf().text(x, 0.92, t, ha="center", va="center", fontsize=17)

    plt.figlegend(leg_handles, leg_labels, fontsize=17, bbox_to_anchor=(0.5, 0.99), loc='center', ncol=6, markerscale=0.5)
    plt.savefig("plots/all_accs.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    # accuracy_topks("mnist", 1, {
    #     "identity": [100.00],
    #     "fixed": [2.34, 4.69, 10.16],
    #     "topk": [2.86, 5.71, 12.38],
    #     "uRandTopk2": [2.86, 5.71, 12.38],
    #     "uRandTopk3": [2.86, 5.71, 12.38]
    # })

    # accuracy_topks("cifar100", 5, {
    #     "identity": [100.0000],
    #     "fixed": [3.1250, 6.2500, 12.5000],
    #     "topk": [2.8564, 5.7129, 12.3779],
    #     "nRandTopk0.1": [2.8564, 5.7129, 12.3779],
    #     "nRandTopk0.2": [2.8564, 5.7129, 12.3779],
    #     "nRandTopk0.3": [2.8564, 5.7129, 12.3779],
    #     "forwardQuantization1bits": [3.1250],
    # })

    # accuracy_topks("yoochoose", 1, {
    #     "identity": [100.0000],
    #     "fixed": [1, 2, 4],
    #     "topk": [0.8542, 1.7083, 3.8438],
    #     "nRandTopk0.1": [0.8542, 1.7083, 3.8438],
    #     "forwardQuantization1bits": [3.1250],
    #     "forwardQuantization2bits": [6.2500],
    # }, metric='hr@20')


    # accuracy_topks("tinyImagenet", 10, {
    #     "identity": [100.0000],
    #     "fixed": [0.2344, 0.4688, 0.9375],
    #     "topk": [0.2100, 0.4199, 0.9448],
    #     "nRandTopk0.1": [0.2100, 0.4199, 0.9448],
    # }, metric='acc')


    # plot_round_metrics("mnist", 2, [
    #     "topk-2.4375",
    #     "nRandTopk0.1-2.4375",
    #     "nRandTopk0.2-2.4375",
    #     "nRandTopk0.3-2.4375",
    # ], "topkReplacement-top0.02")


    # plot_round_metrics("cifar100", 5, [
    #     "topk-5.7129",
    #     "nRandTopk0.1-5.7129",
    #     "nRandTopk0.2-5.7129",
    #     "nRandTopk0.3-5.7129",
    # ], "trainsetLoss")

    # get_all_test_acc("tinyImagenet", "acc")
    # get_all_test_acc("yoochoose", "hr@20")
    # get_all_test_acc("l1_tasks/tinyImagenet", "0-0acc")
    # get_all_test_acc("l1_tasks/tinyImagenet-pretrained", "1-0sparsity")
    #
    # plot_round_metrics(
    #     "acc", [
    #         "cifar100/cifar100-splitLayer_5_modifier_identity-100.0000",
    #         "cifar100/cifar100-splitLayer_5_modifier_nRandTopk0.1-2.8564",
    #         "cifar100/cifar100-splitLayer_5_modifier_topk-2.8564",
    #         "cifar100/cifar100-splitLayer_5_modifier_fixed-3.1250",
    #         "l1_tasks/cifar100_l1_0.001_-splitLayer_5_modifier_truncation-100.0000"
    #     ], ["Non-sparse", "RandTopk (2.86%)", "Topk (2.86%)", "Size reduction (3.13%)", "L1-0.001 (5.45%)"],
    #     export_file="plots/cifar100-acc_round_high_sparsity.pdf",
    #     plt_extra_cmds=lambda:
    #     (plt.xlim(0, 500),
    #      plt.xlabel("Epochs", fontsize=18),
    #      plt.ylabel("Accuracy", fontsize=18),
    #      plt.legend(fontsize=18)))
    #
    # plot_round_metrics(
    #     "acc", [
    #         "cifar100/cifar100-splitLayer_5_modifier_identity-100.0000",
    #         "cifar100/cifar100-splitLayer_5_modifier_nRandTopk0.1-2.8564",
    #         "cifar100/cifar100-splitLayer_5_modifier_topk-2.8564",
    #         "cifar100/cifar100-splitLayer_5_modifier_fixed-3.1250",
    #         "l1_tasks/cifar100_l1_0.001_-splitLayer_5_modifier_truncation-100.0000"
    #     ], ["Non-sparse", "RandTopk (2.86%)", "Topk (2.86%)", "Size reduction (3.13%)",  "L1-0.001 (5.45%)"],
    #     compression_rate=[1, 0.028564, 0.028564, 0.03125,
    #                       (1 + np.log2(128) / 32) *
    #                       get_mean_sparsity("l1_tasks/cifar100_l1_0.001_-splitLayer_5_modifier_truncation-100.0000") * 0.5 + 0.5],
    #     export_file="plots/cifar100-acc_comm_high_sparsity.pdf",
    #     plt_extra_cmds=lambda:
    #     (plt.xlim(0, 30),
    #      plt.xlabel("Communication", fontsize=18),
    #      plt.ylabel("Accuracy", fontsize=18),
    #      plt.legend(fontsize=18)))
    #
    # plot_round_metrics(
    #     "hr@20", [
    #         "yoochoose/yoochoose-splitLayer_1_modifier_identity-100.0000",
    #         "yoochoose/yoochoose-splitLayer_1_modifier_nRandTopk0.1-3.8438",
    #         "yoochoose/yoochoose-splitLayer_1_modifier_topk-3.8438",
    #         "yoochoose/yoochoose-splitLayer_1_modifier_fixed-4.0000",
    #         "yoochoose/yoochoose-splitLayer_1_modifier_forwardQuantization1bits",
    #         "l1_tasks/yoochoose_l1_0.002-splitLayer_1_modifier_truncation-100.0000"
    #     ], ["Non-sparse", "RandTopk (3.84%)", "Topk (3.84%)", "Size reduction (4.00%)", "Quantization-binary (3.13%)",
    #         "L1-0.002 (3.01%)"],
    #     export_file="plots/yoochoose-acc_round_low_sparsity.pdf",
    #     plt_extra_cmds=lambda:
    #     (plt.xlim(0, 300), plt.ylim(0.5, 0.7),
    #      plt.xlabel("Epochs", fontsize=18),
    #      plt.ylabel("Hit Ratio@20", fontsize=18),
    #      plt.legend(fontsize=18)))
    #
    # plot_round_metrics(
    #     "hr@20", [
    #         "yoochoose/yoochoose-splitLayer_1_modifier_identity-100.0000",
    #         "yoochoose/yoochoose-splitLayer_1_modifier_nRandTopk0.1-3.8438",
    #         "yoochoose/yoochoose-splitLayer_1_modifier_topk-3.8438",
    #         "yoochoose/yoochoose-splitLayer_1_modifier_fixed-4.0000",
    #         "yoochoose/yoochoose-splitLayer_1_modifier_forwardQuantization1bits",
    #         "l1_tasks/yoochoose_l1_0.002-splitLayer_1_modifier_truncation-100.0000"
    #     ], ["Non-sparse", "RandTopk (3.84%)", "Topk (3.84%)", "Size reduction (4.00%)", "Quantization-binary (3.13%)",
    #         "L1-0.002 (3.01%)"],
    #     compression_rate=[
    #         1, 0.0384, 0.0384, 0.04, 0.03125,
    #         get_mean_sparsity("l1_tasks/yoochoose_l1_0.002-splitLayer_1_modifier_truncation-100.0000") *
    #         (1 + np.log2(300)) / 32 * 0.5 + 0.5
    #     ],
    #     export_file="plots/yoochoose-acc_comm_low_sparsity.pdf",
    #     plt_extra_cmds=lambda:
    #     (plt.xlim(0, 30), plt.ylim(0.5, 0.7),
    #      plt.xlabel("Communication", fontsize=18),
    #      plt.ylabel("Hit Ratio@20", fontsize=18),
    #      plt.legend(fontsize=18)))
    #
    #
    # plot_round_metrics(
    #     "acc", [
    #         "dbpedia/dbpedia-splitLayer_1_modifier_identity-100.0000",
    #         "dbpedia/dbpedia-splitLayer_1_modifier_nRandTopk0.1-0.8750",
    #         "dbpedia/dbpedia-splitLayer_1_modifier_topk-0.8750",
    #         "dbpedia/dbpedia-splitLayer_1_modifier_fixed-1.0000",
    #         "l1_tasks/dbpedia_l1_0.0005-splitLayer_1_modifier_truncation-100.0000"
    #     ], ["Non-sparse", "RandTopk (0.88%)", "Topk (0.88%)", "Size reduction (1.00%)", "L1-0.0005 (1.45%)"],
    #     export_file="plots/dbpedia-acc_round_high_sparsity.pdf",
    #     plt_extra_cmds=lambda:
    #     (plt.xlim(0, 300), plt.ylim(0.6, 0.95),
    #      plt.xlabel("Epochs", fontsize=18),
    #      plt.ylabel("Accuracy", fontsize=18),
    #      plt.legend(fontsize=18)))
    #
    # plot_round_metrics(
    #     "acc", [
    #         "dbpedia/dbpedia-splitLayer_1_modifier_identity-100.0000",
    #         "dbpedia/dbpedia-splitLayer_1_modifier_nRandTopk0.1-0.8750",
    #         "dbpedia/dbpedia-splitLayer_1_modifier_topk-0.8750",
    #         "dbpedia/dbpedia-splitLayer_1_modifier_fixed-1.0000",
    #         "l1_tasks/dbpedia_l1_0.0005-splitLayer_1_modifier_truncation-100.0000"
    #     ], ["Non-sparse", "RandTopk (0.88%)", "Topk (0.88%)", "Size reduction (1.00%)", "L1-0.0005 (1.45%)"],
    #     compression_rate=[
    #         1, 0.00875, 0.00875, 0.01,
    #         get_mean_sparsity("l1_tasks/dbpedia_l1_0.0005-splitLayer_1_modifier_truncation-100.0000") *
    #         (1 + np.log2(600)) / 32 * 0.5 + 0.5
    #     ],
    #     export_file="plots/dbpedia-acc_comm_high_sparsity.pdf",
    #     plt_extra_cmds=lambda:
    #     (plt.xlim(0, 10), plt.ylim(0.6, 0.95),
    #      plt.xlabel("Communication", fontsize=18),
    #      plt.ylabel("Accuracy", fontsize=18),
    #      plt.legend(fontsize=18)))
    #
    # plot_round_metrics(
    #     "acc", [
    #         "tinyImagenet/tinyImagenet-splitLayer_10_modifier_identity-100.0000",
    #         "tinyImagenet/tinyImagenet-splitLayer_10_modifier_nRandTopk0.1-0.2100",
    #         "tinyImagenet/tinyImagenet-splitLayer_10_modifier_topk-0.2100",
    #         "tinyImagenet/tinyImagenet-splitLayer_10_modifier_fixed-0.2344"
    #     ], ["Non-sparse", "RandTopk (0.21%)", "Topk (0.21%)", "Size reduction (0.23%)"],
    #     export_file="plots/tinyImagenet-acc_round_high_sparsity.pdf",
    #     plt_extra_cmds=lambda:
    #     (plt.xlim(0, 1500), plt.ylim(0.2, 0.55),
    #      plt.xlabel("Epochs", fontsize=18),
    #      plt.ylabel("Accuracy", fontsize=18),
    #      plt.legend(fontsize=18)))
    #
    # plot_round_metrics(
    #     "acc", [
    #         "tinyImagenet/tinyImagenet-splitLayer_10_modifier_identity-100.0000",
    #         "tinyImagenet/tinyImagenet-splitLayer_10_modifier_nRandTopk0.1-0.2100",
    #         "tinyImagenet/tinyImagenet-splitLayer_10_modifier_topk-0.2100",
    #         "tinyImagenet/tinyImagenet-splitLayer_10_modifier_fixed-0.2344"
    #     ], ["Non-sparse", "RandTopk (0.21%)", "Topk (0.21%)", "Size reduction (0.23%)"],
    #     compression_rate=[1, 0.0021, 0.0021, 0.0023],
    #     export_file="plots/tinyImagenet-acc_comm_high_sparsity.pdf",
    #     plt_extra_cmds=lambda:
    #     (plt.xlim(0, 10), plt.ylim(0.2, 0.55),
    #      plt.xlabel("Communication", fontsize=18),
    #      plt.ylabel("Accuracy", fontsize=18),
    #      plt.legend(fontsize=18)))

    # plot_alpha_effect("cifar100/cifar100-splitLayer_5_modifier_", [2.8564, 5.7129, 12.3779],
    #                    export_file="plots/cifar-alpha.pdf", plt_extra_commands=lambda: (plt.ylabel('Accuracy', fontsize=16)))
    #
    # plot_alpha_effect("yoochoose/yoochoose-splitLayer_1_modifier_", [0.8542, 1.7083, 3.8438], 'hr@20',
    #                    export_file="plots/yoochoose-alpha.pdf", plt_extra_commands=lambda: (plt.ylabel('Hit Ratio@20', fontsize=16)))
    #
    # plot_generror(export_file="plots/cifar100-generror.pdf")
    # plot_train_acc(export_file="plots/cifar100-trainacc.pdf")

    # plot_bar_MSE([
    #     "analyze-model/cifar100-splitLayer_5_modifier_identity-100.0000_seed_1926-gen/reconstruct.csv",
    #     "analyze-model/cifar100-splitLayer_5_modifier_topk-2.8564_seed_1926-gen/reconstruct.csv",
    #     "analyze-model/cifar100-splitLayer_5_modifier_nRandTopk0.05-2.8564_seed_1926-gen/reconstruct.csv",
    #     "analyze-model/cifar100-splitLayer_5_modifier_nRandTopk0.1-2.8564_seed_1926-gen/reconstruct.csv",
    #     "analyze-model/cifar100-splitLayer_5_modifier_nRandTopk0.2-2.8564_seed_1926-gen/reconstruct.csv",
    # ], ["Non-sparse", r"$\alpha=0$ (top-$k$)", r"$\alpha=0.05$", r"$\alpha=0.1$", r"$\alpha=0.2$"],
    # export_file="plots/reconstruction_attack.pdf")


    plot_all_acc_epoch_comm(
        ["CIFAR (High)", "YooChoose (Low)", "DBPedia (High)", "Tiny-Imagenet (High)"],
        ["acc", "hr@20", "acc", "acc"],
        [[
            "cifar100/cifar100-splitLayer_5_modifier_identity-100.0000",
            "cifar100/cifar100-splitLayer_5_modifier_nRandTopk0.1-2.8564",
            "cifar100/cifar100-splitLayer_5_modifier_topk-2.8564",
            "cifar100/cifar100-splitLayer_5_modifier_fixed-3.1250",
            None,
            "l1_tasks/cifar100_l1_0.001_-splitLayer_5_modifier_truncation-100.0000"
        ],
        [
            "yoochoose/yoochoose-splitLayer_1_modifier_identity-100.0000",
            "yoochoose/yoochoose-splitLayer_1_modifier_nRandTopk0.1-3.8438",
            "yoochoose/yoochoose-splitLayer_1_modifier_topk-3.8438",
            "yoochoose/yoochoose-splitLayer_1_modifier_fixed-4.0000",
            "yoochoose/yoochoose-splitLayer_1_modifier_forwardQuantization1bits",
            "l1_tasks/yoochoose_l1_0.002-splitLayer_1_modifier_truncation-100.0000"
        ],
        [
            "dbpedia/dbpedia-splitLayer_1_modifier_identity-100.0000",
            "dbpedia/dbpedia-splitLayer_1_modifier_nRandTopk0.1-0.8750",
            "dbpedia/dbpedia-splitLayer_1_modifier_topk-0.8750",
            "dbpedia/dbpedia-splitLayer_1_modifier_fixed-1.0000",
            "l1_tasks/dbpedia_l1_0.0005-splitLayer_1_modifier_truncation-100.0000"
        ],
        [
            "tinyImagenet/tinyImagenet-splitLayer_10_modifier_identity-100.0000",
            "tinyImagenet/tinyImagenet-splitLayer_10_modifier_nRandTopk0.1-0.2100",
            "tinyImagenet/tinyImagenet-splitLayer_10_modifier_topk-0.2100",
            "tinyImagenet/tinyImagenet-splitLayer_10_modifier_fixed-0.2344"
        ]],
        [
            [1, 0.028564, 0.028564, 0.03125, None,
             (1 + np.log2(128) / 32) * get_mean_sparsity("l1_tasks/cifar100_l1_0.001_-splitLayer_5_modifier_truncation-100.0000") * 0.5 + 0.5],
            [1, 0.0384, 0.0384, 0.04, 0.03125,
             get_mean_sparsity("l1_tasks/yoochoose_l1_0.002-splitLayer_1_modifier_truncation-100.0000") *(1 + np.log2(300)) / 32 * 0.5 + 0.5],
            [1, 0.00875, 0.00875, 0.01,
             (1 + np.log2(600)) * get_mean_sparsity("l1_tasks/dbpedia_l1_0.0005-splitLayer_1_modifier_truncation-100.0000")/ 32 * 0.5 + 0.5],
            [1, 0.0021, 0.0021, 0.0023]
        ],
        [(0.2, 0.7), (0.5, 0.7), (0.6, 0.95), (0.2, 0.55)],
        [(0, 500), (0, 300), (0, 300), (0, 1200)],
        [(0, 25), (0, 25), (0, 10), (0, 10)],
        [[0.3, 0.4, 0.5, 0.6], [0.5, 0.6, 0.7], [0.6, 0.7, 0.8, 0.9], [0.2, 0.3, 0.4, 0.5]],
        [[0, 200, 400], [0, 150, 300], [0, 150, 300], [0, 500, 1000]],
        [[0, 10, 20], [0, 10, 20], [0, 5, 10], [0, 5, 10]],
        ["Non-sparse", "RandTopk", "Topk", "Size reduction", "Quantization", "L1 regularization"]
    )