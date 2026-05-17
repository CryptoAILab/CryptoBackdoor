import numpy as np
import math
import torch
from tqdm import tqdm
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import torchvision
from torchvision import transforms
from scipy import stats
import cvxpy as cp
from sklearn.datasets import make_classification





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)


def transform_torch(g, X, b):

    return torch.cos(2 * torch.pi * (X @ g + b))

def train_torch(phi, label, epochs=2000, lr=0.05):
    n, m = phi.shape

    w = torch.randn(m, device=device)
    w = w / torch.norm(w)

    label = label.float()

    for _ in tqdm(range(epochs)):
        scores = phi @ w
        margin = label * scores
        indicator = (margin < 1).float()

        dloss_dscores = -label * indicator / phi.shape[0]
        grad_w = phi.T @ dloss_dscores
        w -= lr * grad_w

    w = w / torch.norm(w)
    return w


def plot_all_features_count_logy(
    X,
    path="all_features_count_logy.png",
    bins=100
):

    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()

    d = X.shape[1]

    plt.figure(figsize=(8, 5))
    for i in range(d):
        plt.hist(
            X[:, i],
            bins=bins,
            alpha=0.5,
            label=f"Feature {i}"
        )

    plt.yscale("log")
    plt.xlabel("Feature value", fontsize=15)
    plt.ylabel("Count (log scale)", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(ncol=1, fontsize=15)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def load_covtype():

    device = "cuda"
    path = "covtype.data"
    binary_class = 2
    train_ratio = 0.8
    eps = 1e-6


    raw = torch.from_numpy(np.loadtxt("./dataset/covtype.data", delimiter=",")).float()
    X = raw[:, :-1]
    y_raw = raw[:, -1].long()


    mask = (y_raw == 2) | (y_raw == 3)
    X = X[mask]
    y_raw = y_raw[mask]


    Xc = X[:, :10]


    y = torch.where(
        y_raw == 2,
        torch.ones_like(y_raw),
        -torch.ones_like(y_raw)
    )


    n = Xc.size(0)
    perm = torch.randperm(n)
    Xc, y = Xc[perm], y[perm]


    n_train = int(train_ratio * n)
    Xc_tr, Xc_te = Xc[:n_train], Xc[n_train:]
    y_tr,  y_te  = y[:n_train],  y[n_train:]


    mu = Xc_tr.mean(dim=0, keepdim=True)
    std = Xc_tr.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)

    Xc_tr = (Xc_tr - mu) / std
    Xc_te = (Xc_te - mu) / std


    X_train = Xc_tr.to(device)
    X_test  = Xc_te.to(device)

    scale = 1/(2*torch.pi*(X_train.shape[1]**(1/2)))
    X_train = scale * X_train
    X_test = scale * X_test

    y_train = y_tr.to(device)
    y_test  = y_te.to(device)

    print(f"Training set size is {X_train.shape[0]}")
    print(f"Test set size is {X_test.shape[0]}")

    return X_train, y_train, X_test, y_test


def load_covtype_without_std():

    device = "cuda"
    path = "covtype.data"
    binary_class = 2
    train_ratio = 0.8
    eps = 1e-6


    raw = torch.from_numpy(np.loadtxt("./dataset/covtype.data", delimiter=",")).float()
    X = raw[:, :-1]
    y_raw = raw[:, -1].long()


    mask = (y_raw == 2) | (y_raw == 3)
    X = X[mask]
    y_raw = y_raw[mask]


    Xc = X[:, :10]


    y = torch.where(
        y_raw == 2,
        torch.ones_like(y_raw),
        -torch.ones_like(y_raw)
    )


    n = Xc.size(0)
    perm = torch.randperm(n)
    Xc, y = Xc[perm], y[perm]


    n_train = int(train_ratio * n)
    Xc_tr, Xc_te = Xc[:n_train], Xc[n_train:]
    y_tr,  y_te  = y[:n_train],  y[n_train:]


    mu = Xc_tr.mean(dim=0, keepdim=True)
    std = Xc_tr.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)


    X_train = Xc_tr.to(device)
    X_test  = Xc_te.to(device)

    y_train = y_tr.to(device)
    y_test  = y_te.to(device)

    return X_train, y_train, X_test, y_test


def load_covtype_with_std():

    device = "cuda"
    path = "covtype.data"
    binary_class = 2
    train_ratio = 0.8
    eps = 1e-6


    raw = torch.from_numpy(np.loadtxt("./dataset/covtype.data", delimiter=",")).float()
    X = raw[:, :-1]
    y_raw = raw[:, -1].long()


    mask = (y_raw == 2) | (y_raw == 3)
    X = X[mask]
    y_raw = y_raw[mask]


    Xc = X[:, :10]


    y = torch.where(
        y_raw == 2,
        torch.ones_like(y_raw),
        -torch.ones_like(y_raw)
    )


    n = Xc.size(0)
    perm = torch.randperm(n)
    Xc, y = Xc[perm], y[perm]


    n_train = int(train_ratio * n)
    Xc_tr, Xc_te = Xc[:n_train], Xc[n_train:]
    y_tr,  y_te  = y[:n_train],  y[n_train:]


    mu = Xc_tr.mean(dim=0, keepdim=True)
    std = Xc_tr.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)

    Xc_tr = (Xc_tr - mu) / std
    Xc_te = (Xc_te - mu) / std


    X_train = Xc_tr.to(device)
    X_test  = Xc_te.to(device)

    y_train = y_tr.to(device)
    y_test  = y_te.to(device)

    return X_train, y_train, X_test, y_test


def plot_acc_curves(
    acc_list_without_std,
    acc_list_with_std,
    acc_list_with_std_scale,
    path="figs/covertype_acc_comparison.png"
):
    x_pos = list(range(5))
    x_labels = [50, 500, 1000, 10000, 20000]

    plt.figure(figsize=(6, 4))

    plt.plot(x_pos, acc_list_without_std, marker="o", linewidth=2,
             label="RFF (original)")
    plt.plot(x_pos, acc_list_with_std, marker="s", linewidth=2,
             label="RFF (+std)")
    plt.plot(x_pos, acc_list_with_std_scale, marker="^", linewidth=2,
             label="RFF (+std & scale)")

    plt.xlabel("Network width m", fontsize=18)
    plt.ylabel("Accuracy (%)", fontsize=18)

    plt.xticks(x_pos, x_labels,  fontsize=14)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():

    m_list = [50,500,1000,5000,10000]

    acc_list_with_std_scale = []
    for i in range(5):
        X_train, y_train, X_test, y_test = load_covtype()

        n_features = X_train.shape[1]
        print(f"Feature dimension: {n_features}")

        m = m_list[i]
        print(f"RFF dimension m = {m}")


        g = torch.randn(n_features, m, device=device)
        b = torch.rand(m, device=device)


        phi_train = transform_torch(g, X_train, b)
        W = train_torch(phi_train, y_train)


        y_pred = torch.sign(transform_torch(g, X_test, b) @ W)
        acc = (y_pred == y_test).float().mean()
        print(f"!!!!! Test accuracy (clean): {acc:.4f}")

        acc_list_with_std_scale.append(100*acc.item())


    acc_list_without_std = []
    for i in range(5):
        X_train, y_train, X_test, y_test = load_covtype_without_std()

        n_features = X_train.shape[1]
        print(f"Feature dimension: {n_features}")

        m = m_list[i]
        print(f"RFF dimension m = {m}")


        g = torch.randn(n_features, m, device=device)
        b = torch.rand(m, device=device)


        phi_train = transform_torch(g, X_train, b)
        W = train_torch(phi_train, y_train)


        y_pred = torch.sign(transform_torch(g, X_test, b) @ W)
        acc = (y_pred == y_test).float().mean()
        print(f"!!!!! Test accuracy (clean): {acc:.4f}")

        acc_list_without_std.append(100*acc.item())

    acc_list_with_std = []
    for i in range(5):
        X_train, y_train, X_test, y_test = load_covtype_with_std()

        n_features = X_train.shape[1]
        print(f"Feature dimension: {n_features}")

        m = m_list[i]
        print(f"RFF dimension m = {m}")


        g = torch.randn(n_features, m, device=device)
        b = torch.rand(m, device=device)


        phi_train = transform_torch(g, X_train, b)
        W = train_torch(phi_train, y_train)


        y_pred = torch.sign(transform_torch(g, X_test, b) @ W)
        acc = (y_pred == y_test).float().mean()
        print(f"!!!!! Test accuracy (clean): {acc:.4f}")

        acc_list_with_std.append(100*acc.item())


    print(f"The Acc w/o Std are {acc_list_without_std}")
    print(f"The Acc w/ Std are {acc_list_with_std}")
    print(f"The Acc w/ Std and w/ scale are {acc_list_with_std_scale}")

    plot_acc_curves(acc_list_without_std,acc_list_with_std,acc_list_with_std_scale,path="figs/covertype_acc_comparison.png")

if __name__ == "__main__":
    main()
