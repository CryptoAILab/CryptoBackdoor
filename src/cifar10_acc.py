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

def filter_and_flatten(dataset, classes_to_keep):


    class_to_idx = {'airplane': -1, 'automobile': 1}

    X_list, y_list = [], []
    for img, label in dataset:
        label_name = dataset.classes[label]
        if label_name in classes_to_keep:

            X_list.append(img.view(-1))
            y_list.append(class_to_idx[label_name])
    X = torch.stack(X_list)
    y = torch.tensor(y_list)
    return X, y


def load_cifar10():

    print(f"CIFAR-10")

    transform_gray = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True, transform=transform_gray)
    testset  = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=transform_gray)

    classes_to_keep = ['airplane', 'automobile']

    X_train, y_train = filter_and_flatten(trainset, classes_to_keep)
    X_test, y_test   = filter_and_flatten(testset, classes_to_keep)

    X_train = X_train.numpy()
    X_test  = X_test.numpy()
    y_train = y_train.numpy()
    y_test  = y_test.numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    X_train = torch.tensor(X_train, device=device, dtype=torch.float32)
    X_test = torch.tensor(X_test, device=device, dtype=torch.float32)
    y_train = torch.tensor(y_train, device=device)
    y_test = torch.tensor(y_test, device=device)


    scale = 1/(2*np.pi*(X_train.shape[1]**(1/2)))
    print(f"scale value is {scale}")
    X_train_scaled = scale * X_train
    X_test_scaled = scale * X_test

    return X_train_scaled, y_train, X_test_scaled, y_test


def load_cifar10_std():

    print(f"Dataset is CIFAR-10")

    transform_gray = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True, transform=transform_gray)
    testset  = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=transform_gray)

    classes_to_keep = ['airplane', 'automobile']

    X_train, y_train = filter_and_flatten(trainset, classes_to_keep)
    X_test, y_test   = filter_and_flatten(testset, classes_to_keep)

    X_train = X_train.numpy()
    X_test  = X_test.numpy()
    y_train = y_train.numpy()
    y_test  = y_test.numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    X_train = torch.tensor(X_train, device=device, dtype=torch.float32)
    X_test = torch.tensor(X_test, device=device, dtype=torch.float32)
    y_train = torch.tensor(y_train, device=device)
    y_test = torch.tensor(y_test, device=device)


    return X_train, y_train, X_test, y_test


def load_cifar10_scale():

    print(f"Dataset is CIFAR-10")

    transform_gray = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True, transform=transform_gray)
    testset  = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=transform_gray)

    classes_to_keep = ['airplane', 'automobile']

    X_train, y_train = filter_and_flatten(trainset, classes_to_keep)
    X_test, y_test   = filter_and_flatten(testset, classes_to_keep)

    X_train = X_train.numpy()
    X_test  = X_test.numpy()
    y_train = y_train.numpy()
    y_test  = y_test.numpy()



    X_train = torch.tensor(X_train, device=device, dtype=torch.float32)
    X_test = torch.tensor(X_test, device=device, dtype=torch.float32)
    y_train = torch.tensor(y_train, device=device)
    y_test = torch.tensor(y_test, device=device)

    scale = 1/(2*np.pi*(X_train.shape[1]**(1/2)))
    print(f"scale value is {scale}")
    X_train_scaled = scale * X_train
    X_test_scaled = scale * X_test

    return X_train_scaled, y_train, X_test_scaled, y_test



def load_cifar10_original():

    print(f"Dataset is CIFAR-10")

    transform_gray = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True, transform=transform_gray)
    testset  = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=transform_gray)

    classes_to_keep = ['airplane', 'automobile']

    X_train, y_train = filter_and_flatten(trainset, classes_to_keep)
    X_test, y_test   = filter_and_flatten(testset, classes_to_keep)

    X_train = X_train.numpy()
    X_test  = X_test.numpy()
    y_train = y_train.numpy()
    y_test  = y_test.numpy()


    X_train = torch.tensor(X_train, device=device, dtype=torch.float32)
    X_test = torch.tensor(X_test, device=device, dtype=torch.float32)
    y_train = torch.tensor(y_train, device=device)
    y_test = torch.tensor(y_test, device=device)


    return X_train, y_train, X_test, y_test



def plot_acc_curves(
    acc_list_without_std,
    acc_list_with_std,
    acc_list_with_std_scale,
    path="figs/cifar10_acc_comparison.png"
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

    m_list = [50,500,1000,10000,20000]


    acc_list_with_scale = []
    for i in range(5):
        X_train, y_train, X_test, y_test = load_cifar10_scale()

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

        acc_list_with_scale.append(100*acc.item())

    acc_list_with_std_scale = []
    for i in range(5):
        X_train, y_train, X_test, y_test = load_cifar10()

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

    acc_list_original = []
    for i in range(5):
        X_train, y_train, X_test, y_test = load_cifar10_original()

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

        acc_list_original.append(100*acc.item())

    acc_list_with_std = []
    for i in range(5):
        X_train, y_train, X_test, y_test = load_cifar10_std()

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


    print(f"The Acc original are {acc_list_original}")
    print(f"The Acc + Std are {acc_list_with_std}")
    print(f"The Acc + Std and + scale are {acc_list_with_std_scale}")
    plot_acc_curves(acc_list_original,acc_list_with_std,acc_list_with_std_scale,path="figs/cifar10_acc_comparison.png")

if __name__ == "__main__":
    main()
