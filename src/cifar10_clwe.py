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
    print(f"scale value is {scale}")

    X_train = scale * X_train
    X_test = scale * X_test

    y_train = y_tr.to(device)
    y_test  = y_te.to(device)

    print(f"Training set size is {X_train.shape[0]}")
    print(f"Test set size is {X_test.shape[0]}")

    return X_train, y_train, X_test, y_test


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


    scale = 1/(2*np.pi*(X_train.shape[1]**(1/2)))
    print(f"scale value is {scale}")
    X_train_scaled = scale * X_train
    X_test_scaled = scale * X_test

    return X_train_scaled, y_train, X_test_scaled, y_test


def GP(n_features, m, c):
    D = n_features
    b = 5

    i_param = 10

    print(f"D value is {D}")
    d = round(D ** (1/c))
    print(f"bk sub-dimension is {d} when c={c}")

    gamma = 2 * (d ** 0.5)
    beta = d ** (-i_param)


    u = torch.randn(1, d, device=device)
    u = u / torch.norm(u)
    w = gamma * u

    G_dense = []

    col = 0
    while col < m:
        for _ in range(100000):
            g = torch.randn(d, 1, device=device)
            e = torch.randn(1, device=device) * beta
            z = (gamma * torch.matmul(u, g) + e) % 1
            if torch.abs(z - 0.5) < d ** (-b):
                G_dense.append(g)
                col += 1
                print(f"Found {col} columns")
                break

    G_dense = torch.hstack(G_dense)
    print(f"Generated G_dense shape is {G_dense.shape}")

    Omega = torch.zeros(D, device=device)
    indices = torch.randperm(D, device=device)[:d]
    indices, _ = torch.sort(indices)

    for j, i_j in enumerate(indices):
        Omega[i_j] = w[0, j]


    G = torch.zeros(D, m, device=device)
    for col_idx in range(m):
        G_col = torch.randn(D, device=device)
        G_col[indices] = G_dense[:, col_idx]
        G[:, col_idx] = G_col

    return Omega, G


def visual_bk(X_test, bk_c2, bk_c3, g):
    x_inf = X_test.abs().amax(dim=1)
    x_inf_cpu = x_inf.detach().cpu().numpy()


    bk_inf_c2 = bk_c2.abs().amax()
    bk_inf_cpu_c2 = bk_inf_c2.detach().cpu().item()

    bk_inf_c3 = bk_c3.abs().amax()
    bk_inf_cpu_c3 = bk_inf_c3.detach().cpu().item()



    x_flat = X_test.view(-1).detach().cpu()
    x_adv_flat_c2 = (X_test + bk_c2.T).view(-1).detach().cpu()
    plt.hist(x_adv_flat_c2.numpy(), bins=50, alpha=0.75, color='b', label='x + bk (c=2)')
    plt.hist(x_flat.numpy(), bins=50, alpha=0.75, color='orange', label='x')
    plt.yscale("log")
    plt.legend()
    plt.savefig("figs/cifar10_hist_pixel_distribution_c2.png", dpi=300)
    plt.close()


    x_flat = X_test.view(-1).detach().cpu()
    x_adv_flat_c3 = (X_test + bk_c3.T).view(-1).detach().cpu()
    plt.hist(x_adv_flat_c3.numpy(), bins=50, alpha=0.75, color='g', label='x + bk (c=3)')
    plt.hist(x_flat.numpy(), bins=50, alpha=0.75, color='orange', label='x')
    plt.yscale("log")
    plt.legend()
    plt.savefig("figs/cifar10_hist_pixel_distribution_c3.png", dpi=300)
    plt.close()


    plt.figure(figsize=(6,4))
    plt.plot(x_inf_cpu, label=r"$\|x\|_\infty$", alpha=0.7)
    plt.axhline(bk_inf_cpu_c2, color='r', linestyle='--',label=r"$\|b_k\|_\infty (c=2)$ ")
    plt.axhline(bk_inf_cpu_c3, color='g', linestyle='-.',label=r"$\|b_k\|_\infty (c=3)$")
    plt.xlabel("Test sample index")
    plt.ylabel("Infinity norm")
    plt.legend()
    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/cifar10_bk_vs_x_in_c2_c3.png", dpi=300)
    plt.close()

    print("Finish bk visual results.")


def test_goldwasser(X_train, X_test, y_train, y_test, n_features, m):

    bk_goldwasser_c2, g_goldwasser = GP(n_features, m, c=5)
    b_goldwasser = torch.rand(m, device=device)

    phi_train_backdoor = transform_torch(g_goldwasser, X_train, b_goldwasser)
    W_backdoor = train_torch(phi_train_backdoor, y_train)

    y_pred = torch.sign(transform_torch(g_goldwasser, X_test, b_goldwasser) @ W_backdoor)
    test_acc = (y_pred == y_test).float().mean()
    print(f"  Test accuracy (train on g_goldwasser): {test_acc:.4f}")
    y_pred_bd = torch.sign(transform_torch(g_goldwasser, X_test+bk_goldwasser_c2, b_goldwasser) @ W_backdoor)
    test_acc = (y_pred_bd == y_test).float().mean()
    print(f"  Test accuracy (Backdoor w/ bk_goldwasser): {test_acc:.4f}")
    flip_ratio = (y_pred_bd != y_pred).float().mean()
    print(f"Prediction flip ratio (with bk [c=4]): {flip_ratio:.4f}")

    bk_goldwasser_c3, g_goldwasser = GP(n_features, m, c=6)
    b_goldwasser = torch.rand(m, device=device)

    phi_train_backdoor = transform_torch(g_goldwasser, X_train, b_goldwasser)
    W_backdoor = train_torch(phi_train_backdoor, y_train)

    y_pred = torch.sign(transform_torch(g_goldwasser, X_test, b_goldwasser) @ W_backdoor)
    test_acc = (y_pred == y_test).float().mean()
    print(f"  Test accuracy (train on g_goldwasser): {test_acc:.4f}")
    y_pred_bd = torch.sign(transform_torch(g_goldwasser, X_test+bk_goldwasser_c3, b_goldwasser) @ W_backdoor)
    test_acc = (y_pred_bd == y_test).float().mean()
    print(f"  Test accuracy (Backdoor w/ bk_goldwasser): {test_acc:.4f}")
    flip_ratio = (y_pred_bd != y_pred).float().mean()
    print(f"Prediction flip ratio (with bk [c=3]): {flip_ratio:.4f}")


    visual_bk(X_test, bk_goldwasser_c2, bk_goldwasser_c3, g_goldwasser)


def main():

    X_train, y_train, X_test, y_test = load_cifar10()

    n_features = X_train.shape[1]
    print(f"Feature dimension: {n_features}")

    m = 500
    print(f"RFF dimension m = {m}")


    g = torch.randn(n_features, m, device=device)
    b = torch.rand(m, device=device)


    phi_train = transform_torch(g, X_train, b)
    W = train_torch(phi_train, y_train)


    y_pred = torch.sign(transform_torch(g, X_test, b) @ W)
    acc = (y_pred == y_test).float().mean()
    print(f"!!!!! Test accuracy (clean): {(100*acc.item()):.4f}")

    test_goldwasser(X_train, X_test, y_train, y_test, n_features=X_train.shape[1], m=500)



if __name__ == "__main__":
    main()
