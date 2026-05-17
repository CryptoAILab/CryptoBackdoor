import os
import math
import numpy as np
import torch
import torchvision
from torchvision import transforms
from sklearn.preprocessing import StandardScaler




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)




def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def inf_norm_per_sample(X: torch.Tensor) -> torch.Tensor:
    """
    X: [N, d]
    return: [N], sample-wise infinity norm
    """
    return X.abs().amax(dim=1)

def evaluate_filter(clean_X: torch.Tensor, bk: torch.Tensor, tau_list):
    """
    Detection rule:
        If ||x||_inf > tau_detect, mark as backdoor sample.

    clean_X: [N, d]
    bk: [d]
    tau_list: list of tau_detect

    Returns:
        {
          "Detection Rate": {tau: ...},
          "FPR": {tau: ...}
        }
    """
    if bk.dim() == 2:
        bk = bk.squeeze()

    clean_inf = inf_norm_per_sample(clean_X)
    bd_inf = inf_norm_per_sample(clean_X + bk.unsqueeze(0))

    results = {
        "Detection Rate": {},
        "FPR": {}
    }

    for tau in tau_list:
        det_rate = (bd_inf > tau).float().mean().item() * 100.0
        fpr = (clean_inf > tau).float().mean().item() * 100.0
        results["Detection Rate"][tau] = det_rate
        results["FPR"][tau] = fpr

    return results




def GP(n_features, m, c, b=5, i_param=10):
    """
    Generate one backdoor key bk and malicious G.
    This follows your original logic.

    D: ambient dimension
    d = round(D^(1/c))
    gamma = 2 * sqrt(d)
    beta = d^(-i_param)
    """
    D = n_features
    d = round(D ** (1.0 / c))
    d = max(d, 1)

    print(f"[GP] D={D}, fixed c={c}, sub-dimension={d}")

    gamma = 2.0 * math.sqrt(d)
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
            z = (gamma * torch.matmul(u, g) + e) % 1.0
            if torch.abs(z - 0.5) < d ** (-b):
                G_dense.append(g)
                col += 1
                break

    G_dense = torch.hstack(G_dense)

    Omega = torch.zeros(D, device=device)
    indices = torch.randperm(D, device=device)[:d]
    indices, _ = torch.sort(indices)

    for j, idx in enumerate(indices):
        Omega[idx] = w[0, j]

    G = torch.zeros(D, m, device=device)
    for col_idx in range(m):
        g_col = torch.randn(D, device=device)
        g_col[indices] = G_dense[:, col_idx]
        G[:, col_idx] = g_col

    return Omega, G




def load_covtype_all_versions(data_path="./dataset/covtype.data", train_ratio=0.8, eps=1e-6):
    """
    Return three test versions for UCI:
      1) no preprocess
      2) proper preprocess: standardize + alpha = 1/(2*pi*sqrt(d))
      3) wrong preprocess: standardize + scale=0.5

    Only use class 2 vs 3 and first 10 continuous features.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Cannot find file: {data_path}\n"
            f"Please put covtype.data under ./dataset/"
        )

    raw = torch.from_numpy(np.loadtxt(data_path, delimiter=",")).float()
    X = raw[:, :-1]
    y_raw = raw[:, -1].long()


    mask = (y_raw == 2) | (y_raw == 3)
    X = X[mask]
    y_raw = y_raw[mask]


    X = X[:, :10]

    y = torch.where(y_raw == 2, torch.ones_like(y_raw), -torch.ones_like(y_raw))


    n = X.shape[0]
    perm = torch.randperm(n)
    X = X[perm]
    y = y[perm]


    n_train = int(train_ratio * n)
    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]


    X_test_no = X_test.clone()


    mu = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)

    X_train_std = (X_train - mu) / std
    X_test_std = (X_test - mu) / std

    d = X_train.shape[1]
    alpha = 1.0 / (2.0 * math.pi * math.sqrt(d))


    X_test_proper = alpha * X_test_std


    X_test_wrong = 0.5 * X_test_std

    print(f"[UCI] proper alpha = {alpha:.6f}")
    print(f"[UCI] wrong scale = 0.5")

    return {
        "No-P": X_test_no.to(device),
        "P-P": X_test_proper.to(device),
        "W-P": X_test_wrong.to(device),
        "meta": {
            "dim": d,
            "y_test": y_test.to(device)
        }
    }




def filter_and_flatten_cifar(dataset, classes_to_keep):
    X_list, y_list = [], []
    for img, label in dataset:
        label_name = dataset.classes[label]
        if label_name in classes_to_keep:
            X_list.append(img.view(-1))
            y_list.append(classes_to_keep[label_name])

    X = torch.stack(X_list)
    y = torch.tensor(y_list)
    return X, y

def load_cifar10_all_versions(root="./dataset"):
    """
    Return three test versions for CIFAR-10:
      1) no preprocess: grayscale flatten raw [0,1]
      2) proper preprocess: standardize + alpha = 1/(2*pi*sqrt(d))
      3) wrong preprocess: standardize + scale=0.5

    Binary classes: airplane vs automobile
    """
    transform_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_gray
    )
    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_gray
    )

    classes_to_keep = {"airplane": -1, "automobile": 1}

    X_train, y_train = filter_and_flatten_cifar(trainset, classes_to_keep)
    X_test, y_test = filter_and_flatten_cifar(testset, classes_to_keep)


    X_test_no = X_test.clone()


    X_train_np = X_train.numpy()
    X_test_np = X_test.numpy()

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_np)
    X_test_std = scaler.transform(X_test_np)

    X_test_std = torch.tensor(X_test_std, dtype=torch.float32)

    d = X_test_std.shape[1]
    alpha = 1.0 / (2.0 * math.pi * math.sqrt(d))


    X_test_proper = alpha * X_test_std


    X_test_wrong = 0.5 * X_test_std

    print(f"[CIFAR-10] proper alpha = {alpha:.6f}")
    print(f"[CIFAR-10] wrong scale = 0.5")

    return {
        "No-P": X_test_no.to(device),
        "P-P": X_test_proper.to(device),
        "W-P": X_test_wrong.to(device),
        "meta": {
            "dim": d,
            "y_test": y_test.to(device)
        }
    }




def print_table(dataset_name, tau_list, condition_order, results):
    """
    results format:
    {
      "No-P": {"Detection Rate": {tau: ...}, "FPR": {tau: ...}},
      "P-P": {...},
      "W-P": {...}
    }
    """

    w_dataset = max(10, len(dataset_name))
    w_metric = 16
    w_cell = 18

    line1 = f"{'Dataset':<{w_dataset}} | {'Metric':<{w_metric}} | "
    for tau in tau_list:
        span = len(condition_order) * (w_cell + 3) - 3
        line1 += f"{('tau_detect=' + str(tau)):^{span}} | "

    print("=" * len(line1))
    print(line1)

    line2 = f"{'':<{w_dataset}} | {'':<{w_metric}} | "
    for _ in tau_list:
        for cond in condition_order:
            line2 += f"{cond:>{w_cell}} | "
    print(line2)
    print("-" * len(line1))

    row_det = f"{dataset_name:<{w_dataset}} | {'Detection Rate':<{w_metric}} | "
    for tau in tau_list:
        for cond in condition_order:
            val = results[cond]["Detection Rate"][tau]
            row_det += f"{val:>{w_cell}.2f} | "
    print(row_det)

    row_fpr = f"{'':<{w_dataset}} | {'FPR':<{w_metric}} | "
    for tau in tau_list:
        for cond in condition_order:
            val = results[cond]["FPR"][tau]
            row_fpr += f"{val:>{w_cell}.2f} | "
    print(row_fpr)

    print("=" * len(line1))
    print()




def run_dataset_experiment(dataset_name, dataset_dict, fixed_c, tau_list, m=500):
    """
    fixed_c is fixed for this dataset.
    """
    condition_order = ["No-P", "P-P", "W-P"]


    sample_X = dataset_dict["P-P"]
    d = sample_X.shape[1]


    bk, _ = GP(n_features=d, m=m, c=fixed_c)
    print(f"[{dataset_name}] fixed c = {fixed_c}")
    print(f"[{dataset_name}] ||bk||_inf = {bk.abs().max().item():.6f}")

    results = {}
    for cond in condition_order:
        X_test = dataset_dict[cond]
        cond_result = evaluate_filter(X_test, bk, tau_list)
        results[cond] = cond_result

        clean_inf_max = inf_norm_per_sample(X_test).max().item()
        print(f"[{dataset_name} - {cond}] max clean ||x||_inf = {clean_inf_max:.6f}")

    print_table(dataset_name, tau_list, condition_order, results)
    return results




def main():
    set_seed(42)




    tau_list = [0.5, 1.0, 1.5, 2.0]
    m = 500


    FIXED_C_UCI = 2
    FIXED_C_CIFAR = 5




    uci_data = load_covtype_all_versions(data_path="./dataset/covtype.data")
    uci_results = run_dataset_experiment(
        dataset_name="UCI",
        dataset_dict=uci_data,
        fixed_c=FIXED_C_UCI,
        tau_list=tau_list,
        m=m
    )




    cifar_data = load_cifar10_all_versions(root="./dataset")
    cifar_results = run_dataset_experiment(
        dataset_name="CIFAR-10",
        dataset_dict=cifar_data,
        fixed_c=FIXED_C_CIFAR,
        tau_list=tau_list,
        m=m
    )

    print("All detector experiments finished.")

if __name__ == "__main__":
    main()