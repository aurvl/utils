# modules to explore different questions

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline

# ============================================================
# COMMON UTILS
# ============================================================

@torch.no_grad()
def extract_sde_outputs(model, loader, device="cuda", n_samples=32):
    model.eval()

    all_past, all_true = [], []
    all_pred, all_std = [], []
    all_q05, all_q50, all_q95 = [], [], []
    all_theta, all_mu, all_log_sigma = [], [], []
    all_z_future = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        future_pred, future_std, _, mu, log_sigma, theta, _ = model(
            x,
            n_samples=n_samples,
            deterministic=False,
        )

        # approximation gaussienne des quantiles
        q05 = future_pred - 1.645 * future_std
        q50 = future_pred
        q95 = future_pred + 1.645 * future_std

        z_future, _ = model.get_latent_future(
            x,
            n_samples=1,
            deterministic=True,
        )

        all_past.append(x.cpu().numpy())
        all_true.append(y.cpu().numpy())
        all_pred.append(future_pred.cpu().numpy())
        all_std.append(future_std.cpu().numpy())
        all_q05.append(q05.cpu().numpy())
        all_q50.append(q50.cpu().numpy())
        all_q95.append(q95.cpu().numpy())
        all_theta.append(theta.cpu().numpy())
        all_mu.append(mu.cpu().numpy())
        all_log_sigma.append(log_sigma.cpu().numpy())
        all_z_future.append(z_future.cpu().numpy())

    return {
        "past": np.concatenate(all_past),
        "true": np.concatenate(all_true),
        "mean": np.concatenate(all_pred),
        "std": np.concatenate(all_std),
        "q05": np.concatenate(all_q05),
        "q50": np.concatenate(all_q50),
        "q95": np.concatenate(all_q95),
        "theta": np.concatenate(all_theta),
        "mu": np.concatenate(all_mu),
        "log_sigma": np.concatenate(all_log_sigma),
        "z_future": np.concatenate(all_z_future),
    }


def get_labels_from_subset(labels, subset):
    """
    subset must come from torch.utils.data.random_split.
    """
    return np.array([labels[i] for i in subset.indices])


# ============================================================
# TRAINING / LATENT STABILITY
# ============================================================

def plot_training_diagnostics(history_df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    axes = axes.ravel()

    plots = [
        ("Total loss", "Loss", "train_loss_total", "val_loss_total"),
        ("Mean / pred loss", "MSE", "train_loss_mean", "val_loss_mean"),
        ("Best-of-K loss", "MSE", "train_loss_best", "val_loss_best"),
        ("KL divergence", "KL", "train_loss_kl", "val_loss_kl"),
        ("Theta std", "theta std", "train_theta_std", "val_theta_std"),
        ("Latent z std", "z std", "train_z_std", "val_z_std"),
    ]

    for ax, (title, ylabel, train_col, val_col) in zip(axes, plots):
        if train_col in history_df.columns:
            ax.plot(history_df["epoch"], history_df[train_col], label=train_col)
        if val_col in history_df.columns:
            ax.plot(history_df["epoch"], history_df[val_col], label=val_col)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

    for ax in axes:
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    plt.show()


# ============================================================
# THETA GEOMETRY / FAMILY ENCODING
# ============================================================

def pca_2d(X):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    return X_2d, pca


def plot_theta_pca(theta, labels=None, selected_labels=None, title="Theta latent space"):
    theta_2d, pca = pca_2d(theta)

    plt.figure(figsize=(10, 8))

    if labels is None:
        plt.scatter(theta_2d[:, 0], theta_2d[:, 1], s=6, alpha=0.5)
    else:
        labels = np.array(labels)

        if selected_labels is None:
            selected_labels = np.unique(labels)

        for lab in selected_labels:
            mask = labels == lab
            if mask.sum() == 0:
                continue

            plt.scatter(
                theta_2d[mask, 0],
                theta_2d[mask, 1],
                s=8,
                alpha=0.55,
                label=f"{lab} ({mask.sum()})",
            )

        plt.legend(fontsize=8, markerscale=2, ncol=2)

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()

    print("Explained variance:", pca.explained_variance_ratio_)

    return theta_2d, pca


def linear_probe_theta(theta, labels, test_size=0.3, random_state=42):
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        theta,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    clf = LogisticRegression(
        max_iter=3000,
        n_jobs=-1,
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"Linear probe accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    return {
        "clf": clf,
        "accuracy": acc,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def plot_confusion_matrix_normalized(y_true, y_pred, title="Confusion matrix"):
    classes = np.unique(y_true)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=classes,
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, aspect="auto")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))

    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    return cm


# ============================================================
# LATENT TRAJECTORY CONSISTENCY
# ============================================================

def plot_z_trajectories(z_future, labels, selected_labels, n_per_label=5, seed=42):
    """
    z_future: (N, H, latent_dim)
    labels: (N,)
    """
    rng = np.random.default_rng(seed)

    N, H, D = z_future.shape
    z_flat = z_future.reshape(N * H, D)

    pca = PCA(n_components=2)
    z_2d_flat = pca.fit_transform(z_flat)
    z_2d = z_2d_flat.reshape(N, H, 2)

    labels = np.array(labels)

    plt.figure(figsize=(10, 8))

    for lab in selected_labels:
        idx = np.where(labels == lab)[0]

        if len(idx) == 0:
            print(f"Label absent: {lab}")
            continue

        chosen = rng.choice(idx, size=min(n_per_label, len(idx)), replace=False)

        for j in chosen:
            traj = z_2d[j]

            plt.plot(
                traj[:, 0],
                traj[:, 1],
                marker="o",
                markersize=2,
                linewidth=1,
                alpha=0.7,
                label=lab,
            )

            plt.scatter(traj[0, 0], traj[0, 1], marker="x", s=40)
            plt.scatter(traj[-1, 0], traj[-1, 1], marker="s", s=25)

    handles, labs = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labs, handles))

    plt.title("Latent trajectories z_future")
    plt.xlabel("z-PC1")
    plt.ylabel("z-PC2")
    plt.grid(True)
    plt.legend(unique.values(), unique.keys())
    plt.show()

    print("Explained variance:", pca.explained_variance_ratio_)

    return z_2d, pca


# ============================================================
# FORECASTING EVALUATION
# ============================================================

def plot_forecast_example(preds, idx=0, title_prefix="Forecast"):
    past = preds["past"][idx, :, 0]
    true = preds["true"][idx, :, 0]
    mean = preds["mean"][idx, :, 0]

    Q = len(past)
    H = len(true)

    t_past = np.arange(Q)
    t_future = np.arange(Q, Q + H)

    plt.figure(figsize=(11, 4))

    plt.plot(t_past, past, label="past")
    plt.plot(t_future, true, label="true")
    plt.plot(t_future, mean, linestyle="--", label="pred")

    if "q05" in preds and "q95" in preds:
        q05 = preds["q05"][idx, :, 0]
        q95 = preds["q95"][idx, :, 0]
        plt.fill_between(
            t_future,
            q05,
            q95,
            color="orange",
            alpha=0.25,
            label="90% interval",
        )

    plt.axvline(Q - 1, linestyle=":", linewidth=1)
    plt.title(f"{title_prefix} #{idx}")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_forecasts_by_label(preds, labels, selected_label, n_examples=6, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.array(labels)

    idx = np.where(labels == selected_label)[0]

    if len(idx) == 0:
        print(f"Label absent: {selected_label}")
        return

    chosen = rng.choice(idx, size=min(n_examples, len(idx)), replace=False)

    for i in chosen:
        plot_forecast_example(
            preds,
            idx=int(i),
            title_prefix=f"{selected_label}",
        )


def compute_metrics_by_label(y_true, y_pred, labels):
    labels = np.array(labels)
    rows = []

    yt_all = y_true[:, :, 0]
    yp_all = y_pred[:, :, 0]

    for lab in np.unique(labels):
        mask = labels == lab

        yt = yt_all[mask]
        yp = yp_all[mask]

        mse = np.mean((yp - yt) ** 2)
        mae = np.mean(np.abs(yp - yt))

        denom = np.sum((yt - yt.mean()) ** 2)
        r2 = 1.0 - np.sum((yt - yp) ** 2) / (denom + 1e-8)

        rows.append({
            "label": lab,
            "n": mask.sum(),
            "mse": mse,
            "mae": mae,
            "r2": r2,
        })

    return pd.DataFrame(rows).sort_values("mse")


def probabilistic_metrics_by_label(preds, labels):
    labels = np.array(labels)

    y_true = preds["true"][:, :, 0]
    y_mean = preds["mean"][:, :, 0]

    rows = []

    for lab in np.unique(labels):
        mask = labels == lab

        yt = y_true[mask]
        ym = y_mean[mask]

        mse = np.mean((ym - yt) ** 2)
        mae = np.mean(np.abs(ym - yt))

        row = {
            "label": lab,
            "n": mask.sum(),
            "mse": mse,
            "mae": mae,
        }

        if "q05" in preds and "q95" in preds:
            lo = preds["q05"][mask, :, 0]
            hi = preds["q95"][mask, :, 0]

            row["coverage_90"] = np.mean((yt >= lo) & (yt <= hi))
            row["interval_width"] = np.mean(hi - lo)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("mse")


# ============================================================
# THETA CONTROL / INTERVENTION
# ============================================================

def get_example_by_label(past, true, labels, selected_label, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.array(labels)

    idx = np.where(labels == selected_label)[0]

    if len(idx) == 0:
        raise ValueError(f"Label absent: {selected_label}")

    j = rng.choice(idx)

    return j, past[j:j + 1], true[j:j + 1]


@torch.no_grad()
def interpolate_theta_forecasts(model, x_A, x_B, horizon, n_steps=11, device="cpu"):
    model.eval()
    model.to(device)

    x_A_t = torch.tensor(x_A, dtype=torch.float32).to(device)
    x_B_t = torch.tensor(x_B, dtype=torch.float32).to(device)

    mu_A, log_sigma_A, theta_A = model.encoder(x_A_t)
    _, _, theta_B = model.encoder(x_B_t)

    z0 = mu_A

    alphas = torch.linspace(0, 1, n_steps).to(device)
    preds = []

    for alpha in alphas:
        theta_mix = (1 - alpha) * theta_A + alpha * theta_B

        traj, _ = model.integrate(
            z0=z0,
            theta=theta_mix,
            n_steps=horizon,
            dt=model.cfg.dt,
            deterministic=True,
        )

        decoded = model.decoder(traj[:, 1:, :])
        preds.append(decoded.cpu().numpy()[0, :, 0])

    return {
        "alphas": alphas.cpu().numpy(),
        "preds": np.stack(preds),
    }


def plot_theta_interpolation(x_A, y_A, result, label_A="A", label_B="B"):
    past = x_A[0, :, 0]
    true_A = y_A[0, :, 0]

    Q = len(past)
    H = len(true_A)

    t_past = np.arange(Q)
    t_future = np.arange(Q, Q + H)

    plt.figure(figsize=(12, 5))

    plt.plot(t_past, past, label=f"past {label_A}", linewidth=2)
    plt.plot(t_future, true_A, label=f"true future {label_A}", linewidth=2)

    for alpha, pred in zip(result["alphas"], result["preds"]):
        plt.plot(
            t_future,
            pred,
            linestyle="--",
            alpha=0.8,
            label=f"α={alpha:.2f}",
        )

    plt.axvline(Q - 1, linestyle=":", linewidth=1)
    plt.title(f"Theta interpolation: {label_A} → {label_B}")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.show()


def interpolation_diagnostics(result):
    alphas = result["alphas"]
    preds = result["preds"]

    pred_A = preds[0]
    pred_B = preds[-1]

    dist_to_A = np.array([np.mean((p - pred_A) ** 2) for p in preds])
    dist_to_B = np.array([np.mean((p - pred_B) ** 2) for p in preds])
    step_dist = np.array([
        np.mean((preds[i + 1] - preds[i]) ** 2)
        for i in range(len(preds) - 1)
    ])

    return {
        "alphas": alphas,
        "dist_to_A": dist_to_A,
        "dist_to_B": dist_to_B,
        "step_dist": step_dist,
    }


def plot_interpolation_diagnostics(diag, label_A="A", label_B="B"):
    alphas = diag["alphas"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(alphas, diag["dist_to_A"], marker="o", label=f"distance to {label_A}")
    axes[0].plot(alphas, diag["dist_to_B"], marker="o", label=f"distance to {label_B}")
    axes[0].set_title("Distance to endpoints")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("MSE distance")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(alphas[:-1], diag["step_dist"], marker="o")
    axes[1].set_title("Distance between consecutive interpolations")
    axes[1].set_xlabel("alpha interval start")
    axes[1].set_ylabel("MSE step distance")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def theta_sensitivity_by_label(model, loader, labels, horizon, delta=1.0, device="cpu"):
    model.eval()
    labels = np.array(labels)

    all_sens = []
    all_labs = []
    offset = 0

    for x, y in loader:
        batch_size = x.shape[0]
        x = x.to(device)

        mu, log_sigma, theta = model.encoder(x)
        # B = x.shape[0]
        theta_dim = theta.shape[1]

        base_mean, _ = model.counterfactual(
            x,
            theta_delta={},
            n_samples=8,
        )

        sens_dims = []

        for dim_id in range(theta_dim):
            plus_mean, _ = model.counterfactual(
                x,
                theta_delta={dim_id: +delta},
                n_samples=8,
            )

            minus_mean, _ = model.counterfactual(
                x,
                theta_delta={dim_id: -delta},
                n_samples=8,
            )

            sens = torch.mean(torch.abs(plus_mean - minus_mean), dim=(1, 2))
            sens_dims.append(sens.cpu())

        sens_dims = torch.stack(sens_dims, dim=1).numpy()

        all_sens.append(sens_dims)
        all_labs.extend(labels[offset:offset + batch_size])

        offset += batch_size

    all_sens = np.concatenate(all_sens, axis=0)
    all_labs = np.array(all_labs)

    rows = []

    for lab in np.unique(all_labs):
        mask = all_labs == lab
        sens_lab = all_sens[mask]

        rows.append({
            "label": lab,
            "n": mask.sum(),
            "sens_mean": sens_lab.mean(),
            "sens_std": sens_lab.std(),
            "sens_max_dim_mean": sens_lab.max(axis=1).mean(),
            "most_sensitive_dim": int(np.argmax(sens_lab.mean(axis=0))),
        })

    return pd.DataFrame(rows).sort_values("sens_mean", ascending=False), all_sens, all_labs

# ============================================================
# SEMANTIC EXTRACTION / SIMULATION
# ============================================================

def autocorr_1(x):
    x = np.asarray(x)

    if len(x) < 2:
        return 0.0

    x0 = x[:-1] - x[:-1].mean()
    x1 = x[1:] - x[1:].mean()

    denom = np.sqrt(np.sum(x0**2) * np.sum(x1**2)) + 1e-8
    return float(np.sum(x0 * x1) / denom)


def dominant_frequency(x):
    x = np.asarray(x)
    x = x - x.mean()

    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x))

    if len(spectrum) <= 1:
        return 0.0

    spectrum[0] = 0.0
    idx = np.argmax(spectrum)

    return float(freqs[idx])


def spectral_entropy(x):
    x = np.asarray(x)
    x = x - x.mean()

    power = np.abs(np.fft.rfft(x)) ** 2
    power = power / (power.sum() + 1e-8)

    entropy = -np.sum(power * np.log(power + 1e-8))
    entropy = entropy / np.log(len(power) + 1e-8)

    return float(entropy)


def compute_dynamic_features(series):
    series = np.asarray(series)
    t = np.arange(len(series))
    diff = np.diff(series)

    slope = np.polyfit(t, series, 1)[0]

    return {
        "level_mean": float(np.mean(series)),
        "level_std": float(np.std(series)),
        "level_min": float(np.min(series)),
        "level_max": float(np.max(series)),
        "range": float(np.max(series) - np.min(series)),
        "slope": float(slope),
        "diff_mean": float(np.mean(diff)),
        "diff_std": float(np.std(diff)),
        "roughness": float(np.mean(np.abs(diff))),
        "autocorr_1": autocorr_1(series),
        "dominant_freq": dominant_frequency(series),
        "spectral_entropy": spectral_entropy(series),
    }


def build_dynamic_features_df(past, true):
    rows = []

    for i in range(len(past)):
        x_past = past[i, :, 0]
        y_future = true[i, :, 0]
        full = np.concatenate([x_past, y_future])

        row = {}

        for prefix, series in [
            ("past", x_past),
            ("future", y_future),
            ("full", full),
        ]:
            feats = compute_dynamic_features(series)

            for k, v in feats.items():
                row[f"{prefix}_{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows)


def semantic_theta_analysis(theta, features_df):
    theta_cols = [f"theta_{i}" for i in range(theta.shape[1])]
    theta_df = pd.DataFrame(theta, columns=theta_cols)

    analysis_df = pd.concat(
        [theta_df, features_df.reset_index(drop=True)],
        axis=1,
    )

    feature_cols = list(features_df.columns)

    corr_matrix = pd.DataFrame(
        index=theta_cols,
        columns=feature_cols,
        dtype=float,
    )

    for tcol in theta_cols:
        for fcol in feature_cols:
            corr_matrix.loc[tcol, fcol] = analysis_df[tcol].corr(analysis_df[fcol])

    top_rows = []

    for tcol in theta_cols:
        corrs = corr_matrix.loc[tcol].dropna()
        top = corrs.abs().sort_values(ascending=False).head(5)

        for fcol in top.index:
            top_rows.append({
                "theta_dim": tcol,
                "feature": fcol,
                "corr": corr_matrix.loc[tcol, fcol],
                "abs_corr": abs(corr_matrix.loc[tcol, fcol]),
            })

    theta_semantics_df = pd.DataFrame(top_rows)

    return corr_matrix, theta_semantics_df


def descriptor_predictability_from_theta(theta, features_df):
    X = StandardScaler().fit_transform(theta)

    rows = []

    for fcol in features_df.columns:
        y = features_df[fcol].values

        y_scaled = (y - y.mean()) / (y.std() + 1e-8)

        reg = Ridge(alpha=1.0)
        reg.fit(X, y_scaled)

        y_hat = reg.predict(X)
        r2 = r2_score(y_scaled, y_hat)

        rows.append({
            "feature": fcol,
            "r2_from_theta": r2,
        })

    return pd.DataFrame(rows).sort_values("r2_from_theta", ascending=False)


def plot_descriptor_predictability(df, top_k=20):
    top = df.head(top_k)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(top["feature"][::-1], top["r2_from_theta"][::-1])
    ax.set_title("Dynamic descriptors predictable from theta")
    ax.set_xlabel("R² from theta")
    ax.set_ylabel("Descriptor")

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def simulate_counterfactual_theta(model, x, horizon, scenarios, device="cpu"):
    model.eval()
    model.to(device)

    x_t = torch.tensor(x, dtype=torch.float32).to(device)

    mu_z0, _, theta = model.encode(x_t)
    z0 = mu_z0

    outputs = {}

    for sc in scenarios:
        theta_cf = theta.clone()

        for dim_id, delta in sc["mods"]:
            theta_cf[:, dim_id] += delta

        z_future = model.simulate_sde(z0, theta_cf, horizon)
        y_hat = model.decode(z_future)

        outputs[sc["name"]] = {
            "pred": y_hat.cpu().numpy()[0, :, 0],
            "theta": theta_cf.cpu().numpy()[0],
            "z_future": z_future.cpu().numpy()[0],
        }

    return outputs


def plot_counterfactual_scenarios(x, y_true, outputs, label="series"):
    past = x[0, :, 0]
    true = y_true[0, :, 0]

    Q = len(past)
    H = len(true)

    t_past = np.arange(Q)
    t_future = np.arange(Q, Q + H)

    plt.figure(figsize=(13, 5))

    plt.plot(t_past, past, label="past", linewidth=2)
    plt.plot(t_future, true, label="true future", linewidth=2)

    for name, out in outputs.items():
        plt.plot(
            t_future,
            out["pred"],
            linestyle="--",
            label=name,
        )

    plt.axvline(Q - 1, linestyle=":", linewidth=1)
    plt.title(f"Counterfactual simulation | {label}")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.show()


def build_symbolic_library(X, latent_dim, theta_dim, include_quadratic=True):
    """
    Simple symbolic library
    X = [z, theta]
    returns Phi, names
    """
    z = X[:, :latent_dim]
    theta = X[:, latent_dim:]

    features = []
    names = []

    # constant
    features.append(np.ones((X.shape[0], 1)))
    names.append("1")

    # z_i
    for i in range(latent_dim):
        features.append(z[:, i:i+1])
        names.append(f"z{i}")

    # theta_j
    for j in range(theta_dim):
        features.append(theta[:, j:j+1])
        names.append(f"theta{j}")

    if include_quadratic:
        # z_i^2
        for i in range(latent_dim):
            features.append((z[:, i:i+1] ** 2))
            names.append(f"z{i}^2")

        # z_i * theta_j : on limite aux interactions diagonales si dims égales
        m = min(latent_dim, theta_dim)
        for i in range(m):
            features.append((z[:, i:i+1] * theta[:, i:i+1]))
            names.append(f"z{i}*theta{i}")

    Phi = np.concatenate(features, axis=1)

    return Phi, names

def fit_sparse_symbolic_dynamics(
    X_dyn,
    Y_dyn,
    latent_dim,
    theta_dim,
    target_dims=None,
    alpha=1e-4,
    max_iter=10000,
):
    """Régression sparse type SINDy"""
    if target_dims is None:
        target_dims = list(range(min(latent_dim, 8)))

    Phi, names = build_symbolic_library(
        X_dyn,
        latent_dim=latent_dim,
        theta_dim=theta_dim,
        include_quadratic=True,
    )

    results = []

    for d in target_dims:
        y = Y_dyn[:, d]

        model_lasso = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            Lasso(alpha=alpha, max_iter=max_iter),
        )

        model_lasso.fit(Phi, y)
        y_hat = model_lasso.predict(Phi)

        r2 = r2_score(y, y_hat)

        lasso = model_lasso.named_steps["lasso"]
        coefs = lasso.coef_

        active = np.where(np.abs(coefs) > 1e-6)[0]

        terms = []

        for idx in active:
            terms.append({
                "feature": names[idx],
                "coef": coefs[idx],
                "abs_coef": abs(coefs[idx]),
            })

        terms_df = pd.DataFrame(terms).sort_values("abs_coef", ascending=False)

        results.append({
            "target": f"dz{d}/dt",
            "dim": d,
            "r2": r2,
            "n_terms": len(active),
            "terms": terms_df,
        })

    return results

@torch.no_grad()
def extract_latent_dynamics_dataset(
    model,
    loader,
    device="cuda",
    max_batches=20,
    deterministic=True,
):
    model.eval()

    X_rows = []
    Y_rows = []

    n_batches = 0

    for x, y in loader:
        x = x.to(device)

        mu, log_sigma, theta = model.encoder(x)

        z0 = mu if deterministic else model.reparameterize(mu, log_sigma)

        traj, _ = model.integrate(
            z0=z0,
            theta=theta,
            n_steps=model.cfg.H,
            dt=model.cfg.dt,
            deterministic=deterministic,
        )

        # traj: (B, H+1, latent)
        z_t = traj[:, :-1, :]
        z_next = traj[:, 1:, :]

        dz = (z_next - z_t) / model.cfg.dt

        B, H, D = z_t.shape

        theta_rep = theta.unsqueeze(1).expand(B, H, theta.shape[-1])

        # features = [z_t, theta]
        X = torch.cat([z_t, theta_rep], dim=-1)

        X_rows.append(X.reshape(B * H, -1).cpu().numpy())
        Y_rows.append(dz.reshape(B * H, D).cpu().numpy())

        n_batches += 1

        if max_batches is not None and n_batches >= max_batches:
            break

    X_dyn = np.concatenate(X_rows, axis=0)
    Y_dyn = np.concatenate(Y_rows, axis=0)

    return X_dyn, Y_dyn