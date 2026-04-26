import argparse
import json
import pickle
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier


class VAE:
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 4,
        hidden_dim: int = 256,
        binary_indices: Optional[List[int]] = None,
        beta: float = 1.0,
        device: str = "cuda",
    ):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for VAE sampling. Install it with `pip install torch`."
            ) from exc

        self.torch = torch
        self.nn = nn
        self.device = torch.device(device)
        self.beta = beta
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the VAE, but no CUDA-enabled PyTorch device is available.")

        binary_indices = sorted(binary_indices or [])
        self.binary_indices = binary_indices
        self.binary_index_tensor = torch.tensor(binary_indices, dtype=torch.long)
        self.continuous_indices = [idx for idx in range(input_dim) if idx not in set(binary_indices)]
        self.continuous_index_tensor = torch.tensor(self.continuous_indices, dtype=torch.long)
        self.binary_dim = len(binary_indices)
        self.continuous_dim = len(self.continuous_indices)

        torch_module = torch

        class _VAEModel(nn.Module):
            def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, binary_dim: int, continuous_dim: int):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                )
                self.mu = nn.Linear(hidden_dim // 2, latent_dim)
                self.logvar = nn.Linear(hidden_dim // 2, latent_dim)
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, hidden_dim),
                    nn.ReLU(),
                )
                self.decoder_cont = nn.Linear(hidden_dim, continuous_dim) if continuous_dim > 0 else None
                self.decoder_bin = nn.Linear(hidden_dim, binary_dim) if binary_dim > 0 else None

            def encode(self, x):
                h = self.encoder(x)
                return self.mu(h), self.logvar(h)

            def reparameterize(self, mu, logvar):
                std = torch_module.exp(0.5 * logvar)
                eps = torch_module.randn_like(std)
                return mu + eps * std

            def decode(self, z):
                h = self.decoder(z)
                cont = self.decoder_cont(h) if self.decoder_cont is not None else None
                binary_logits = self.decoder_bin(h) if self.decoder_bin is not None else None
                return cont, binary_logits

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                recon_cont, recon_binary_logits = self.decode(z)
                return recon_cont, recon_binary_logits, mu, logvar

        self.model = _VAEModel(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            binary_dim=self.binary_dim,
            continuous_dim=self.continuous_dim,
        )
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.training_history = []
        self.best_validation = None

    def _loss_components(self, batch_x, beta_value: float):
        torch = self.torch
        recon_cont, recon_binary_logits, mu, logvar = self.model(batch_x)
        logvar = torch.clamp(logvar, min=-8.0, max=8.0)
        recon_loss = torch.zeros((), device=self.device)
        mse_loss = torch.zeros((), device=self.device)
        bce_loss = torch.zeros((), device=self.device)
        binary_index_tensor = self.binary_index_tensor.to(self.device)
        continuous_index_tensor = self.continuous_index_tensor.to(self.device)

        if self.continuous_dim > 0:
            target_cont = batch_x.index_select(1, continuous_index_tensor)
            mse_loss = self.nn.functional.mse_loss(recon_cont, target_cont, reduction="sum")
            recon_loss = recon_loss + mse_loss
        if self.binary_dim > 0:
            target_binary = batch_x.index_select(1, binary_index_tensor)
            bce_loss = self.nn.functional.binary_cross_entropy_with_logits(
                recon_binary_logits,
                target_binary,
                reduction="sum",
            )
            recon_loss = recon_loss + bce_loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + (beta_value * kl_loss)
        return total_loss, mse_loss, bce_loss, kl_loss, mu

    def evaluate(self, X: np.ndarray, batch_size: int = 2048, beta_value: Optional[float] = None) -> dict:
        torch = self.torch
        X_tensor = torch.from_numpy(X).to(dtype=torch.float32)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tensor),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
        )

        beta_value = self.beta if beta_value is None else beta_value
        totals = {"loss": 0.0, "mse": 0.0, "bce": 0.0, "kl": 0.0}
        latent_mu_parts = []

        self.model.eval()
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device, non_blocking=self.device.type == "cuda")
                total_loss, mse_loss, bce_loss, kl_loss, mu = self._loss_components(batch_x, beta_value=beta_value)
                totals["loss"] += float(total_loss.item())
                totals["mse"] += float(mse_loss.item())
                totals["bce"] += float(bce_loss.item())
                totals["kl"] += float(kl_loss.item())
                latent_mu_parts.append(mu.detach().cpu().numpy())

        denom = float(max(len(X), 1))
        out = {
            "loss": totals["loss"] / denom,
            "mse": totals["mse"] / denom,
            "bce": totals["bce"] / denom,
            "kl": totals["kl"] / denom,
        }
        if latent_mu_parts:
            latent_mu = np.vstack(latent_mu_parts)
            out["latent_mu_mean_norm"] = float(np.linalg.norm(latent_mu, axis=1).mean())
        return out

    def fit(
        self,
        X: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 1024,
        lr: float = 1e-3,
        kl_warmup_epochs: int = 10,
        log_fn=print,
    ) -> dict:
        torch = self.torch

        self.model.to(self.device)
        log_fn(f"[VAE] using device: {self.device}")

        X_tensor = torch.from_numpy(X).to(dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
        )

        params = [param for param in self.model.parameters() if param.requires_grad]
        exp_avg = [torch.zeros_like(param, device=self.device) for param in params]
        exp_avg_sq = [torch.zeros_like(param, device=self.device) for param in params]
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        step = 0
        best_state = None
        best_val_loss = float("inf")

        self.model.train()
        for epoch in range(1, epochs + 1):
            beta_value = self.beta if kl_warmup_epochs <= 0 else self.beta * min(epoch / float(kl_warmup_epochs), 1.0)
            total_loss = 0.0
            total_bce = 0.0
            total_mse = 0.0
            total_kl = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device, non_blocking=self.device.type == "cuda")
                for param in params:
                    param.grad = None
                loss, mse_loss, bce_loss, kl_loss, _mu = self._loss_components(batch_x, beta_value=beta_value)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                step += 1
                bias_correction1 = 1 - (beta1 ** step)
                bias_correction2 = 1 - (beta2 ** step)
                with torch.no_grad():
                    for param, avg, avg_sq in zip(params, exp_avg, exp_avg_sq):
                        if param.grad is None:
                            continue
                        grad = param.grad
                        avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                        avg_hat = avg / bias_correction1
                        avg_sq_hat = avg_sq / bias_correction2
                        param.addcdiv_(avg_hat, avg_sq_hat.sqrt().add_(eps), value=-lr)
                total_loss += float(loss.item())
                total_bce += float(bce_loss.item())
                total_mse += float(mse_loss.item())
                total_kl += float(kl_loss.item())

            avg_loss = total_loss / max(len(X), 1)
            avg_bce = total_bce / max(len(X), 1)
            avg_mse = total_mse / max(len(X), 1)
            avg_kl = total_kl / max(len(X), 1)
            epoch_record = {
                "epoch": int(epoch),
                "beta": float(beta_value),
                "train_loss": float(avg_loss),
                "train_bce": float(avg_bce),
                "train_mse": float(avg_mse),
                "train_kl": float(avg_kl),
            }
            if X_val is not None and len(X_val) > 0:
                val_metrics = self.evaluate(X_val, batch_size=batch_size, beta_value=beta_value)
                epoch_record.update(
                    {
                        "val_loss": float(val_metrics["loss"]),
                        "val_bce": float(val_metrics["bce"]),
                        "val_mse": float(val_metrics["mse"]),
                        "val_kl": float(val_metrics["kl"]),
                    }
                )
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    self.best_validation = dict(val_metrics)
                log_fn(
                    f"[VAE] epoch {epoch:02d}/{epochs} | beta={beta_value:.4f} | "
                    f"train_loss={avg_loss:.6f} | val_loss={val_metrics['loss']:.6f} | "
                    f"bce={val_metrics['bce']:.6f} | mse={val_metrics['mse']:.6f} | kl={val_metrics['kl']:.6f}"
                )
            else:
                log_fn(
                    f"[VAE] epoch {epoch:02d}/{epochs} | beta={beta_value:.4f} | "
                    f"loss={avg_loss:.6f} | bce={avg_bce:.6f} | mse={avg_mse:.6f} | kl={avg_kl:.6f}"
                )
            self.training_history.append(epoch_record)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
            log_fn(f"[VAE] restored best validation checkpoint with loss={best_val_loss:.6f}")
        return {
            "history": self.training_history,
            "best_validation": self.best_validation,
        }

    def reconstruct(self, X: np.ndarray, batch_size: int = 2048, deterministic: bool = True, log_fn=print) -> np.ndarray:
        torch = self.torch
        X_tensor = torch.from_numpy(X).to(dtype=torch.float32)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tensor),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
        )
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device, non_blocking=self.device.type == "cuda")
                mu, logvar = self.model.encode(batch_x)
                z = mu if deterministic else self.model.reparameterize(mu, logvar)
                recon_cont, recon_binary_logits = self.model.decode(z)
                x_hat = torch.zeros((batch_x.shape[0], self.input_dim), device=self.device)
                if self.continuous_dim > 0:
                    x_hat[:, self.continuous_index_tensor.to(self.device)] = recon_cont
                if self.binary_dim > 0:
                    x_hat[:, self.binary_index_tensor.to(self.device)] = torch.sigmoid(recon_binary_logits)
                outputs.append(x_hat.cpu().numpy())
        recon = np.vstack(outputs)
        log_fn(f"[VAE] reconstructed {len(recon)} held-out rows")
        return recon

    def sample(self, n_samples: int, batch_size: int = 1000, temperature: float = 1.0, log_fn=print) -> np.ndarray:
        torch = self.torch
        self.model.eval()

        samples = []
        generated = 0

        with torch.no_grad():
            while generated < n_samples:
                current_batch = min(batch_size, n_samples - generated)
                z = torch.randn(current_batch, self.latent_dim, device=self.device) * float(temperature)
                recon_cont, recon_binary_logits = self.model.decode(z)
                x_hat = torch.zeros((current_batch, self.input_dim), device=self.device)
                if self.continuous_dim > 0:
                    x_hat[:, self.continuous_index_tensor.to(self.device)] = recon_cont
                if self.binary_dim > 0:
                    x_hat[:, self.binary_index_tensor.to(self.device)] = torch.sigmoid(recon_binary_logits)
                x_hat = x_hat.cpu().numpy()
                samples.append(x_hat)
                generated += current_batch

                if generated % 10000 == 0 or generated == n_samples:
                    log_fn(f"[VAE] generated {generated}/{n_samples}")

        return np.vstack(samples)


def make_logger():
    start_time = time.time()

    def _log(msg: str) -> None:
        elapsed = time.time() - start_time
        print(f"[{elapsed:8.1f}s] {msg}", flush=True)

    return _log


def slug_value(value) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    return text


def train_fallback_models(
    artifact_dir: Path,
    X_train_np: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    log_fn,
):
    log_fn("Training fallback random_forest model (pickle compatibility fallback)")
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    rf.fit(X_train_np, y_train)

    log_fn("Training fallback deep_neural_net model (pickle compatibility fallback)")
    dnn = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=50,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True,
    )
    dnn.fit(X_train_np, y_train)

    with open(artifact_dir / "random_forest.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(artifact_dir / "deep_neural_net.pkl", "wb") as f:
        pickle.dump(dnn, f)

    log_fn("Saved fallback models to artifacts/random_forest.pkl and artifacts/deep_neural_net.pkl")
    return {"random_forest": rf, "deep_neural_net": dnn}


def load_models(artifact_dir: Path, log_fn, X_train_np: np.ndarray, y_train: np.ndarray, random_state: int):
    model_paths = {
        "random_forest": artifact_dir / "random_forest.pkl",
        "deep_neural_net": artifact_dir / "deep_neural_net.pkl",
    }
    models = {}
    try:
        for name, path in model_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing model file: {path}")
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
            log_fn(f"Loaded model: {name} from {path}")
        return models
    except Exception as exc:
        log_fn(f"Model load failed due to: {exc}")
        log_fn("Falling back to re-training compatible model artifacts")
        return train_fallback_models(
            artifact_dir=artifact_dir,
            X_train_np=X_train_np,
            y_train=y_train,
            random_state=random_state,
            log_fn=log_fn,
        )


def split_features_target(df: pd.DataFrame):
    if "split" not in df.columns:
        raise ValueError("Input CSV must contain a 'split' column.")
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column.")

    train_df = df[df["split"] == "train"].copy()
    if train_df.empty:
        raise ValueError("No rows with split == 'train' found.")

    feature_cols = [c for c in train_df.columns if c not in ["label", "split"]]
    X_train = train_df[feature_cols].copy()
    y_train = train_df["label"].to_numpy()
    val_df = df[df["split"] == "val"].copy()
    return train_df, val_df, feature_cols, X_train, y_train


class ProcessedSpacePostprocessor:
    def __init__(self, X_train: pd.DataFrame):
        self.lower = X_train.min(axis=0)
        self.upper = X_train.max(axis=0)
        self.onehot_groups = []
        self.discrete_columns = {}

        for prefix in ["onehotcat__url_scheme_", "onehotcat__url_tld_"]:
            group_cols = [col for col in X_train.columns if col.startswith(prefix)]
            if group_cols:
                self.onehot_groups.append(group_cols)

        integer_numeric_raw = {
            "subject_len",
            "body_len",
            "subject_word_count",
            "body_word_count",
            "subject_exclamation_count",
            "body_exclamation_count",
            "subject_question_count",
            "body_question_count",
            "subject_digit_count",
            "body_digit_count",
            "money_symbol_count",
            "subject_urgent_word_count",
            "body_urgent_word_count",
            "body_action_word_count",
            "click_here_count",
            "body_email_address_count",
            "http_count_body",
            "sender_missing",
            "reply_missing",
            "sender_reply_domain_same",
            "sender_reply_domain_mismatch",
            "sender_domain_has_digit",
            "sender_local_has_digit",
            "sender_is_free_provider",
            "sender_domain_len",
            "replyto_domain_len",
            "url_count_parsed",
            "distinct_url_domain_count",
            "distinct_url_registered_domain_count",
            "url_https_count",
            "url_http_count",
            "url_other_scheme_count",
            "url_has_ip_host",
            "url_shortener_count",
            "url_hyphen_host_count",
            "url_at_symbol_count",
            "url_query_count",
            "url_fragment_count",
            "url_port_count",
            "url_percent_encoded_count",
            "url_suspicious_token_count",
            "url_digit_char_count",
            "max_url_len",
            "max_url_dot_count",
            "primary_url_subdomain_depth",
            "max_url_path_depth",
            "has_long_url",
            "attachment_count_parsed",
            "attachment_has_macro_or_archive",
            "num_links",
            "has_links",
            "num_attachments",
            "has_attachment",
            "has_urgent_words",
            "sender_replyto_mismatch",
            "suspicious_sender_domain",
            "suspicious_attachment_type",
        }

        for col in X_train.columns:
            if col.startswith("labelcat__"):
                self.discrete_columns[col] = np.sort(X_train[col].dropna().unique().astype(np.float32))
            elif col.startswith("num__"):
                raw_name = col.split("num__", 1)[1]
                if raw_name in integer_numeric_raw:
                    self.discrete_columns[col] = np.sort(X_train[col].dropna().unique().astype(np.float32))

    @staticmethod
    def snap_to_valid_values(values: np.ndarray, valid_values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, valid_values[0], valid_values[-1]).astype(np.float32, copy=False)
        idx = np.searchsorted(valid_values, clipped)
        idx = np.clip(idx, 0, len(valid_values) - 1)
        prev_idx = np.clip(idx - 1, 0, len(valid_values) - 1)
        next_idx = idx
        prev_vals = valid_values[prev_idx]
        next_vals = valid_values[next_idx]
        use_next = np.abs(next_vals - clipped) <= np.abs(clipped - prev_vals)
        return np.where(use_next, next_vals, prev_vals)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = out.clip(lower=self.lower, upper=self.upper, axis="columns")

        for group_cols in self.onehot_groups:
            values = out[group_cols].to_numpy(dtype=np.float32, copy=True)
            projected = np.zeros_like(values, dtype=np.float32)
            max_idx = values.argmax(axis=1)
            projected[np.arange(len(values)), max_idx] = 1.0
            out.loc[:, group_cols] = projected

        for col, valid_values in self.discrete_columns.items():
            out[col] = self.snap_to_valid_values(out[col].to_numpy(dtype=np.float32, copy=False), valid_values)

        return out.astype(np.float32)


def generate_local_permutation(
    X_train: pd.DataFrame,
    X_train_np: np.ndarray,
    n_samples: int,
    random_state: int,
    n_neighbors: int,
    noise_scale: float,
    log_fn,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    X_values = X_train.to_numpy(dtype=np.float32, copy=False)
    n_rows, n_features = X_values.shape
    effective_neighbors = max(2, min(n_neighbors, n_rows))

    log_fn(f"[LocalPermutation] fitting neighborhood index with k={effective_neighbors}")
    neighbor_index = NearestNeighbors(n_neighbors=effective_neighbors)
    neighbor_index.fit(X_train_np)

    chunks = []
    generated = 0

    while generated < n_samples:
        batch = min(1000, n_samples - generated)
        anchor_idx = rng.integers(0, n_rows, size=batch)
        neighborhood = neighbor_index.kneighbors(X_train_np[anchor_idx], return_distance=False)
        local_cloud = X_values[neighborhood]
        local_scale = local_cloud.std(axis=1).astype(np.float32)
        local_scale = np.maximum(local_scale, 1e-3)
        noise = rng.normal(loc=0.0, scale=noise_scale, size=(batch, n_features)).astype(np.float32)
        batch_data = X_values[anchor_idx] + (noise * local_scale)

        chunks.append(batch_data)
        generated += batch
        if generated % 10000 == 0 or generated == n_samples:
            log_fn(f"[LocalPermutation] generated {generated}/{n_samples}")

    return pd.DataFrame(np.vstack(chunks), columns=X_train.columns)


def run_teacher_plausibility_filter(
    candidate_df: pd.DataFrame,
    artifact_dir: Path,
    helper_python: str,
    threshold: float,
    require_agreement: bool,
    round_idx: int,
    log_fn,
) -> Tuple[pd.DataFrame, dict]:
    temp_input = artifact_dir / f"_tmp_vae_candidates_round_{round_idx}.csv"
    temp_output = artifact_dir / f"_tmp_vae_plausibility_round_{round_idx}.npz"
    candidate_df.to_csv(temp_input, index=False, float_format="%.6g")

    command = [
        helper_python,
        "score_teacher_plausibility.py",
        "--input-csv",
        str(temp_input),
        "--output-npz",
        str(temp_output),
        "--artifact-dir",
        str(artifact_dir),
        "--threshold",
        str(threshold),
    ]
    if require_agreement:
        command.append("--require-agreement")

    log_fn(
        f"[TeacherFilter] scoring {len(candidate_df)} candidate rows with LayoutLM teachers "
        f"(threshold={threshold:.2f}, require_agreement={require_agreement})"
    )
    subprocess.run(command, check=True)

    with np.load(temp_output) as score_data:
        plausible_mask = score_data["plausible_mask"].astype(bool)
        agreement_ratio = float(score_data["agreement_mask"].mean()) if "agreement_mask" in score_data else None
        confidence_ratio = float(score_data["confidence_mask"].mean()) if "confidence_mask" in score_data else None
        mean_rf_confidence = float(score_data["random_forest_confidence"].mean())
        mean_dnn_confidence = float(score_data["deep_neural_net_confidence"].mean())
    filtered_df = candidate_df.loc[plausible_mask].reset_index(drop=True)
    stats = {
        "candidate_count": int(len(candidate_df)),
        "kept_count": int(plausible_mask.sum()),
        "kept_ratio": float(plausible_mask.mean()) if len(plausible_mask) else 0.0,
        "agreement_ratio": agreement_ratio,
        "confidence_ratio": confidence_ratio,
        "mean_rf_confidence": mean_rf_confidence,
        "mean_dnn_confidence": mean_dnn_confidence,
    }

    try:
        temp_input.unlink(missing_ok=True)
        temp_output.unlink(missing_ok=True)
    except TypeError:
        if temp_input.exists():
            temp_input.unlink()
        if temp_output.exists():
            temp_output.unlink()

    return filtered_df, stats


def add_model_outputs(df: pd.DataFrame, feature_cols, models: dict, log_fn) -> pd.DataFrame:
    X = df[feature_cols].to_numpy()
    out = df.copy()

    for model_name, model in models.items():
        log_fn(f"Scoring synthetic data with model: {model_name}")
        out[f"{model_name}_pred"] = model.predict(X)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                out[f"{model_name}_prob_1"] = proba[:, 1]
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data from train split.")
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/processed_dataset_with_split.csv",
        help="Path to processed dataset CSV with label and split columns.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="artifacts",
        help="Artifact directory containing model pickles and output CSVs.",
    )
    parser.add_argument("--n-samples", type=int, default=100000, help="Synthetic rows per method.")
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=4,
        help="Latent space dimension for VAE.",
    )
    parser.add_argument("--vae-epochs", type=int, default=25, help="Number of VAE training epochs.")
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="KL divergence weight in the VAE loss.",
    )
    parser.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=10,
        help="Linearly warm beta from 0 to --beta over this many epochs.",
    )
    parser.add_argument(
        "--sampling-temperature",
        type=float,
        default=0.85,
        help="Latent-space sampling temperature multiplier.",
    )
    parser.add_argument(
        "--local-neighbors",
        type=int,
        default=25,
        help="Neighborhood size used to estimate local noise around each anchor sample.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.15,
        help="Gaussian noise multiplier applied to the local per-feature neighborhood standard deviation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for VAE training and sampling. Use 'cuda' to require GPU.",
    )
    parser.add_argument(
        "--teacher-filter",
        action="store_true",
        help="Keep only high-plausibility VAE samples using the saved teacher models.",
    )
    parser.add_argument(
        "--teacher-filter-threshold",
        type=float,
        default=0.80,
        help="Minimum max probability required from each teacher during plausibility filtering.",
    )
    parser.add_argument(
        "--teacher-filter-overgenerate",
        type=float,
        default=1.75,
        help="Oversampling multiplier used before teacher filtering so enough rows survive.",
    )
    parser.add_argument(
        "--teacher-filter-python",
        type=str,
        default=r"C:\Users\thanh\anaconda3\envs\LayoutLM\python.exe",
        help="Python executable that can load the saved sklearn teacher models.",
    )
    parser.add_argument(
        "--teacher-filter-require-agreement",
        action="store_true",
        help="Require the RF and DNN teachers to agree before keeping a sampled row.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.10,
        help="Fraction of train rows reserved for held-out reconstruction metrics.",
    )
    parser.add_argument(
        "--include-model-outputs",
        action="store_true",
        help="Append teacher-model predictions and probabilities to the synthetic files.",
    )
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Train/evaluate the VAE and save reconstructions without generating prior samples.",
    )
    parser.add_argument(
        "--skip-local-permutation",
        action="store_true",
        help="Skip regenerating local-permutation samples and only run the VAE path.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    log = make_logger()

    if args.n_samples % 1000 == 0:
        sample_tag = f"{args.n_samples // 1000}k"
    else:
        sample_tag = str(args.n_samples)
    vae_tag = (
        f"vae_ld{args.latent_dim}_warm{args.kl_warmup_epochs}_temp{slug_value(args.sampling_temperature)}"
        + ("_filtered" if args.teacher_filter else "")
    )

    input_path = Path(args.input)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(exist_ok=True, parents=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    log(f"Loading processed dataset from {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    log(f"Loaded dataset shape: {df.shape}")

    train_df, val_df, feature_cols, X_train, y_train = split_features_target(df)
    log(f"Train rows: {len(train_df)} | Feature count: {len(feature_cols)}")
    postprocessor = ProcessedSpacePostprocessor(X_train)
    binary_indices = [idx for idx, col in enumerate(feature_cols) if col.startswith("onehotcat__")]
    if val_df.empty:
        raise ValueError("No rows with split == 'val' found; cannot compute held-out reconstruction metrics.")
    X_train_df = X_train.copy()
    X_holdout_df = val_df[feature_cols].copy()
    log(f"VAE training uses all train rows: train={len(X_train_df)} | heldout(val)={len(X_holdout_df)}")

    train_imputer = SimpleImputer(strategy="median")
    X_train_np = train_imputer.fit_transform(X_train_df).astype(np.float32)
    X_holdout_np = train_imputer.transform(X_holdout_df).astype(np.float32)

    models = None
    if args.include_model_outputs:
        models = load_models(
            artifact_dir=artifact_dir,
            log_fn=log,
            X_train_np=train_imputer.transform(X_train).astype(np.float32),
            y_train=y_train,
            random_state=args.random_state,
        )

    local_path = None
    if args.skip_local_permutation:
        log("Skipping local permutation generation")
    else:
        log("Generating synthetic data with local permutation")
        local_perm_df = generate_local_permutation(
            X_train=X_train,
            X_train_np=X_train_np,
            n_samples=args.n_samples,
            random_state=args.random_state,
            n_neighbors=args.local_neighbors,
            noise_scale=args.noise_scale,
            log_fn=log,
        )
        local_perm_df = postprocessor.transform(local_perm_df)
        if models is not None:
            local_perm_df = add_model_outputs(local_perm_df, feature_cols, models, log)

        local_path = artifact_dir / f"synthetic_local_permutation_{sample_tag}.csv"
        local_perm_df.to_csv(local_path, index=False, float_format="%.6g")
        log(f"Saved local permutation synthetic data: {local_path}")

    log("Preparing data and training VAE")

    vae = VAE(
        input_dim=X_train_np.shape[1],
        latent_dim=args.latent_dim,
        hidden_dim=256,
        binary_indices=binary_indices,
        beta=args.beta,
        device=args.device,
    )
    vae.fit(
        X=X_train_np,
        X_val=X_holdout_np,
        epochs=args.vae_epochs,
        batch_size=1024,
        lr=1e-3,
        kl_warmup_epochs=args.kl_warmup_epochs,
        log_fn=log,
    )

    heldout_metrics = vae.evaluate(X_holdout_np, batch_size=1024, beta_value=args.beta)
    log(
        f"[VAE] held-out reconstruction metrics | loss={heldout_metrics['loss']:.6f} | "
        f"bce={heldout_metrics['bce']:.6f} | mse={heldout_metrics['mse']:.6f} | kl={heldout_metrics['kl']:.6f}"
    )
    train_reconstruction = vae.reconstruct(X_train_np, batch_size=1024, deterministic=True, log_fn=log)
    train_reconstruction_df = pd.DataFrame(train_reconstruction, columns=feature_cols)
    train_reconstruction_df = postprocessor.transform(train_reconstruction_df)
    train_reconstruction_path = artifact_dir / f"synthetic_{vae_tag}_reconstruction_train.csv"
    train_reconstruction_df.to_csv(train_reconstruction_path, index=False, float_format="%.6g")
    log(f"Saved VAE full-train reconstruction CSV: {train_reconstruction_path}")

    holdout_reconstruction = vae.reconstruct(X_holdout_np, batch_size=1024, deterministic=True, log_fn=log)
    holdout_reconstruction_df = pd.DataFrame(holdout_reconstruction, columns=feature_cols)
    holdout_reconstruction_df = postprocessor.transform(holdout_reconstruction_df)
    reconstruction_path = artifact_dir / f"synthetic_{vae_tag}_reconstruction_holdout.csv"
    holdout_reconstruction_df.to_csv(reconstruction_path, index=False, float_format="%.6g")
    log(f"Saved VAE held-out reconstruction CSV: {reconstruction_path}")

    teacher_filter_stats = []
    vae_path = None
    if args.skip_sampling:
        log("Skipping prior-sample generation after saving reconstructions")
    else:
        log("Sampling synthetic data from VAE latent space")
        if args.teacher_filter:
            kept_batches = []
            kept_total = 0
            round_idx = 0
            while kept_total < args.n_samples:
                round_idx += 1
                remaining = args.n_samples - kept_total
                candidate_count = min(max(int(np.ceil(remaining * args.teacher_filter_overgenerate)), 10000), 100000)
                vae_samples = vae.sample(
                    n_samples=candidate_count,
                    batch_size=1000,
                    temperature=args.sampling_temperature,
                    log_fn=log,
                )
                candidate_df = pd.DataFrame(vae_samples, columns=feature_cols)
                candidate_df = postprocessor.transform(candidate_df)
                filtered_df, round_stats = run_teacher_plausibility_filter(
                    candidate_df=candidate_df,
                    artifact_dir=artifact_dir,
                    helper_python=args.teacher_filter_python,
                    threshold=args.teacher_filter_threshold,
                    require_agreement=args.teacher_filter_require_agreement,
                    round_idx=round_idx,
                    log_fn=log,
                )
                teacher_filter_stats.append(round_stats)
                kept_batches.append(filtered_df)
                kept_total += len(filtered_df)
                log(
                    f"[TeacherFilter] round={round_idx} | kept={round_stats['kept_count']}/{round_stats['candidate_count']} "
                    f"({round_stats['kept_ratio']:.4f}) | cumulative={kept_total}/{args.n_samples}"
                )
                if round_stats["kept_count"] == 0:
                    raise RuntimeError("Teacher filtering rejected an entire VAE batch; lower the threshold or increase temperature.")
            vae_df = pd.concat(kept_batches, ignore_index=True).iloc[: args.n_samples].copy()
        else:
            vae_samples = vae.sample(
                n_samples=args.n_samples,
                batch_size=1000,
                temperature=args.sampling_temperature,
                log_fn=log,
            )
            vae_df = pd.DataFrame(vae_samples, columns=feature_cols)
            vae_df = postprocessor.transform(vae_df)
        if models is not None:
            vae_df = add_model_outputs(vae_df, feature_cols, models, log)

        vae_path = artifact_dir / f"synthetic_{vae_tag}_{sample_tag}.csv"
        vae_df.to_csv(vae_path, index=False, float_format="%.6g")
        log(f"Saved VAE synthetic data: {vae_path}")

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_path": str(input_path.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "vae_tag": vae_tag,
        "n_samples": int(args.n_samples),
        "feature_count": int(len(feature_cols)),
        "latent_dim": int(args.latent_dim),
        "beta": float(args.beta),
        "kl_warmup_epochs": int(args.kl_warmup_epochs),
        "sampling_temperature": float(args.sampling_temperature),
        "teacher_filter": bool(args.teacher_filter),
        "teacher_filter_threshold": float(args.teacher_filter_threshold),
        "teacher_filter_require_agreement": bool(args.teacher_filter_require_agreement),
        "train_rows": int(len(X_train_df)),
        "heldout_rows": int(len(X_holdout_df)),
        "heldout_reconstruction_metrics": heldout_metrics,
        "best_validation": vae.best_validation,
        "teacher_filter_rounds": teacher_filter_stats,
        "outputs": {
            "vae_csv": str(vae_path.resolve()) if vae_path is not None else None,
            "train_reconstruction_csv": str(train_reconstruction_path.resolve()),
            "heldout_reconstruction_csv": str(reconstruction_path.resolve()),
            "local_permutation_csv": str(local_path.resolve()) if local_path is not None else None,
        },
        "training_history": vae.training_history,
    }
    summary_path = artifact_dir / f"{vae_tag}_{sample_tag}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log(f"Saved VAE summary JSON: {summary_path}")

    log("Done")


if __name__ == "__main__":
    main()
