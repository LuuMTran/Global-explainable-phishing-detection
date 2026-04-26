import ast
import json
import pickle
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path("business_phishing_dataset.csv")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

COMMON_MULTI_PART_SUFFIXES = {
    "ac.uk",
    "co.jp",
    "co.nz",
    "co.uk",
    "com.au",
    "com.br",
    "com.cn",
    "com.mx",
    "com.sg",
    "com.tr",
    "gov.au",
    "gov.uk",
    "net.au",
    "org.au",
    "org.uk",
}
FREE_EMAIL_PROVIDERS = {
    "163.com",
    "aol.com",
    "gmail.com",
    "googlemail.com",
    "hotmail.com",
    "icloud.com",
    "mail.com",
    "outlook.com",
    "proton.me",
    "protonmail.com",
    "qq.com",
    "yahoo.com",
    "yahoo.com.au",
    "ymail.com",
}
SHORTENER_DOMAINS = {
    "bit.ly",
    "buff.ly",
    "cutt.ly",
    "goo.gl",
    "is.gd",
    "ow.ly",
    "rb.gy",
    "rebrand.ly",
    "shorturl.at",
    "t.co",
    "tinyurl.com",
}
URGENT_PATTERN = re.compile(
    r"\b(urgent|immediately|asap|verify|verification|suspend|action required|payment|overdue|invoice)\b",
    flags=re.IGNORECASE,
)
ACTION_PATTERN = re.compile(
    r"\b(click here|log in|login|sign in|signin|confirm|update|review|open attachment|reset)\b",
    flags=re.IGNORECASE,
)
EMAIL_ADDRESS_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SUSPICIOUS_URL_TOKEN_PATTERN = re.compile(
    r"(login|verify|update|secure|account|password|invoice|payment|bank|confirm|signin|webscr)",
    flags=re.IGNORECASE,
)
IPV4_PATTERN = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}")

start_time = time.time()


def log_progress(msg: str) -> None:
    elapsed = time.time() - start_time
    print(f"[{elapsed:8.1f}s] {msg}", flush=True)


def parse_list_cell(value: str) -> list[str]:
    value = str(value).strip()
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return []


def safe_urlparse(url: str):
    try:
        return urlparse(url)
    except Exception:
        return None


def parsed_has_port(parsed) -> bool:
    try:
        return parsed.port is not None
    except Exception:
        return False


def normalize_domain(value: str) -> str:
    value = str(value or "").strip().lower().strip(".")
    value = value.strip("[]")
    return value or "unknown"


def extract_domain_from_email(value: str) -> str:
    match = re.search(r"@([^>\s]+)", str(value or ""))
    return normalize_domain(match.group(1)) if match else "unknown"


def extract_localpart_from_email(value: str) -> str:
    match = re.search(r"([^@\s<]+)@", str(value or ""))
    return match.group(1).strip().lower() if match else "unknown"


def extract_tld_from_domain(domain: str) -> str:
    domain = normalize_domain(domain)
    if domain == "unknown":
        return "none"
    if IPV4_PATTERN.fullmatch(domain):
        return ".ip"
    if "." not in domain:
        return "none"
    return "." + domain.split(".")[-1]


def extract_registered_domain(hostname: str) -> str:
    hostname = normalize_domain(hostname)
    if hostname == "unknown" or "." not in hostname:
        return hostname
    if IPV4_PATTERN.fullmatch(hostname):
        return hostname

    parts = [part for part in hostname.split(".") if part]
    if len(parts) < 2:
        return hostname

    two_part_suffix = ".".join(parts[-2:])
    if len(parts) >= 3 and two_part_suffix in COMMON_MULTI_PART_SUFFIXES:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def extract_primary_scheme(urls: list[str]) -> str:
    for url in urls:
        parsed = safe_urlparse(url)
        if parsed is None:
            continue
        scheme = (parsed.scheme or "").strip().lower()
        if scheme:
            return scheme
    return "none"


def extract_primary_tld(urls: list[str]) -> str:
    for url in urls:
        parsed = safe_urlparse(url)
        if parsed is None:
            continue
        hostname = normalize_domain(parsed.hostname or "")
        if hostname != "unknown" and "." in hostname:
            return extract_tld_from_domain(hostname)
    return "none"


def build_url_features(urls: list[str]) -> dict[str, float | int | str]:
    parsed_urls = [parsed for parsed in (safe_urlparse(url) for url in urls) if parsed is not None]
    hosts = [normalize_domain(parsed.hostname or "") for parsed in parsed_urls if normalize_domain(parsed.hostname or "") != "unknown"]
    registered_domains = [extract_registered_domain(host) for host in hosts]
    schemes = [(parsed.scheme or "").strip().lower() for parsed in parsed_urls if (parsed.scheme or "").strip()]

    max_url_len = max((len(url) for url in urls), default=0)
    mean_url_len = (sum(len(url) for url in urls) / len(urls)) if urls else 0.0
    dot_counts = [host.count(".") for host in hosts]
    subdomain_depths = [max(host.count(".") - 1, 0) for host in hosts]
    path_depths = [len([part for part in (parsed.path or "").split("/") if part]) for parsed in parsed_urls]

    return {
        "url_scheme": extract_primary_scheme(urls),
        "url_tld": extract_primary_tld(urls),
        "primary_url_registered_domain": registered_domains[0] if registered_domains else "none",
        "url_count_parsed": len(urls),
        "distinct_url_domain_count": len(set(hosts)),
        "distinct_url_registered_domain_count": len(set(registered_domains)),
        "url_https_count": sum(scheme == "https" for scheme in schemes),
        "url_http_count": sum(scheme == "http" for scheme in schemes),
        "url_other_scheme_count": sum(scheme not in {"http", "https"} for scheme in schemes),
        "url_has_ip_host": int(any(IPV4_PATTERN.fullmatch(host) for host in hosts)),
        "url_shortener_count": sum(host in SHORTENER_DOMAINS for host in hosts),
        "url_hyphen_host_count": sum("-" in host for host in hosts),
        "url_at_symbol_count": sum("@" in url for url in urls),
        "url_query_count": sum(bool(parsed.query) for parsed in parsed_urls),
        "url_fragment_count": sum(bool(parsed.fragment) for parsed in parsed_urls),
        "url_port_count": sum(parsed_has_port(parsed) for parsed in parsed_urls),
        "url_percent_encoded_count": sum(url.count("%") for url in urls),
        "url_suspicious_token_count": sum(len(SUSPICIOUS_URL_TOKEN_PATTERN.findall(url)) for url in urls),
        "url_digit_char_count": sum(sum(char.isdigit() for char in url) for url in urls),
        "max_url_len": max_url_len,
        "mean_url_len": mean_url_len,
        "max_url_dot_count": max(dot_counts, default=0),
        "primary_url_subdomain_depth": max(subdomain_depths, default=0),
        "max_url_path_depth": max(path_depths, default=0),
        "has_long_url": int(max_url_len >= 60),
    }


def build_attachment_features(attachments: list[str]) -> dict[str, int]:
    lowered = [attachment.lower() for attachment in attachments]
    suspicious_ext_pattern = re.compile(r"\.(zip|rar|7z|exe|js|scr|iso|img|hta|lnk|docm|xlsm|pptm)$")
    return {
        "attachment_count_parsed": len(attachments),
        "attachment_has_macro_or_archive": int(any(suspicious_ext_pattern.search(name) for name in lowered)),
    }


def evaluate_model(model, X_split, y_true):
    y_pred = model.predict(X_split)
    y_prob = model.predict_proba(X_split)[:, 1] if hasattr(model, "predict_proba") else None

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
    if y_prob is not None:
        result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return result


def main() -> dict:
    log_progress("Step 1/10: Loading dataset")
    raw_df = pd.read_csv(DATA_PATH, low_memory=False)
    log_progress(f"Loaded dataset shape: {raw_df.shape}")
    print("Columns:", list(raw_df.columns))

    log_progress("Step 2/10: Basic cleaning")
    text_cols = [
        "subject",
        "body",
        "sender_email",
        "reply_to",
        "urls",
        "attachments",
        "attack_type",
        "source_file",
    ]
    for col in text_cols:
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].fillna("").astype(str)

    if "label" not in raw_df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    log_progress("Step 3/10: Feature extraction (label-leak free)")
    feat_df = raw_df.copy()
    feat_df["parsed_urls"] = feat_df["urls"].apply(parse_list_cell)
    feat_df["parsed_attachments"] = feat_df["attachments"].apply(parse_list_cell)

    feat_df["subject_len"] = feat_df["subject"].str.len()
    feat_df["body_len"] = feat_df["body"].str.len()
    feat_df["subject_word_count"] = feat_df["subject"].str.split().str.len()
    feat_df["body_word_count"] = feat_df["body"].str.split().str.len()
    feat_df["subject_exclamation_count"] = feat_df["subject"].str.count("!")
    feat_df["body_exclamation_count"] = feat_df["body"].str.count("!")
    feat_df["subject_question_count"] = feat_df["subject"].str.count(r"\?")
    feat_df["body_question_count"] = feat_df["body"].str.count(r"\?")
    feat_df["subject_digit_count"] = feat_df["subject"].str.count(r"\d")
    feat_df["body_digit_count"] = feat_df["body"].str.count(r"\d")
    feat_df["subject_upper_ratio"] = feat_df["subject"].apply(lambda x: sum(ch.isupper() for ch in x) / max(len(x), 1))
    feat_df["body_upper_ratio"] = feat_df["body"].apply(lambda x: sum(ch.isupper() for ch in x) / max(len(x), 1))
    feat_df["money_symbol_count"] = feat_df["body"].str.count(r"[$€£]")
    feat_df["subject_urgent_word_count"] = feat_df["subject"].apply(lambda x: len(URGENT_PATTERN.findall(x)))
    feat_df["body_urgent_word_count"] = feat_df["body"].apply(lambda x: len(URGENT_PATTERN.findall(x)))
    feat_df["body_action_word_count"] = feat_df["body"].apply(lambda x: len(ACTION_PATTERN.findall(x)))
    feat_df["click_here_count"] = feat_df["body"].str.count(r"click here", flags=re.IGNORECASE)
    feat_df["body_email_address_count"] = feat_df["body"].apply(lambda x: len(EMAIL_ADDRESS_PATTERN.findall(x)))
    feat_df["http_count_body"] = feat_df["body"].str.count(r"https?://", flags=re.IGNORECASE)

    feat_df["sender_domain"] = feat_df["sender_email"].apply(extract_domain_from_email)
    feat_df["replyto_domain"] = feat_df["reply_to"].apply(extract_domain_from_email)
    feat_df["sender_tld"] = feat_df["sender_domain"].apply(extract_tld_from_domain)
    feat_df["replyto_tld"] = feat_df["replyto_domain"].apply(extract_tld_from_domain)
    feat_df["sender_localpart"] = feat_df["sender_email"].apply(extract_localpart_from_email)
    feat_df["sender_missing"] = feat_df["sender_domain"].eq("unknown").astype(int)
    feat_df["reply_missing"] = feat_df["replyto_domain"].eq("unknown").astype(int)
    feat_df["sender_reply_domain_same"] = (
        (feat_df["sender_domain"] != "unknown")
        & (feat_df["replyto_domain"] != "unknown")
        & feat_df["sender_domain"].eq(feat_df["replyto_domain"])
    ).astype(int)
    feat_df["sender_reply_domain_mismatch"] = (
        (feat_df["replyto_domain"] != "unknown")
        & feat_df["sender_domain"].ne(feat_df["replyto_domain"])
    ).astype(int)
    feat_df["sender_domain_has_digit"] = feat_df["sender_domain"].str.contains(r"\d", regex=True).astype(int)
    feat_df["sender_local_has_digit"] = feat_df["sender_localpart"].str.contains(r"\d", regex=True).astype(int)
    feat_df["sender_is_free_provider"] = feat_df["sender_domain"].isin(FREE_EMAIL_PROVIDERS).astype(int)
    feat_df["sender_domain_len"] = feat_df["sender_domain"].str.len()
    feat_df["replyto_domain_len"] = feat_df["replyto_domain"].str.len()

    url_feature_df = pd.DataFrame(feat_df["parsed_urls"].apply(build_url_features).tolist(), index=feat_df.index)
    attachment_feature_df = pd.DataFrame(
        feat_df["parsed_attachments"].apply(build_attachment_features).tolist(),
        index=feat_df.index,
    )
    feat_df = pd.concat([feat_df, url_feature_df, attachment_feature_df], axis=1)

    passthrough_numeric_candidates = [
        "num_links",
        "has_links",
        "num_attachments",
        "has_attachment",
        "has_urgent_words",
        "sender_replyto_mismatch",
        "suspicious_sender_domain",
        "suspicious_attachment_type",
    ]

    engineered_numeric_features = [
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
        "subject_upper_ratio",
        "body_upper_ratio",
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
        "mean_url_len",
        "max_url_dot_count",
        "primary_url_subdomain_depth",
        "max_url_path_depth",
        "has_long_url",
        "attachment_count_parsed",
        "attachment_has_macro_or_archive",
    ]
    leakage_columns = ["label", "attack_type", "source_file"]

    feat_df["label"] = feat_df["label"].astype(int)
    numeric_features = [col for col in engineered_numeric_features + passthrough_numeric_candidates if col in feat_df.columns]
    onehot_features = [col for col in ["url_scheme", "url_tld"] if col in feat_df.columns]
    label_encoded_features = [
        col
        for col in ["sender_domain", "replyto_domain", "sender_tld", "replyto_tld", "primary_url_registered_domain"]
        if col in feat_df.columns
    ]

    model_feature_columns = numeric_features + onehot_features + label_encoded_features
    for col in leakage_columns:
        if col in model_feature_columns:
            raise ValueError(f"Leakage column detected in features: {col}")

    X = feat_df[model_feature_columns].copy()
    y = feat_df["label"].copy()

    print(f"Numeric features: {len(numeric_features)}")
    print(f"One-hot categorical features: {len(onehot_features)} -> {onehot_features}")
    print(f"Label-encoded categorical features: {len(label_encoded_features)} -> {label_encoded_features}")
    print("Leakage columns excluded:", leakage_columns)

    log_progress("Step 4/10: Splitting dataset (70:10:20)")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(2 / 3),
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    log_progress("Step 5/10: Building preprocessing pipeline (includes normalization)")
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    label_encoder_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("label_encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    onehot_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("onehotcat", onehot_transformer, onehot_features),
            ("labelcat", label_encoder_transformer, label_encoded_features),
        ]
    )

    log_progress("Step 5/10: Fitting/transforming preprocessor")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = list(preprocessor.get_feature_names_out())
    print(f"Processed feature count: {len(feature_names)}")

    log_progress("Step 6/10: Saving processed dataset (before training)")
    train_df_proc = pd.DataFrame(X_train_proc, columns=feature_names)
    train_df_proc["label"] = y_train.to_numpy()
    train_df_proc["split"] = "train"

    val_df_proc = pd.DataFrame(X_val_proc, columns=feature_names)
    val_df_proc["label"] = y_val.to_numpy()
    val_df_proc["split"] = "val"

    test_df_proc = pd.DataFrame(X_test_proc, columns=feature_names)
    test_df_proc["label"] = y_test.to_numpy()
    test_df_proc["split"] = "test"

    processed_all = pd.concat([train_df_proc, val_df_proc, test_df_proc], axis=0, ignore_index=True)
    processed_path = ARTIFACT_DIR / "processed_dataset_with_split.csv"
    processed_all.to_csv(processed_path, index=False)

    if (not processed_path.exists()) or processed_path.stat().st_size <= 0:
        raise RuntimeError("Processed dataset was not saved correctly.")

    log_progress(
        f"Processed CSV saved: {processed_path.resolve()} ({processed_path.stat().st_size} bytes)"
    )

    log_progress("Step 7/10: Initializing models")
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        ),
        "deep_neural_net": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=5e-4,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            batch_size=512,
            max_iter=60,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=6,
            verbose=True,
        ),
    }

    metrics = {}
    log_progress("Step 8/10: Training and evaluating models")
    for name, model in models.items():
        model_start = time.time()
        log_progress(f"Training started: {name}")
        model.fit(X_train_proc, y_train)
        log_progress(f"Training finished: {name} in {time.time() - model_start:.1f}s")

        metrics[name] = {
            "train": evaluate_model(model, X_train_proc, y_train),
            "val": evaluate_model(model, X_val_proc, y_val),
            "test": evaluate_model(model, X_test_proc, y_test),
        }

        with open(ARTIFACT_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)
        log_progress(f"Saved model pickle: {name}")

    with open(ARTIFACT_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    log_progress("Step 9/10: Saving metrics and artifact manifest")
    summary = {
        "random_state": RANDOM_STATE,
        "split_ratio": {"train": 0.7, "val": 0.1, "test": 0.2},
        "rows": {
            "train": int(len(train_df_proc)),
            "val": int(len(val_df_proc)),
            "test": int(len(test_df_proc)),
            "total": int(len(processed_all)),
        },
        "features": {
            "numeric": numeric_features,
            "onehot": onehot_features,
            "label_encoded": label_encoded_features,
            "excluded_leakage_columns": leakage_columns,
        },
        "metrics": metrics,
    }

    metrics_path = ARTIFACT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    artifact_manifest = {
        "processed_dataset": str(processed_path.resolve()),
        "preprocessor": str((ARTIFACT_DIR / "preprocessor.pkl").resolve()),
        "random_forest_model": str((ARTIFACT_DIR / "random_forest.pkl").resolve()),
        "deep_neural_net_model": str((ARTIFACT_DIR / "deep_neural_net.pkl").resolve()),
        "metrics": str(metrics_path.resolve()),
    }

    manifest_path = ARTIFACT_DIR / "artifact_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(artifact_manifest, f, indent=2)

    for model_name, model_results in metrics.items():
        print(f"\n===== {model_name} =====")
        for split_name in ["train", "val", "test"]:
            split_metrics = model_results[split_name]
            roc_auc = split_metrics.get("roc_auc")
            roc_text = f", roc_auc={roc_auc:.4f}" if roc_auc is not None else ""
            print(
                f"{split_name}: accuracy={split_metrics['accuracy']:.4f}, "
                f"precision={split_metrics['precision']:.4f}, "
                f"recall={split_metrics['recall']:.4f}, "
                f"f1={split_metrics['f1']:.4f}{roc_text}"
            )

    print("\nSaved artifacts:")
    print(f"- {processed_path.resolve()}")
    print(f"- {(ARTIFACT_DIR / 'preprocessor.pkl').resolve()}")
    print(f"- {(ARTIFACT_DIR / 'random_forest.pkl').resolve()}")
    print(f"- {(ARTIFACT_DIR / 'deep_neural_net.pkl').resolve()}")
    print(f"- {metrics_path.resolve()}")
    print(f"- {manifest_path.resolve()}")
    log_progress("Step 10/10: Pipeline completed")
    return summary


if __name__ == "__main__":
    main()
