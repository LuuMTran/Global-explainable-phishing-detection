import os
import re
import json
import html
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


# =========================
# CONFIG
# =========================
DATA_ROOT = "datasets"   # Put all dataset files here
OUTPUT_DIR = "output"
OUTPUT_CSV = "business_phishing_dataset.csv"
OUTPUT_JSON = "business_phishing_dataset.json"


# =========================
# TEXT CLEANING
# =========================
def clean_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    text = str(text)
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove quoted-printable soft line breaks
    text = text.replace("=\n", "").replace("=\r\n", "")

    # Make CSV-friendly
    text = text.replace("\n", " ").replace("\r", " ")

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_urls(text: str) -> List[str]:
    if not text:
        return []
    pattern = r"(https?://[^\s\"'<>]+|www\.[^\s\"'<>]+)"
    urls = re.findall(pattern, text)
    return list(dict.fromkeys(urls))


def has_urgent_words(text: str) -> int:
    words = [
        "urgent", "immediately", "asap", "action required", "important",
        "verify", "suspended", "locked", "update now", "click below",
        "confirm now", "reset password", "security alert"
    ]
    text = text.lower()
    return int(any(w in text for w in words))


def normalize_label(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    s = str(value).strip().lower()

    if s in {"1", "phishing", "spam", "malicious", "fraud"}:
        return 1
    if s in {"0", "legitimate", "ham", "benign", "normal"}:
        return 0

    try:
        n = int(float(s))
        if n in (0, 1):
            return n
    except Exception:
        pass

    return None


def infer_attack_type(body: str, subject: str = "") -> str:
    text = f"{subject} {body}".lower()

    rules = [
        ("credential_theft", [
            "login", "log in", "sign in", "password", "verify account",
            "account suspended", "mailbox", "office 365", "microsoft 365",
            "confirm your identity", "webmail"
        ]),
        ("invoice_fraud", [
            "invoice", "payment", "wire transfer", "bank transfer",
            "purchase order", "remittance", "receipt"
        ]),
        ("executive_impersonation", [
            "ceo", "cfo", "gift card", "confidential task", "urgent request",
            "kindly handle this", "are you available"
        ]),
        ("fake_it_alert", [
            "security alert", "mailbox full", "password expires",
            "reset your password", "account locked", "server upgrade"
        ]),
        ("vendor_impersonation", [
            "supplier", "vendor", "quotation", "update payment details",
            "new bank details"
        ])
    ]

    for attack_type, keywords in rules:
        if any(k in text for k in keywords):
            return attack_type

    return "unknown"


def extract_domain(email_text: str) -> str:
    if not email_text or "@" not in email_text:
        return ""
    match = re.search(r"@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", email_text)
    return match.group(1).lower() if match else ""


# =========================
# COLUMN DETECTION
# =========================
def find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    for col in columns:
        col_low = col.lower()
        for candidate in candidates:
            if candidate.lower() in col_low:
                return col

    return None


TEXT_CANDIDATES = [
    "body", "text", "email", "email_text", "message", "content",
    "email body", "email_body", "raw_text", "raw_email"
]

SUBJECT_CANDIDATES = [
    "subject", "title", "email_subject"
]

LABEL_CANDIDATES = [
    "label", "class", "target", "is_phishing"
]

SENDER_CANDIDATES = [
    "sender", "from", "from_email", "email_from"
]

REPLYTO_CANDIDATES = [
    "reply_to", "replyto"
]


# =========================
# CSV PARSER
# =========================
def parse_labelled_csv(file_path: Path) -> List[Dict[str, Any]]:
    try:
        df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)
    except Exception as e:
        print(f"[SKIP] {file_path.name} -> could not read CSV: {e}")
        return []

    if df.empty:
        return []

    columns = list(df.columns)

    text_col = find_column(columns, TEXT_CANDIDATES)
    label_col = find_column(columns, LABEL_CANDIDATES)
    subject_col = find_column(columns, SUBJECT_CANDIDATES)
    sender_col = find_column(columns, SENDER_CANDIDATES)
    replyto_col = find_column(columns, REPLYTO_CANDIDATES)

    if text_col is None or label_col is None:
        print(f"[SKIP] {file_path.name} -> no usable text column or label column")
        return []

    records = []

    for _, row in df.iterrows():
        body = clean_text(row[text_col])
        label = normalize_label(row[label_col])

        if not body or label is None:
            continue

        subject = clean_text(row[subject_col]) if subject_col else ""
        sender = clean_text(row[sender_col]) if sender_col else ""
        reply_to = clean_text(row[replyto_col]) if replyto_col else ""

        # Always extract real URLs from the body.
        # This avoids treating binary url flags like "1" as actual URLs.
        urls = extract_urls(body)

        record = {
            "subject": subject,
            "body": body,
            "sender_email": sender,
            "reply_to": reply_to,
            "urls": urls,
            "attachments": [],
            "label": label,
            "attack_type": infer_attack_type(body, subject) if label == 1 else "legitimate",
            "source_file": str(file_path)
        }
        records.append(record)

    print(f"[OK] Parsed CSV: {file_path.name} -> {len(records)} rows")
    return records


# =========================
# TXT MAILBOX PARSER
# =========================
def parse_mbox_txt(file_path: Path, default_label: int = 1) -> List[Dict[str, Any]]:
    try:
        text = file_path.read_text(encoding="latin-1", errors="ignore")
    except Exception as e:
        print(f"[SKIP] {file_path.name} -> could not read TXT: {e}")
        return []

    # Split mailbox-style file into chunks
    chunks = re.split(r"\nFrom .+?\n", text)

    records = []

    for chunk in chunks:
        if len(chunk.strip()) < 100:
            continue

        subject_match = re.search(r"^Subject:\s*(.*)$", chunk, flags=re.I | re.M)
        from_match = re.search(r"^From:\s*(.*)$", chunk, flags=re.I | re.M)
        reply_to_match = re.search(r"^Reply-To:\s*(.*)$", chunk, flags=re.I | re.M)

        subject = clean_text(subject_match.group(1)) if subject_match else ""
        sender = clean_text(from_match.group(1)) if from_match else ""
        reply_to = clean_text(reply_to_match.group(1)) if reply_to_match else ""

        body = clean_text(chunk)
        if not body:
            continue

        urls = extract_urls(body)

        record = {
            "subject": subject,
            "body": body,
            "sender_email": sender,
            "reply_to": reply_to,
            "urls": urls,
            "attachments": [],
            "label": default_label,
            "attack_type": infer_attack_type(body, subject) if default_label == 1 else "legitimate",
            "source_file": str(file_path)
        }
        records.append(record)

    print(f"[OK] Parsed TXT mailbox: {file_path.name} -> {len(records)} rows")
    return records


# =========================
# FEATURE ENGINEERING
# =========================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["num_links"] = df["urls"].apply(len)
    df["has_links"] = (df["num_links"] > 0).astype(int)
    df["num_attachments"] = 0
    df["has_attachment"] = 0
    df["has_urgent_words"] = df["body"].apply(has_urgent_words)

    def replyto_mismatch(row: pd.Series) -> int:
        sender_domain = extract_domain(row["sender_email"])
        reply_domain = extract_domain(row["reply_to"])
        if not sender_domain or not reply_domain:
            return 0
        return int(sender_domain != reply_domain)

    df["sender_replyto_mismatch"] = df.apply(replyto_mismatch, axis=1)

    def suspicious_sender(sender_email: str) -> int:
        sender_email = str(sender_email).lower()
        return int(bool(re.search(r"\d|secure|verify|update|login", sender_email)))

    df["suspicious_sender_domain"] = df["sender_email"].apply(suspicious_sender)
    df["suspicious_attachment_type"] = 0

    return df


# =========================
# MAIN BUILDER
# =========================
def build_dataset(data_root: str) -> pd.DataFrame:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {data_root}")

    all_records = []

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".csv":
                all_records.extend(parse_labelled_csv(file_path))

            elif suffix == ".txt":
                # For mailbox-style phishing txt files
                all_records.extend(parse_mbox_txt(file_path, default_label=1))

        except Exception as e:
            print(f"[ERROR] {file_path.name}: {e}")

    if not all_records:
        raise ValueError("No usable records found.")

    df = pd.DataFrame(all_records)

    # Remove empty body rows
    df = df[df["body"].str.len() > 0].copy()

    # Deduplicate
    df["dedupe_key"] = (
        df["subject"].fillna("").str.lower().str.strip() + "||" +
        df["body"].fillna("").str.lower().str.strip() + "||" +
        df["label"].astype(str)
    )

    before = len(df)
    df = df.drop_duplicates(subset="dedupe_key").copy()
    after = len(df)
    print(f"[INFO] Removed {before - after} duplicate rows")

    df = add_features(df)

    final_columns = [
        "subject",
        "body",
        "sender_email",
        "reply_to",
        "urls",
        "attachments",
        "label",
        "attack_type",
        "num_links",
        "has_links",
        "num_attachments",
        "has_attachment",
        "has_urgent_words",
        "sender_replyto_mismatch",
        "suspicious_sender_domain",
        "suspicious_attachment_type",
        "source_file"
    ]

    df = df[final_columns].reset_index(drop=True)
    return df


def save_outputs(df: pd.DataFrame, output_dir: str, csv_name: str, json_name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, csv_name)
    json_path = os.path.join(output_dir, json_name)

    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {csv_path}")
    print(f"[SAVED] {json_path}")


def print_summary(df: pd.DataFrame) -> None:
    print("\n========== SUMMARY ==========")
    print(f"Total rows: {len(df)}")
    print("\nLabel distribution:")
    print(df["label"].value_counts(dropna=False))
    print("\nAttack type distribution:")
    print(df["attack_type"].value_counts(dropna=False))
    print("\nColumns:")
    print(list(df.columns))
    print("\nSample rows:")
    print(df.head(5).to_string())


if __name__ == "__main__":
    df_final = build_dataset(DATA_ROOT)
    print_summary(df_final)
    save_outputs(df_final, OUTPUT_DIR, OUTPUT_CSV, OUTPUT_JSON)