"""Funcții comune pentru încărcarea și curățarea datelor.

Codul este mutat din notebook-urile de la Seminarul 4 (T1_phishing.ipynb și T2_ai_text.ipynb),
ca să poată fi refolosit de agenți și, mai târziu, de aplicația web.
"""
import html
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

SEED = 42

# ---------------------------------------------------------------------------
# Curățarea textului
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
HTML_RE = re.compile(r"<[^>]+>")


def clean_text(text):
    """Curăță un email: scoate HTML-ul și secvențele "\\r\\n" rămase ca text,
    înlocuiește linkurile cu 'URL' și adresele de email cu 'EMAIL', elimină spațiile multiple."""
    text = html.unescape(str(text))
    text = re.sub(r"\\[rnt]", " ", text)
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" URL ", text)
    text = EMAIL_RE.sub(" EMAIL ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_dataframe(df):
    """Aceiași pași ca în notebook-urile de la Seminarul 4: curățare, eliminarea
    emailurilor (aproape) goale și a duplicatelor."""
    df = df.dropna(subset=["text_raw"]).copy()
    df["text"] = df["text_raw"].apply(clean_text)
    df = df[df["text"].str.split().str.len() >= 3]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["id"] = np.arange(len(df))
    return df


# ---------------------------------------------------------------------------
# Încărcarea seturilor de date
# ---------------------------------------------------------------------------
def _pick_text(frame, candidates):
    cols = {c.lower().strip(): c for c in frame.columns}
    if "body" in cols:
        subject = frame[cols["subject"]].fillna("").astype(str) if "subject" in cols else ""
        return subject + " " + frame[cols["body"]].fillna("").astype(str)
    for cand in candidates:
        if cand in cols:
            return frame[cols[cand]].astype(str)
    others = [c for c in frame.columns if c.lower().strip() != "label"]
    return frame[max(others, key=lambda c: frame[c].astype(str).str.len().mean())].astype(str)


def load_t1(csv_files):
    """Setul de date T1 (Al-Subaiey et al., 2024): 1 = phishing, 0 = legitim."""
    combined = [f for f in csv_files if os.path.basename(f).lower() == "phishing_email.csv"]
    parts = []
    for f in combined or csv_files:
        frame = pd.read_csv(f, on_bad_lines="skip", low_memory=False)
        label_col = next((c for c in frame.columns if c.lower().strip() in ["label", "class", "email type"]), None)
        if label_col is None:
            continue
        lab = frame[label_col].astype(str).str.strip().str.lower()
        if lab.isin(["0", "1", "0.0", "1.0"]).all():
            lab = lab.astype(float).astype(int)
        else:
            lab = lab.str.contains("phish|spam|fraud|malicious").astype(int)
        parts.append(pd.DataFrame({
            "text_raw": _pick_text(frame, ["text_combined", "text", "email text", "email_text", "message"]).values,
            "label": lab.values}))
    return pd.concat(parts, ignore_index=True)


LABEL_PATTERNS = [re.compile(r',\s*"?(\d{1,2})"?\s*$'), re.compile(r',\s*"?([A-Za-z_\-]{2,15})"?\s*$')]


def _read_csv_robust(path):
    try:
        frame = pd.read_csv(path)
        if frame.shape[1] >= 2 and len(frame) > 0:
            return frame
    except Exception:
        pass
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()[1:]
    for pattern in LABEL_PATTERNS:
        rows, buffer = [], []
        for line in lines:
            buffer.append(line)
            m = pattern.search(line)
            if m:
                full = "\n".join(buffer)
                rows.append((full[:full.rfind(",")].strip().strip('"'), m.group(1)))
                buffer = []
        if rows:
            return pd.DataFrame(rows, columns=["text", "label"])
    return pd.DataFrame(columns=["text", "label"])


def load_t2(csv_files):
    """Setul de date T2 (Greco et al., 2024): 1 = generat de AI, 0 = scris de om."""
    root = os.path.commonpath(csv_files) if len(csv_files) > 1 else os.path.dirname(csv_files[0])
    parts = []
    for f in csv_files:
        rel = os.path.relpath(f, root).lower()
        is_llm = any(k in rel for k in ["llm", "gpt", "ai_", "ai-", "_ai", "-ai", "machine"])
        is_human = "human" in rel
        origin = "ai" if is_llm and not is_human else ("human" if is_human and not is_llm else None)
        if origin is None:
            continue
        email_type = "phishing" if "phish" in rel else "legit"
        frame = _read_csv_robust(f)
        parts.append(pd.DataFrame({"text_raw": _pick_text(frame, ["text", "email", "content", "message"]).values,
                                   "origin": origin, "email_type": email_type}))
    df = pd.concat(parts, ignore_index=True)
    df["label"] = (df["origin"] == "ai").astype(int)
    return df


# ---------------------------------------------------------------------------
# Modelul de bază, refolosit de agentul hibrid și de aplicație
# ---------------------------------------------------------------------------
def train_baseline(df, test_texts, stratify_col="label"):
    """Antrenează TF-IDF + regresie logistică pe toate emailurile care NU sunt în setul de test
    salvat la Seminarul 4. Parametrul C se alege pe un set de validare, ca în notebook-uri."""
    rest = df[~df["text"].isin(set(test_texts))]
    train, val = train_test_split(rest, test_size=0.15 / 0.85, random_state=SEED, stratify=rest[stratify_col])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                                 sublinear_tf=True, strip_accents="unicode")
    X_train = vectorizer.fit_transform(train["text"])
    X_val = vectorizer.transform(val["text"])
    best = None
    for C in [0.1, 1, 10]:
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(X_train, train["label"])
        f1 = f1_score(val["label"], clf.predict(X_val))
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    return vectorizer, best[2], {"C": best[1], "f1_validare": best[0], "n_antrenare": len(train)}


def ml_predict(vectorizer, clf, texts, top_k=5):
    """Probabilitatea clasei 1 și cuvintele care au contat cel mai mult, pentru fiecare text."""
    X = vectorizer.transform(list(texts))
    probs = clf.predict_proba(X)[:, 1]
    names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]
    top_words = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        contrib = row.data * coefs[row.indices]
        order = np.argsort(-np.abs(contrib))[:top_k]
        top_words.append([(names[row.indices[j]], round(float(contrib[j]), 3)) for j in order])
    return probs, top_words
