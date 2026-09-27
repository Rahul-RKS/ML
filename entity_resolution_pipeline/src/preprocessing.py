"""Name and address normalization.

Deterministic string cleanup only (no external lookups). Country is kept as an
open string label - lowercased/stripped but never filtered or hard-coded to a
fixed set, so unseen labels pass through unchanged.
"""

import re
import string
import time

# ASCII punctuation -> space, applied with str.translate (C-speed, far faster than
# a regex sub over millions of rows). Unicode letters are left untouched.
_PUNCT_TABLE = {ord(c): " " for c in string.punctuation}

# Legal suffixes / entity-type words collapsed to a canonical token so that
# "Acme Corp", "Acme Corporation" and "Acme Inc" line up on the name-core.
LEGAL_SUFFIX_MAP = {
    "corporation": "corp", "corp": "corp", "incorporated": "inc", "inc": "inc",
    "company": "co", "co": "co", "limited": "ltd", "ltd": "ltd", "llc": "llc",
    "llp": "llp", "lp": "lp", "plc": "plc",
    "private": "pvt", "pvt": "pvt", "pte": "pte",
    "gmbh": "gmbh", "ag": "ag", "sa": "sa", "sarl": "sarl", "sas": "sas",
    "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa",
}

# Address abbreviations expanded to a single canonical form.
ADDRESS_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "ln": "lane", "dr": "drive", "hwy": "highway",
    "sq": "square", "apt": "apartment", "bldg": "building", "fl": "floor",
    "ste": "suite", "opp": "opposite", "nr": "near", "no": "number",
    "sec": "sector", "ph": "phase", "flr": "floor", "cross": "cross",
    "main": "main", "blk": "block", "jn": "junction", "jnc": "junction",
}

_WS_RE = re.compile(r"\s+")
# Landmark phrases ("near SBI ATM", "opposite ...") — captured as a flag, then
# stripped from the address core so they do not dominate token similarity.
_LANDMARK_RE = re.compile(r"\b(near|opp|opposite|behind|beside|next to|above|below)\b")
# Postal codes: 6-digit India PIN, 5-digit US ZIP, French 5-digit, generic 4-6.
_POSTAL_RE = re.compile(r"\b(\d{4,6})\b")


def _clean_base(text):
    if text is None:
        return ""
    text = str(text).lower()
    if "&" in text:
        text = text.replace("&", " and ")  # keep "&" and "and" consistent
    text = text.translate(_PUNCT_TABLE)    # strip punctuation (fast path)
    return _WS_RE.sub(" ", text).strip()


def normalize_name(name):
    """Lowercase, de-punctuate, collapse '&'->'and', canonicalize legal suffixes."""
    base = _clean_base(name)
    if not base:
        return ""
    tokens = [LEGAL_SUFFIX_MAP.get(tok, tok) for tok in base.split()]
    return " ".join(tokens)


def name_core(name):
    """Name with legal-suffix tokens removed entirely — the distinctive part.

    "acme corp" and "acme inc" both reduce to "acme", which is what we want for
    high-recall blocking.
    """
    norm = normalize_name(name)
    suffix_tokens = set(LEGAL_SUFFIX_MAP.values())
    core = [t for t in norm.split() if t not in suffix_tokens]
    return " ".join(core) if core else norm


def extract_postal(address):
    """Return the first plausible postal code found, or '' if none."""
    if not address:
        return ""
    m = _POSTAL_RE.search(str(address))
    return m.group(1) if m else ""


def normalize_address(address):
    """Expand common abbreviations, drop landmark phrases, de-punctuate.

    Returns (normalized_address, has_landmark_flag, postal_code).
    """
    if address is None:
        return "", 0, ""
    postal = extract_postal(address)
    raw = str(address).lower()
    has_landmark = 1 if _LANDMARK_RE.search(raw) else 0
    base = _clean_base(address)
    if not base:
        return "", has_landmark, postal
    # Drop landmark keyword tokens so they do not inflate token overlap.
    landmark_words = {"near", "opp", "opposite", "behind", "beside", "next", "above", "below"}
    tokens = [ADDRESS_ABBREV.get(t, t) for t in base.split() if t not in landmark_words]
    return _WS_RE.sub(" ", " ".join(tokens)).strip(), has_landmark, postal


def normalize_country(country):
    """Lowercase/strip only. Never filter or map to a fixed set (France is unseen)."""
    if country is None:
        return ""
    return str(country).strip().lower()


def preprocess_frame(df):
    """Build a lean normalized frame from a source dataframe.

    Expects columns: entity_id, business_name, business_address, country.
    Returns a NEW minimal frame with only the columns anything downstream uses:
    entity_id, name_core, addr_norm, has_landmark, postal, country_norm, blob.

    This is written for multi-million-row scale: we iterate the raw columns once
    as plain Python lists (no pandas string '+', no intermediate Series of
    tuples), never keep the raw text columns, and drop the unused name_norm
    column entirely (features use name_core; blocking uses blob).
    """
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()
    countries = df["country"].tolist()
    ids = df["entity_id"].tolist()
    # Free the source frame's big string columns before we build the new one.
    del df

    n = len(ids)
    name_cores, addr_norms, has_landmarks, postals, blobs = [], [], [], [], []
    started = time.perf_counter()
    for idx, (nm, ad) in enumerate(zip(names, addrs), start=1):
        nc = name_core(nm)
        an, lm, pc = normalize_address(ad)
        name_cores.append(nc)
        addr_norms.append(an)
        has_landmarks.append(lm)
        postals.append(pc)
        blobs.append(f"{nc} {an}".strip())
        if idx % 100_000 == 0 or idx == n:
            elapsed = time.perf_counter() - started
            rate = idx / elapsed if elapsed else 0.0
            print(
                f"        normalized {idx:,}/{n:,} rows "
                f"({rate:,.0f} rows/s, {elapsed:.0f}s elapsed)",
                flush=True,
            )

    import pandas as pd
    return pd.DataFrame({
        "entity_id": ids,
        "name_core": name_cores,
        "addr_norm": addr_norms,
        "has_landmark": has_landmarks,
        "postal": postals,
        "country_norm": [normalize_country(c) for c in countries],
        "blob": blobs,
    })
