#!/usr/bin/env python3
"""
Multi-agent variant prioritization with optional mapping-CSV support.
Same as multi_agent_multiprocessor.py but can load a CloudMap mapping-interval
variants CSV (e.g. outputs/mapping_interval_X_0_variants.csv) to enrich variants
with Parental_ratio, Hawaiian_ratio, QUAL, DP, EMS_label, and optionally
Change_type for scoring, pre-filtering, and LLM input/output.
"""

import csv
import copy
import json
import math
import os
import pathlib
import re
import shutil
import sys
import textwrap
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError, confloat, conint, field_validator
from tabulate import tabulate

# =========================
# Paths
# =========================
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
CSV_PATH = os.getenv("CSV_PATH", "drp1.csv")
OUTPUT_DIR = pathlib.Path("llm_variant_prioritization_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_STDIN_LINES: List[str] = []
if not sys.stdin.isatty():
    try:
        _STDIN_LINES = [line.rstrip("\n\r") for line in sys.stdin]
    except Exception:
        pass


def _input(prompt: str) -> str:
    if _STDIN_LINES:
        return _STDIN_LINES.pop(0) if _STDIN_LINES else ""
    return input(prompt).strip()

# =========================
# API configuration
# =========================
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE", "")
OPENAI_SEED = os.getenv("OPENAI_SEED", "")
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "360"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "5"))
OPENAI_BACKOFF_BASE = float(os.getenv("OPENAI_BACKOFF_BASE", "2.0"))

TOP_K = 30
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
os.environ["OPENAI_API_KEY"] = "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"
os.environ["NCBI_API_KEY"] = "605418d3446353da6f9b17b37310caf65e08"
WORMBASE_HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    ),
}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except Exception:
        return default


RESERVED_CPUS = max(0, _int_env("MULTI_AGENT_RESERVED_CPUS", 2))
AVAILABLE_CPUS = max(1, os.cpu_count() or 1)
PARALLEL_WORKERS = max(1, AVAILABLE_CPUS - RESERVED_CPUS)
print(
    f"[Parallel] CPUs={AVAILABLE_CPUS}, reserved={RESERVED_CPUS}, worker_slots={PARALLEL_WORKERS}",
    flush=True,
)


def stage_worker_count(item_count: int) -> int:
    if item_count <= 1:
        return 1
    return max(1, min(PARALLEL_WORKERS, item_count))


def parallel_map_ordered(items: List[Any], worker_fn) -> List[Any]:
    if not items:
        return []
    workers = stage_worker_count(len(items))
    if workers == 1:
        return [worker_fn(item) for item in items]

    results: List[Any] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(worker_fn, item): idx for idx, item in enumerate(items)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return results

def _with_fallback(p: str, candidates: List[str]) -> str:
    pth = pathlib.Path(p)
    if pth.exists():
        return str(pth)
    for c in candidates:
        if pathlib.Path(c).exists():
            print(f"[Info] Using fallback file: {c}")
            return c
    return str(pth)


def exists_or_raise(p: str) -> None:
    if not pathlib.Path(p).exists():
        raise FileNotFoundError(
            f"File not found: {p}\n"
            "Upload or place drp1.csv in the working directory, or set CSV_PATH."
        )


CSV_PATH = _with_fallback(
    CSV_PATH,
    [
        "/mnt/data/drp1.csv",
        "/mnt/data/hu80_WS245_annotated-variants-mapping-region.csv",
        "/mnt/data/hu80_WS245_annotated-variants-mapping-region (1).csv",
    ],
)

# =========================
# Networking helpers
# =========================
def _requests_get(
    url: str,
    params: Dict[str, Any],
    timeout: int = 30,
    retries: int = 3,
    backoff: float = 2.0,
) -> requests.Response:
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(backoff**i)


def _requests_post_with_retries(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int,
    max_retries: int,
    backoff_base: float,
) -> requests.Response:
    attempt = 0
    while True:
        try:
            return requests.post(url, headers=headers, json=payload, timeout=timeout)
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ):
            if attempt >= max_retries - 1:
                raise
            time.sleep(backoff_base**attempt)
            attempt += 1


def _post_chat(payload: Dict[str, Any], timeout: int = OPENAI_TIMEOUT) -> str:
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
        "Content-Type": "application/json",
    }
    r = _requests_post_with_retries(
        f"{OPENAI_BASE_URL}/chat/completions",
        headers=headers,
        payload=payload,
        timeout=timeout,
        max_retries=OPENAI_MAX_RETRIES,
        backoff_base=OPENAI_BACKOFF_BASE,
    )
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text[:2000]}")
    data = r.json()
    return data["choices"][0]["message"]["content"]


def openai_chat(
    messages: List[Dict[str, str]],
    model: str,
    response_json: bool = True,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    timeout: int = OPENAI_TIMEOUT,
) -> str:
    payload: Dict[str, Any] = {"model": model, "messages": messages}
    if response_json:
        payload["response_format"] = {"type": "json_object"}
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if seed is not None:
        payload["seed"] = int(seed)

    try:
        return _post_chat(payload, timeout=timeout)
    except RuntimeError as err:
        body = str(err)
        retried = False
        if ("param" in body and "temperature" in body) or (
            "Unsupported value" in body and "temperature" in body
        ):
            payload.pop("temperature", None)
            retried = True
        if "param" in body and '"seed"' in body:
            payload.pop("seed", None)
            retried = True
        if ("param" in body and "response_format" in body) or ("json_object" in body):
            payload.pop("response_format", None)
            retried = True
        if retried:
            return _post_chat(payload, timeout=timeout)
        raise


# =========================
# CSV helpers
# =========================
def coalesce(colnames: List[str], df: pd.DataFrame) -> Optional[str]:
    for c in colnames:
        if c in df.columns:
            return c
    return None


def normalize_chr_label(chrom: Any) -> str:
    s = str(chrom)
    s = re.sub(r"(?i)^chromosome\s*", "", s)
    s = re.sub(r"(?i)^chr", "", s)
    s = s.strip()
    map_arabic = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "10": "X"}
    if s in map_arabic:
        return map_arabic[s]
    return s.upper()


def build_variant_id(chrom: Any, pos: Any, ref: Any, alt: Any) -> str:
    c = normalize_chr_label(chrom) if pd.notna(chrom) else "NA"
    p = str(int(pos)) if pd.notna(pos) else "NA"
    r = str(ref).upper() if pd.notna(ref) else "N"
    a = str(alt).upper() if pd.notna(alt) else "N"
    return f"{c}:{p} {r}>{a}"


def short(s: Any, maxlen: int = 60) -> str:
    t = "" if s is None else str(s)
    return t if len(t) <= maxlen else (t[: maxlen - 3] + "...")


def coerce_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return None
        f = float(x)
        if math.isnan(f):
            return None
        return f
    except Exception:
        return None


# =========================
# Mapping and CSV configuration (requires coerce_float)
# =========================
# Optional mapping CSV (CloudMap mapping-interval variants)
MAPPING_OUTPUT_PATH = os.getenv("MAPPING_OUTPUT_PATH", "").strip()
if not MAPPING_OUTPUT_PATH:
    MAPPING_OUTPUT_PATH = _input(
        "Optional mapping CSV path (e.g. outputs/mapping_interval_X_0_variants.csv). Press Enter to skip: "
    ).strip()
if MAPPING_OUTPUT_PATH:
    MAPPING_OUTPUT_PATH = _with_fallback(
        MAPPING_OUTPUT_PATH,
        ["outputs/mapping_interval_X_0_variants.csv", "mapping_interval_X_0_variants.csv"],
    )
USE_MAPPING = bool(MAPPING_OUTPUT_PATH and pathlib.Path(MAPPING_OUTPUT_PATH).exists())

# Optional variant CSV path from stdin (e.g. from run_prioritising.py / prioritising_input.txt line 4)
_variant_csv_stdin = _input(
    "Optional main variant CSV path (e.g. drp1.csv). Press Enter to use default: "
).strip()
if _variant_csv_stdin:
    CSV_PATH = _variant_csv_stdin
    if not pathlib.Path(CSV_PATH).is_absolute():
        CSV_PATH = str(SCRIPT_DIR / CSV_PATH)

FILTER_MIN_PARENTAL_RATIO = coerce_float(os.getenv("FILTER_MIN_PARENTAL_RATIO", "0.40")) or 0.40
PREFER_EMS = os.getenv("PREFER_EMS", "true").strip().lower() in ("1", "true", "yes")

def prompt_user_annotation_file(script_dir: pathlib.Path) -> str:
    file_name = _input(
        "Optional user annotation CSV filename in this script directory "
        "(e.g. user_annotation.csv). Press Enter to skip: "
    ).strip()
    if not file_name:
        print("[Info] No user annotation file provided. Continuing without annotations.")
        return ""
    if not file_name.lower().endswith(".csv"):
        print(f"[Info] '{file_name}' is not a .csv file. Continuing without annotations.")
        return ""

    candidate = script_dir / file_name
    if not candidate.exists():
        print(
            f"[Info] User annotation file not found in {script_dir}: {file_name}. "
            "Continuing without annotations."
        )
        return ""
    return str(candidate)


def normalize_gene_key(gene: Any) -> str:
    if gene is None:
        return ""
    return re.sub(r"\s+", "", str(gene).strip()).upper()


def load_user_annotations(file_path: str) -> Tuple[Dict[str, str], str]:
    path = pathlib.Path(file_path or "")
    if not file_path or not file_path.strip():
        return {}, ""
    if not path.exists():
        print(f"[Info] User annotation file not found: {file_path}. Continuing without annotations.")
        return {}, ""

    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    annotation_map: Dict[str, str] = {}
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        unquoted = line.strip().strip('"').strip("'")

        gene = ""
        feature = ""
        if "\t" in unquoted:
            gene, feature = unquoted.split("\t", 1)
        else:
            parts: List[str] = []
            try:
                parts = next(csv.reader([unquoted]))
            except Exception:
                pass
            if len(parts) >= 2:
                gene = parts[0]
                feature = ",".join(parts[1:])
            else:
                ws_parts = unquoted.split(None, 1)
                if len(ws_parts) == 2:
                    gene, feature = ws_parts[0], ws_parts[1]

        gene = gene.strip().strip('"').strip("'")
        feature = feature.strip().strip('"').strip("'")
        if not gene or not feature:
            continue

        gene_key = normalize_gene_key(gene)
        feature_key = normalize_gene_key(feature)
        if gene_key in {"GENE", "GENE_ID", "WBGENE", "WBGENE_ID", "GENE_NAME"} and feature_key in {
            "FEATURE",
            "ANNOTATION",
            "NOTE",
            "CLASS",
            "LABEL",
        }:
            continue

        existing = annotation_map.get(gene_key, "")
        if existing:
            values = {x.strip() for x in existing.split(";") if x.strip()}
            if feature not in values:
                annotation_map[gene_key] = f"{existing}; {feature}"
        else:
            annotation_map[gene_key] = feature

    if annotation_map:
        print(f"Loaded user annotations from {path} ({len(annotation_map)} genes).")
    else:
        print(f"[Info] No valid gene-feature pairs parsed from {path}.")
    return annotation_map, raw_text


def fetch_wormbase_context(wbgene_id: str) -> Dict[str, str]:
    out = {"WB_Overview": "", "MANUAL_DESCRIPTION_WB": "", "wormbase_gene_name": ""}
    gene_id = (wbgene_id or "").strip()
    if not gene_id:
        return out

    url = f"http://rest.wormbase.org/rest/widget/gene/{gene_id}/overview"
    url_gene = f"http://rest.wormbase.org/rest/field/gene/{gene_id}/legacy_manual_description"

    try:
        response = requests.get(url, headers=WORMBASE_HEADERS, timeout=20)
        if response.ok:
            data = response.json()
            fields = data.get("fields", {}) if isinstance(data, dict) else {}
            concise = ((fields.get("concise_description") or {}).get("data") or {}).get("text")
            gene_name = ((fields.get("name") or {}).get("data") or {}).get("label")
            out["WB_Overview"] = str(concise or "")
            out["wormbase_gene_name"] = str(gene_name or "")
    except Exception:
        pass

    try:
        response_gene = requests.get(url_gene, headers=WORMBASE_HEADERS, timeout=20)
        if response_gene.ok:
            data_gene = response_gene.json()
            manual = ((data_gene.get("legacy_manual_description") or {}).get("data") or {}).get("text")
            out["MANUAL_DESCRIPTION_WB"] = str(manual or "")
    except Exception:
        pass

    return out


def fetch_wormbase_context_safe(wbgene_id: str) -> Tuple[str, Dict[str, str]]:
    try:
        ctx = fetch_wormbase_context(wbgene_id)
    except Exception:
        ctx = {"WB_Overview": "", "MANUAL_DESCRIPTION_WB": "", "wormbase_gene_name": ""}
    return wbgene_id, ctx


def lookup_user_annotation(
    gene_name: str,
    wbgene_id: str,
    wormbase_gene_name: str,
    annotation_map: Dict[str, str],
) -> str:
    for key in [wbgene_id, gene_name, wormbase_gene_name]:
        norm = normalize_gene_key(key)
        if norm and norm in annotation_map:
            return annotation_map[norm]
    return ""


def parse_info_fields(info_value: Any) -> Dict[str, str]:
    out = {"gene": "", "effect": "", "impact": "", "wbgene_id": ""}
    if not isinstance(info_value, str) or not info_value:
        return out

    ann_payload = ""
    m = re.search(r"ANN=([^;]+)", info_value)
    if m:
        ann_payload = m.group(1)
    elif info_value.startswith("ANN="):
        ann_payload = info_value[4:]
    else:
        return out

    # Use first ANN item, matching notebook logic: split(',') then split('|').
    first_ann = ann_payload.split(",")[0]
    cols = first_ann.split("|")
    if len(cols) > 1:
        out["effect"] = cols[1].strip()
    if len(cols) > 2:
        out["impact"] = cols[2].strip()
    if len(cols) > 3:
        out["gene"] = cols[3].strip()
    if len(cols) > 4:
        m_wb_col = re.search(r"WBGene\d{8}", cols[4])
        if m_wb_col:
            out["wbgene_id"] = m_wb_col.group(0)
    if not out["wbgene_id"]:
        m_wb = re.search(r"WBGene\d{8}", first_ann)
        if m_wb:
            out["wbgene_id"] = m_wb.group(0)

    return out


def effect_severity_score(effect_text: str, impact_bucket: str) -> float:
    t = (effect_text or "").lower()
    base = {"HIGH": 3.0, "MODERATE": 2.0, "LOW": 1.0, "UNKNOWN": 0.5}.get(
        (impact_bucket or "UNKNOWN").upper(),
        0.5,
    )
    if any(
        k in t
        for k in [
            "stop_gained",
            "frameshift",
            "splice_acceptor",
            "splice_donor",
            "start_lost",
            "stop_lost",
            "exon_loss",
        ]
    ):
        base = max(base, 3.2)
    elif any(k in t for k in ["missense", "inframe_insertion", "inframe_deletion", "protein_altering"]):
        base = max(base, 2.2)
    elif any(k in t for k in ["synonymous", "upstream", "downstream", "utr", "intron"]):
        base = min(base, 1.0)
    return float(base)


def impact_bucket_from(effect_text: str, impact_bucket: Optional[str]) -> str:
    if isinstance(impact_bucket, str) and impact_bucket:
        ib = impact_bucket.strip().upper()
        if ib in {"HIGH", "MODERATE", "LOW", "MODIFIER"}:
            return ib

    t = (effect_text or "").lower()
    if any(k in t for k in ["stop_gained", "frameshift", "splice_acceptor", "splice_donor", "start_lost", "stop_lost"]):
        return "HIGH"
    if any(k in t for k in ["missense", "inframe_insertion", "inframe_deletion", "protein_altering"]):
        return "MODERATE"
    if any(k in t for k in ["synonymous", "upstream", "downstream", "utr", "intron"]):
        return "LOW"
    return "UNKNOWN"


# =========================
# Load primary CSV
# =========================
exists_or_raise(CSV_PATH)
# Use python engine for more robust handling of CSVs where INFO may be unquoted in some rows
# (e.g. drp1_nonsubstracted_drp1format.csv has commas inside INFO; inconsistent quoting can misalign columns)
try:
    df = pd.read_csv(CSV_PATH, engine="python")
except Exception:
    df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df):,} variants from: {CSV_PATH}")

chrom_col = coalesce(["CHROM", "Chrom", "chr", "chrom", "#CHROM"], df) or "CHROM"
pos_col = coalesce(["POS", "Position", "POS_GRCh", "pos"], df) or "POS"
ref_col = coalesce(["REF", "Ref", "reference_allele", "ref"], df)
alt_col = coalesce(["ALT", "Alt", "alt_allele", "alt"], df)
info_col = coalesce(["INFO", "info"], df)
qual_col = coalesce(["QUAL", "Qual", "quality", "QUALITY"], df)
dp_col = coalesce(["DP", "Depth", "depth"], df)
sift_col = coalesce(["SIFT", "SIFT_pred", "SIFT_PRED"], df)
pphen_col = coalesce(["PolyPhen", "PolyPhen_pred", "PolyPhen_PRED"], df)

if not info_col:
    raise ValueError("No INFO column found in drp1.csv. Expected INFO or info.")

df.columns = [c.strip() for c in df.columns]
df["_CHROM_N"] = df[chrom_col].apply(normalize_chr_label)
df["_POS_I"] = pd.to_numeric(df[pos_col], errors="coerce").astype("Int64")
if ref_col:
    df["_REF_U"] = df[ref_col].astype(str).str.upper()
else:
    df["_REF_U"] = "N"
if alt_col:
    df["_ALT_U"] = df[alt_col].astype(str).str.upper()
else:
    df["_ALT_U"] = "N"

df["variant_id"] = df.apply(lambda r: build_variant_id(r["_CHROM_N"], r["_POS_I"], r["_REF_U"], r["_ALT_U"]), axis=1)
df["pos_key"] = df["_CHROM_N"].astype(str) + ":" + df["_POS_I"].astype(str)

USER_ANNOTATION_FILE = prompt_user_annotation_file(SCRIPT_DIR)
user_annotation_map, annotation_text = load_user_annotations(USER_ANNOTATION_FILE)
use_user_annotation_for_llm = bool(annotation_text.strip())

# =========================
# Load mapping CSV and build lookup maps (when MAPPING_OUTPUT_PATH set)
# =========================
m_parental_by_pos: Dict[str, Any] = {}
m_parental_by_vid: Dict[str, Any] = {}
m_hawaiian_by_pos: Dict[str, Any] = {}
m_hawaiian_by_vid: Dict[str, Any] = {}
m_qual_by_pos: Dict[str, Any] = {}
m_qual_by_vid: Dict[str, Any] = {}
m_dp_by_pos: Dict[str, Any] = {}
m_dp_by_vid: Dict[str, Any] = {}
m_ems_by_pos: Dict[str, Any] = {}
m_ems_by_vid: Dict[str, Any] = {}
m_change_by_pos: Dict[str, Any] = {}
m_change_by_vid: Dict[str, Any] = {}


def _to_map(series_key: pd.Series, series_val: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in zip(series_key, series_val):
        if pd.isna(k):
            continue
        out[str(k)] = v
    return out


def _map_val(vid: str, pkey: str, by_pos: Dict[str, Any], by_vid: Dict[str, Any]) -> Any:
    return by_pos.get(pkey, by_vid.get(vid, None))


if USE_MAPPING:
    mout = pd.read_csv(MAPPING_OUTPUT_PATH)
    mout.columns = [c.strip() for c in mout.columns]
    m_chrom = coalesce(["CHROM", "Chrom", "chr", "chrom"], mout) or "CHROM"
    m_pos = coalesce(["POS", "Position", "POS_GRCh", "pos"], mout) or "POS"
    m_ref = coalesce(["REF", "Ref", "reference_allele", "ref"], mout)
    m_alt = coalesce(["ALT", "Alt", "alt_allele", "alt"], mout)
    mout["_CHROM_N"] = mout[m_chrom].apply(normalize_chr_label)
    mout["_POS_I"] = pd.to_numeric(mout[m_pos], errors="coerce").astype("Int64")
    if m_ref:
        mout["_REF_U"] = mout[m_ref].astype(str).str.upper()
    else:
        mout["_REF_U"] = "N"
    if m_alt:
        mout["_ALT_U"] = mout[m_alt].astype(str).str.upper()
    else:
        mout["_ALT_U"] = "N"
    mout["variant_id"] = mout.apply(
        lambda r: build_variant_id(r["_CHROM_N"], r["_POS_I"], r["_REF_U"], r["_ALT_U"]), axis=1
    )
    mout["pos_key"] = mout["_CHROM_N"].astype(str) + ":" + mout["_POS_I"].astype(str)
    m_parental_by_pos = _to_map(mout["pos_key"], mout.get("Parental_ratio", pd.Series([None] * len(mout))))
    m_parental_by_vid = _to_map(mout["variant_id"], mout.get("Parental_ratio", pd.Series([None] * len(mout))))
    m_hawaiian_by_pos = _to_map(mout["pos_key"], mout.get("Hawaiian_ratio", pd.Series([None] * len(mout))))
    m_hawaiian_by_vid = _to_map(mout["variant_id"], mout.get("Hawaiian_ratio", pd.Series([None] * len(mout))))
    m_qual_by_pos = _to_map(mout["pos_key"], mout.get("QUAL", pd.Series([None] * len(mout))))
    m_qual_by_vid = _to_map(mout["variant_id"], mout.get("QUAL", pd.Series([None] * len(mout))))
    m_dp_by_pos = _to_map(mout["pos_key"], mout.get("DP", pd.Series([None] * len(mout))))
    m_dp_by_vid = _to_map(mout["variant_id"], mout.get("DP", pd.Series([None] * len(mout))))
    m_ems_by_pos = _to_map(mout["pos_key"], mout.get("EMS_label", pd.Series([None] * len(mout))))
    m_ems_by_vid = _to_map(mout["variant_id"], mout.get("EMS_label", pd.Series([None] * len(mout))))
    if "Change_type" in mout.columns:
        m_change_by_pos = _to_map(mout["pos_key"], mout["Change_type"])
        m_change_by_vid = _to_map(mout["variant_id"], mout["Change_type"])
    inter_vid = set(df["variant_id"]).intersection(set(mout["variant_id"]))
    inter_pos = set(df["pos_key"]).intersection(set(mout["pos_key"]))
    print(
        f"[Mapping] Loaded {len(mout):,} rows from {MAPPING_OUTPUT_PATH}; "
        f"variant_id overlap={len(inter_vid)}, pos_key overlap={len(inter_pos)}"
    )

# =========================
# Build records from drp1.csv INFO + other columns
# =========================
packed_all: List[Dict[str, Any]] = []
for _, row in df.iterrows():
    parsed = parse_info_fields(row.get(info_col))

    gene = parsed.get("gene", "")
    effect = parsed.get("effect", "")
    impact_raw = parsed.get("impact", "")
    wbgene_id = parsed.get("wbgene_id", "")
    user_annotation = lookup_user_annotation(gene, wbgene_id, "", user_annotation_map)

    ib = impact_bucket_from(effect, impact_raw)
    qual = coerce_float(row.get(qual_col)) if qual_col else None
    depth = coerce_float(row.get(dp_col)) if dp_col else None

    vid = row["variant_id"]
    pkey = row["pos_key"]
    parental_ratio: Any = None
    hawaiian_ratio: Any = None
    ems_label_raw: Any = ""
    ems = False
    protein_change = ""

    if USE_MAPPING:
        parental_ratio = _map_val(vid, pkey, m_parental_by_pos, m_parental_by_vid)
        hawaiian_ratio = _map_val(vid, pkey, m_hawaiian_by_pos, m_hawaiian_by_vid)
        qual_from_map = _map_val(vid, pkey, m_qual_by_pos, m_qual_by_vid)
        dp_from_map = _map_val(vid, pkey, m_dp_by_pos, m_dp_by_vid)
        if qual_from_map is not None and not (isinstance(qual_from_map, float) and math.isnan(qual_from_map)):
            qual = coerce_float(qual_from_map)
        if dp_from_map is not None and not (isinstance(dp_from_map, float) and math.isnan(dp_from_map)):
            depth = coerce_float(dp_from_map)
        ems_label_raw = _map_val(vid, pkey, m_ems_by_pos, m_ems_by_vid)
        if isinstance(ems_label_raw, str) and ems_label_raw.strip().lower().startswith("ems"):
            ems = True
        change_type = _map_val(vid, pkey, m_change_by_pos, m_change_by_vid)
        if change_type not in (None, "nan", "NaN", ""):
            protein_change = short(str(change_type), 48)

    rec = {
        "variant_id": vid,
        "pos_key": pkey,
        "chrom": row["_CHROM_N"],
        "pos": int(row["_POS_I"]) if pd.notna(row["_POS_I"]) else None,
        "gene": gene,
        "wbgene_id": wbgene_id,
        "effect": effect,
        "impact_bucket": ib,
        "sift": short(row.get(sift_col), 28) if sift_col else "",
        "polyphen": short(row.get(pphen_col), 28) if pphen_col else "",
        "qual": qual,
        "depth": depth,
        "user_annotation": user_annotation if user_annotation else "no annotation",
    }
    if USE_MAPPING:
        rec["parental_ratio"] = parental_ratio
        rec["hawaiian_ratio"] = hawaiian_ratio
        rec["ems_label_raw"] = ems_label_raw if ems_label_raw is not None and not (isinstance(ems_label_raw, float) and math.isnan(ems_label_raw)) else ""
        rec["ems"] = ems
        rec["protein_change"] = protein_change

    imp_w = {"HIGH": 0.45, "MODERATE": 0.25, "LOW": 0.05, "MODIFIER": 0.02, "UNKNOWN": 0.0}.get(rec["impact_bucket"], 0.0)
    qual_w = 0.0 if rec["qual"] is None else min(float(rec["qual"]), 200.0) / 200.0 * 0.05
    rec["_pre_score"] = imp_w + qual_w
    if USE_MAPPING:
        pr = rec.get("parental_ratio")
        pr_val = 0.0 if pr is None or (isinstance(pr, float) and math.isnan(pr)) else float(pr)
        rec["_pre_score"] += pr_val * 0.45
        if PREFER_EMS and rec.get("ems"):
            rec["_pre_score"] += 0.15
    rec["_severity"] = effect_severity_score(rec["effect"], rec["impact_bucket"])
    packed_all.append(rec)


def tie_break_key(r: Dict[str, Any]) -> Tuple[Any, ...]:
    q = r.get("qual") or 0.0
    d = r.get("depth") or 0.0
    if USE_MAPPING:
        pr = r.get("parental_ratio")
        pr_val = 0.0 if pr is None or (isinstance(pr, float) and math.isnan(pr)) else float(pr)
        ems_val = 1.0 if r.get("ems") else 0.0
        return (r["_severity"], pr_val, ems_val, q, d)
    return (r["_severity"], q, d)


packed_dedup: List[Dict[str, Any]] = []
for _, grp in pd.DataFrame(packed_all).groupby("variant_id"):
    best = max(grp.to_dict("records"), key=tie_break_key)
    packed_dedup.append(best)

packed_df = pd.DataFrame(packed_dedup)
n_with_gene = sum(1 for r in packed_dedup if (r.get("gene") or "").strip())
print(
    f"Prepared base records for {len(packed_df):,} variants after dedup by most severe effect "
    f"({n_with_gene:,} with non-empty gene name)."
)
if n_with_gene < 2 and len(packed_dedup) > 2:
    print(
        "[Warning] Most variants have no gene parsed from INFO. If using a CSV with ANN in INFO, "
        "ensure the INFO column is quoted when it contains commas (e.g. drp1_nonsubstracted_drp1format.csv).",
        flush=True,
    )


def _passes_parental_filter(r: Dict[str, Any]) -> bool:
    if not USE_MAPPING:
        return True
    pr = r.get("parental_ratio")
    if pr is None or (isinstance(pr, float) and math.isnan(pr)):
        return True
    try:
        return float(pr) >= FILTER_MIN_PARENTAL_RATIO
    except Exception:
        return True


pool_for_pref = [r for r in packed_dedup if _passes_parental_filter(r)]
if USE_MAPPING and len(pool_for_pref) < len(packed_dedup):
    print(
        f"[Mapping] Filtered to {len(pool_for_pref):,} variants with parental_ratio >= {FILTER_MIN_PARENTAL_RATIO}."
    )


def _gene_key_pref(gene: Any, variant_id: Any) -> str:
    g = str(gene or "").strip().upper()
    if g:
        return g
    return f"__VAR__{str(variant_id or '').strip()}"


gene_best_map: Dict[str, Dict[str, Any]] = {}
for rec in pool_for_pref:
    gkey = _gene_key_pref(rec.get("gene"), rec.get("variant_id"))
    cur = gene_best_map.get(gkey)
    if cur is None or tie_break_key(rec) > tie_break_key(cur):
        gene_best_map[gkey] = rec

pref = sorted(
    gene_best_map.values(),
    key=lambda r: (float(r.get("_pre_score", 0.0) or 0.0), float(r.get("_severity", 0.0) or 0.0)),
    reverse=True,
)
print(f"Will send {len(pref):,} candidate genes to {OPENAI_MODEL}.")

print("Fetching WormBase context...")
unique_wbgene_ids = sorted({(r.get("wbgene_id") or "").strip() for r in pref if r.get("wbgene_id")})
wormbase_context: Dict[str, Dict[str, str]] = {}
if unique_wbgene_ids:
    wormbase_workers = stage_worker_count(len(unique_wbgene_ids))
    print(
        f"[Parallel] WormBase stage: {len(unique_wbgene_ids)} genes with {wormbase_workers} workers.",
        flush=True,
    )
    for gid, wb_ctx in parallel_map_ordered(unique_wbgene_ids, fetch_wormbase_context_safe):
        wormbase_context[gid] = wb_ctx

for rec in packed_dedup:
    gid = (rec.get("wbgene_id") or "").strip()
    wb = wormbase_context.get(gid, {"WB_Overview": "", "MANUAL_DESCRIPTION_WB": "", "wormbase_gene_name": ""})
    rec["WB_Overview"] = wb.get("WB_Overview", "")
    rec["MANUAL_DESCRIPTION_WB"] = wb.get("MANUAL_DESCRIPTION_WB", "")
    rec["wormbase_gene_name"] = wb.get("wormbase_gene_name", "")
    rec["user_annotation"] = lookup_user_annotation(
        rec.get("gene", ""),
        gid,
        rec.get("wormbase_gene_name", ""),
        user_annotation_map,
    ) or "no annotation"

gene_wormbase_context: Dict[str, Dict[str, str]] = {}
for rec in pref:
    gene_name = str(rec.get("gene", "")).strip()
    if not gene_name:
        continue
    current = gene_wormbase_context.get(
        gene_name,
        {"WB_Overview": "", "MANUAL_DESCRIPTION_WB": "", "wormbase_gene_name": ""},
    )
    candidate = {
        "WB_Overview": str(rec.get("WB_Overview", "") or ""),
        "MANUAL_DESCRIPTION_WB": str(rec.get("MANUAL_DESCRIPTION_WB", "") or ""),
        "wormbase_gene_name": str(rec.get("wormbase_gene_name", "") or ""),
    }
    if len(candidate["WB_Overview"]) + len(candidate["MANUAL_DESCRIPTION_WB"]) > len(current["WB_Overview"]) + len(
        current["MANUAL_DESCRIPTION_WB"]
    ):
        gene_wormbase_context[gene_name] = candidate

# =========================
# NCBI gene knowledge + publications
# =========================
GENE_EXTERNAL_INFO_CACHE: Dict[str, Dict[str, Any]] = {}


def fetch_pubmed_abstracts(pubmed_ids: List[str], api_key: str = "") -> Dict[str, str]:
    if not pubmed_ids:
        return {}

    params_fx = {"db": "pubmed", "retmode": "xml", "id": ",".join(pubmed_ids)}
    if api_key:
        params_fx["api_key"] = api_key

    try:
        xml_text = _requests_get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params_fx,
            timeout=20,
        ).text
        root = ET.fromstring(xml_text)
    except Exception:
        return {}

    abstracts: Dict[str, str] = {}
    for art in root.findall(".//PubmedArticle"):
        pmid_node = art.find(".//MedlineCitation/PMID")
        pmid = (pmid_node.text or "").strip() if pmid_node is not None else ""
        if not pmid:
            continue

        parts: List[str] = []
        for node in art.findall(".//Abstract/AbstractText"):
            txt = "".join(node.itertext()).strip()
            if not txt:
                continue
            label = (node.attrib.get("Label") or node.attrib.get("NlmCategory") or "").strip()
            parts.append(f"{label}: {txt}" if label else txt)

        if parts:
            abstracts[pmid] = " ".join(parts)

    return abstracts


def fetch_gene_knowledge(gene: str) -> Dict[str, Any]:
    gene = (gene or "").strip()
    if not gene:
        return {"refs": []}
    cache_key = normalize_gene_key(gene)
    if cache_key in GENE_EXTERNAL_INFO_CACHE:
        return copy.deepcopy(GENE_EXTERNAL_INFO_CACHE[cache_key])

    api_key = os.getenv("NCBI_API_KEY", "")
    refs: List[Dict[str, str]] = []

    # PubMed refs
    try:
        q = f'({gene}) AND (mystery cell of male(MCM) OR neuron) AND elegans'
        params_pm = {
            "db": "pubmed",
            "retmode": "json",
            "retmax": "3",
            "sort": "relevance",
            "term": q,
        }
        if api_key:
            params_pm["api_key"] = api_key

        pm = _requests_get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params_pm,
            timeout=20,
        ).json()
        idlist = [str(x) for x in pm.get("esearchresult", {}).get("idlist", [])[:3]]

        if not idlist:
            q_fb = f'"{gene}"[All Fields] AND (Caenorhabditis elegans[Organism] OR "C. elegans"[All Fields])'
            params_pm["term"] = q_fb
            pm = _requests_get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params_pm,
                timeout=20,
            ).json()
            idlist = [str(x) for x in pm.get("esearchresult", {}).get("idlist", [])[:3]]

        if idlist:
            params_sm = {"db": "pubmed", "retmode": "json", "id": ",".join(idlist)}
            if api_key:
                params_sm["api_key"] = api_key

            sm = _requests_get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params_sm,
                timeout=20,
            ).json()
            abstract_map = fetch_pubmed_abstracts(idlist, api_key=api_key)
            for pid in idlist:
                it = sm.get("result", {}).get(pid, {})
                title = it.get("title", "")
                abstract = abstract_map.get(pid, "")
                if title or abstract:
                    refs.append(
                        {
                            "pmid": pid,
                            "title": title,
                            "abstract": abstract,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                        }
                    )
    except Exception:
        pass

    result = {"refs": refs}
    GENE_EXTERNAL_INFO_CACHE[cache_key] = copy.deepcopy(result)
    return copy.deepcopy(result)


class LiteratureEvaluation(BaseModel):
    mechanistic_match: bool
    relevance_score: conint(ge=1, le=10)
    rationale: str
    extracted_mechanism: str = ""


def evaluate_literature_with_llm(
    gene_name: str,
    literature_text: str,
    Agent1Prompt: str,
) -> LiteratureEvaluation:
    cleaned_literature = (literature_text or "").strip()
    if not cleaned_literature:
        return LiteratureEvaluation(
            mechanistic_match=False,
            relevance_score=1,
            rationale="No literature text provided.",
            extracted_mechanism="",
        )

    system_prompt = """You are Agent1: The Literature Evaluator and Distiller.
Evaluate whether the provided literature text has mechanistic relevance to the requested phenotype and mechanisms.
Return STRICT JSON only with exactly these keys:
- mechanistic_match: boolean
- relevance_score: integer from 1 to 10
- rationale: string, 1-2 sentences grounded in the provided biology
- extracted_mechanism: single compressed sentence; MUST be empty string when relevance_score < 6
Do not include markdown, code fences, or extra keys."""

    user_payload = {
        "gene_name": gene_name,
        "Agent1Prompt": Agent1Prompt,
        "literature_text": cleaned_literature,
        "required_schema": {
            "mechanistic_match": True,
            "relevance_score": 10,
            "rationale": "string",
            "extracted_mechanism": "string",
        },
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
    ]
    raw = openai_chat(
        messages,
        model=OPENAI_MODEL,
        response_json=True,
        temperature=None,
        seed=None,
        timeout=OPENAI_TIMEOUT,
    )
    try:
        parsed = json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise
        parsed = json.loads(m.group(0))

    out = LiteratureEvaluation(**parsed)
    out.rationale = (out.rationale or "").strip()
    out.extracted_mechanism = (out.extracted_mechanism or "").strip()
    if out.relevance_score < 6:
        out.extracted_mechanism = ""
    return out


def filter_and_distill_gene_literature(
    gene_name: str,
    list_of_abstracts: List[Any],
    phenotype_description: str,
    target_mechanisms: str,
) -> Dict[str, Any]:
    agent1_prompt = (
        f"{AGENT1_PROMPT}\n"
        f"Phenotype: {phenotype_description}\n"
        f"Target mechanisms: {target_mechanisms}\n"
        "Score only mechanistic relevance to this phenotype."
    )

    extracted_mechanisms: List[str] = []
    kept_pubmed_ids: List[str] = []
    kept_refs: List[Dict[str, str]] = []
    pubmed_agent1_comments: List[str] = []

    total_refs = len(list_of_abstracts or [])
    for ref_idx, item in enumerate(list_of_abstracts or [], start=1):
        print(
            f"[Agent1] {gene_name}: evaluating reference {ref_idx}/{total_refs}...",
            flush=True,
        )
        pmid = ""
        title = ""
        abstract = ""
        if isinstance(item, dict):
            pmid = str(item.get("pmid", "")).strip()
            title = str(item.get("title", "")).strip()
            abstract = str(item.get("abstract", "")).strip()
            literature_text = f"Title: {title}\nAbstract: {abstract}".strip()
        else:
            literature_text = str(item or "").strip()

        if not literature_text:
            continue

        try:
            evaluation = evaluate_literature_with_llm(gene_name, literature_text, agent1_prompt)
        except Exception as exc:
            print(
                f"[Agent1] {gene_name}: reference {ref_idx}/{total_refs} skipped ({exc.__class__.__name__}).",
                flush=True,
            )
            continue

        # Keep per-publication Agent1 decision text for output table.
        if pmid:
            pub_label = f"PMID{pmid}"
        elif title:
            pub_label = short(title, 60)
        else:
            pub_label = f"REF{ref_idx}"
        relevance_label = "Relevant" if int(evaluation.relevance_score) >= 6 else "Irrelevant"
        reason_txt = re.sub(r"\s+", " ", (evaluation.rationale or "").strip())
        pubmed_agent1_comments.append(
            f"{pub_label}: {relevance_label} - score:{int(evaluation.relevance_score)} Reason:{reason_txt}"
        )

        if int(evaluation.relevance_score) < 6:
            continue

        mechanism = (evaluation.extracted_mechanism or "").strip()
        if mechanism:
            extracted_mechanisms.append(mechanism)
        if pmid:
            kept_pubmed_ids.append(pmid)
        if isinstance(item, dict):
            kept_refs.append(item)

    dedup_mechanisms: List[str] = []
    seen_mech = set()
    for mechanism in extracted_mechanisms:
        if mechanism not in seen_mech:
            dedup_mechanisms.append(mechanism)
            seen_mech.add(mechanism)

    dedup_pubmed_ids: List[str] = []
    seen_ids = set()
    for pmid in kept_pubmed_ids:
        if pmid not in seen_ids:
            dedup_pubmed_ids.append(pmid)
            seen_ids.add(pmid)

    dedup_refs: List[Dict[str, str]] = []
    seen_ref_ids = set()
    for ref in kept_refs:
        ref_id = str(ref.get("pmid", "")).strip()
        if ref_id and ref_id in seen_ref_ids:
            continue
        if ref_id:
            seen_ref_ids.add(ref_id)
        dedup_refs.append(ref)

    print(
        f"[Agent1] {gene_name}: kept {len(dedup_pubmed_ids)} relevant references.",
        flush=True,
    )
    return {
        "distilled_mechanism": " ".join(dedup_mechanisms),
        "filtered_pubmed_ids": dedup_pubmed_ids,
        "filtered_refs": dedup_refs,
        "agent1_pubmed_comments": pubmed_agent1_comments,
    }


def build_gene_knowledge_entry(
    task: Tuple[str, Dict[str, str], str, str],
) -> Tuple[str, Dict[str, Any], int, int]:
    gene_name, wb_ctx, phenotype_description, target_mechanisms = task
    print(f"[NCBI] {gene_name}: fetching references...", flush=True)
    try:
        gene_info = fetch_gene_knowledge(gene_name)
    except Exception:
        gene_info = {"refs": []}

    wb = wb_ctx or {"WB_Overview": "", "MANUAL_DESCRIPTION_WB": "", "wormbase_gene_name": ""}
    gene_info["WB_Overview"] = wb.get("WB_Overview", "")
    gene_info["MANUAL_DESCRIPTION_WB"] = wb.get("MANUAL_DESCRIPTION_WB", "")
    gene_info["wormbase_gene_name"] = wb.get("wormbase_gene_name", "")

    refs = gene_info.get("refs", [])
    distill = filter_and_distill_gene_literature(
        gene_name,
        refs,
        phenotype_description=phenotype_description,
        target_mechanisms=target_mechanisms,
    )
    gene_info["filtered_pubmed_ids"] = list(distill.get("filtered_pubmed_ids", []))
    gene_info["filtered_refs"] = list(distill.get("filtered_refs", []))
    gene_info["agent1_pubmed_comments"] = list(distill.get("agent1_pubmed_comments", []))
    gene_info["distilled_mechanism"] = str(distill.get("distilled_mechanism", "") or "")
    return gene_name, gene_info, len(refs), len(gene_info["filtered_pubmed_ids"])

AGENT1_PROMPT = _input(
    "Agent1Prompt (define phenotype + possible mechanisms causing that phenotype): "
).strip()
if not AGENT1_PROMPT:
    AGENT1_PROMPT = (
        "Phenotype: loss of mystery cell of male(MCM) neurons. "
        "Mechanisms: disrupted cell-fate specification, neuronal differentiation, "
        "axon guidance, survival, or reporter expression."
    )

phenotype_description = (
    _input("Phenotype description for Agent1 filtering (press Enter to reuse Agent1Prompt): ").strip()
    or AGENT1_PROMPT
)
target_mechanisms = (
    _input("Target mechanisms for Agent1 filtering (press Enter to reuse Agent1Prompt): ").strip()
    or AGENT1_PROMPT
)

unique_genes = sorted({(r.get("gene") or "").strip() for r in pref if r.get("gene")})
gene_knowledge: Dict[str, Dict[str, Any]] = {}
print("Fetching gene publications from NCBI...")
total_genes = len(unique_genes)
gene_tasks: List[Tuple[str, Dict[str, str], str, str]] = []
for gene_idx, g in enumerate(unique_genes, start=1):
    print(f"[NCBI] Queue gene {gene_idx}/{total_genes}: {g}", flush=True)
    wb_ctx = gene_wormbase_context.get(
        g,
        {"WB_Overview": "", "MANUAL_DESCRIPTION_WB": "", "wormbase_gene_name": ""},
    )
    gene_tasks.append((g, wb_ctx, phenotype_description, target_mechanisms))

if gene_tasks:
    gene_workers = stage_worker_count(len(gene_tasks))
    print(
        f"[Parallel] NCBI+Agent1 stage: {len(gene_tasks)} genes with {gene_workers} workers.",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=gene_workers) as executor:
        future_to_gene = {
            executor.submit(build_gene_knowledge_entry, task): task[0]
            for task in gene_tasks
        }
        completed = 0
        for future in as_completed(future_to_gene):
            gene_name = future_to_gene[future]
            info_gene, info_data, fetched_count, kept_count = future.result()
            gene_knowledge[info_gene] = info_data
            completed += 1
            print(
                f"[NCBI] Completed {completed}/{total_genes}: {gene_name} "
                f"(fetched {fetched_count}, kept {kept_count}).",
                flush=True,
            )

# =========================
# LLM schemas
# =========================
class VariantRankItem(BaseModel):
    rank: conint(ge=1)
    variant_id: str
    gene: Optional[str] = ""
    causal_probability: confloat(ge=0, le=1)
    confidence: confloat(ge=0, le=1)
    rationale: str
    key_evidence: List[str] = Field(default_factory=list)


class VariantNarrative(BaseModel):
    variant_id: str
    gene: Optional[str] = ""
    narrative: str
    narrative_confidence: confloat(ge=0, le=1)


class LLMVariantOutput(BaseModel):
    summary: str
    most_likely: VariantRankItem
    ranking: List[VariantRankItem]
    annotations: List[VariantNarrative]

    @field_validator("ranking")
    @classmethod
    def non_empty_rank(cls, v):
        if not v:
            raise ValueError("ranking must be non-empty")
        return v

    @field_validator("annotations")
    @classmethod
    def non_empty_ann(cls, v):
        if not v:
            raise ValueError("annotations must be non-empty")
        return v


class LLMRankingOnly(BaseModel):
    summary: str
    most_likely: VariantRankItem
    ranking: List[VariantRankItem]


def try_parse_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            return json.loads(m.group(0))
        raise


# =========================
# Prompts
# =========================
SYSTEM_PROMPT_MAIN = """You are an expert in C. elegans forward genetics and mystery cell of male(MCM) neuron fate.

Context:
- Screen detects loss of mystery cell of male(MCM) neurons via a GFP reporter.
- Goal: identify variants most likely to cause MCM loss.

Evidence you may use:
- Structured fields from the csv file: Gene_name, Effect, impact_bucket,  QUAL, DP, SIFT/PolyPhen.
- NCBI publications: provided PubMed links.
- WormBase context when available: WB_Overview and MANUAL_DESCRIPTION_WB.
- Agent1 distilled mechanisms per gene when available (gene_knowledge.distilled_mechanism).
""" + (
    "\n- Mapping data when provided: Parental_ratio (linkage proxy), EMS_label (EMS mutagen), protein_change/Change_type; higher Parental_ratio increases likelihood; EMS increases prior.\n"
    if USE_MAPPING
    else ""
) + """
Instructions:
- The variants list has exactly one representative variant per gene (already deduplicated). Your ranking must list each gene at most once: return exactly ranking_size entries, each with a different gene (use the variant_id provided for that gene).
- Rank by causal probability (0-1) and provide confidence (0-1).
- For each ranked variant, set:
  - rationale: a single concise sentence summarizing why this gene/variant is placed at its rank, based on the available evidence; do NOT use square brackets or inline evidence lists here.
  - key_evidence: short bullet strings capturing the specific supporting facts (effect/impact, QUAL/DP, distilled_mechanism, WormBase context, PubMed refs, etc.).
- Separately, for each variant in the annotations list, write a 2-3 sentence narrative connecting mutation and gene function to phenotype.
- In the narrative, cite evidence in brackets, e.g., [Effect=stop_gained; impact=HIGH; QUAL=83.1; GeneRef: PMID 12345].
- When a distilled mechanism is available for the gene, also include it explicitly in the bracketed evidence as distilled_mechanism="...".
- When WB_Overview or MANUAL_DESCRIPTION_WB text is available, explicitly cite it in the bracketed evidence as WB_Overview="..." and/or MANUAL_DESCRIPTION_WB="..." (shortened as needed).
- Prefer HIGH-impact (nonsense/frameshift/essential splice) > damaging missense > low/unknown.
- Do not fabricate references; only cite provided PubMed links.
- Output JSON only; no chain-of-thought."""


def _safe_round(x: Any, nd: int = 3) -> Optional[float]:
    try:
        return round(float(x), nd)
    except Exception:
        return None


def format_variant_for_llm(r: Dict[str, Any]) -> Dict[str, Any]:
    g = (r.get("gene") or "").strip()
    gk = gene_knowledge.get(g, {"refs": []})
    wb_overview_src = r.get("WB_Overview", "") or gk.get("WB_Overview", "") or ""
    manual_src = r.get("MANUAL_DESCRIPTION_WB", "") or gk.get("MANUAL_DESCRIPTION_WB", "") or ""
    manual_short = short(manual_src, 480)
    # Trimming policy for LLM payload:
    # - If a manual WormBase description exists, send only MANUAL_DESCRIPTION_WB to the LLM
    #   and omit WB_Overview to save tokens.
    # - If manual is absent, fall back to WB_Overview.
    if manual_short:
        wb_short_for_llm = ""
    else:
        wb_short_for_llm = short(wb_overview_src, 480)

    out = {
        "variant_id": r["variant_id"],
        "Gene_name": g,
        "WBGene_ID": r.get("wbgene_id", ""),
        "Effect": short(r.get("effect"), 90),
        "impact_bucket": r.get("impact_bucket", "UNKNOWN"),
        "QUAL": r.get("qual", None),
        "DP": r.get("depth", None),
        "SIFT": r.get("sift", ""),
        "PolyPhen": r.get("polyphen", ""),
        "_pre_score": round(float(r.get("_pre_score", 0.0)), 3),
        "WB_Overview": wb_short_for_llm,
        "MANUAL_DESCRIPTION_WB": manual_short,
        "gene_knowledge": {
            "distilled_mechanism": short(gk.get("distilled_mechanism", ""), 480),
            "WB_Overview": short(gk.get("WB_Overview", ""), 480),
            "MANUAL_DESCRIPTION_WB": short(gk.get("MANUAL_DESCRIPTION_WB", ""), 480),
        },
    }
    if USE_MAPPING:
        out["Parental_ratio"] = _safe_round(r.get("parental_ratio"), 3)
        out["ems_label"] = "EMS" if r.get("ems") else "non-EMS"
        out["protein_change"] = short(r.get("protein_change", ""), 48)
    if use_user_annotation_for_llm:
        out["user_annotation"] = r.get("user_annotation", "no annotation")
    return out


# pref is already one representative variant per gene (from gene_best_map); no duplicate genes sent to LLM
variants_for_llm = [format_variant_for_llm(r) for r in pref]


def format_pubmed_refs(pubmed_refs: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for ref in pubmed_refs or []:
        pmid = str(ref.get("pmid", "")).strip()
        title = str(ref.get("title", "")).strip()
        url = str(ref.get("url", "")).strip()
        if pmid and title:
            parts.append(f"PMID {pmid}: {title} ({url})" if url else f"PMID {pmid}: {title}")
        elif pmid:
            parts.append(f"PMID {pmid} ({url})" if url else f"PMID {pmid}")
        elif title:
            parts.append(f"{title} ({url})" if url else title)
        elif url:
            parts.append(url)
    return " ; ".join(parts)


variant_pubmed_ref_avaiable_map = {
    v["variant_id"]: format_pubmed_refs(gene_knowledge.get(v.get("Gene_name", ""), {}).get("refs", []))
    for v in variants_for_llm
}

variant_filtered_pubmed_map = {
    v["variant_id"]: "; ".join(gene_knowledge.get(v.get("Gene_name", ""), {}).get("filtered_pubmed_ids", []))
    for v in variants_for_llm
}

variant_agent1_publication_comments_map = {
    v["variant_id"]: " ; ".join(
        gene_knowledge.get(v.get("Gene_name", ""), {}).get("agent1_pubmed_comments", [])
    )
    for v in variants_for_llm
}

USER_PROMPT_MAIN = {
    "task": "Prioritize variants for mystery cell of male(MCM) neuron loss and annotate each with a narrative + confidence.",
    "instructions": {
        "ranking_size": min(TOP_K, len(variants_for_llm)),
        "ranking_unit": "genes",
        "must_return_exact_ranking_size": True,
        "return_all_ranked": True,
        "notes": [
            "Use drp1.csv structured fields and provided NCBI PubMed refs.",
            "Use WormBase context (WB_Overview/MANUAL_DESCRIPTION_WB) when available.",
            "Narratives 2-3 sentences with bracketed evidence and PMID citations when available.",
            "Input is one variant per gene; return exactly ranking_size entries in ranking, one entry per gene (no duplicate genes).",
        ],
    },
    "schema": {
        "summary": "string",
        "most_likely": {
            "rank": 1,
            "variant_id": "string",
            "gene": "string",
            "causal_probability": 0.00,
            "confidence": 0.00,
            "rationale": "string",
            "key_evidence": ["short bullet strings"],
        },
        "ranking": [
            {
                "rank": 1,
                "variant_id": "string",
                "gene": "string",
                "causal_probability": 0.00,
                "confidence": 0.00,
                "rationale": "string",
                "key_evidence": ["short bullet strings"],
            }
        ],
        "annotations": [
            {
                "variant_id": "string",
                "gene": "string",
                "narrative": "string",
                "narrative_confidence": 0.00,
            }
        ],
    },
    "variants": variants_for_llm,
}
if use_user_annotation_for_llm:
    USER_PROMPT_MAIN["user_annotation"] = annotation_text
user_prompt = _input("PROMPT: ").strip()
USER_PROMPT_MAIN["instructions"]["notes"].append(user_prompt)


def prompt_run_count() -> int:
    raw = _input("How many LLM runs do you want on this gene set? ").strip()
    try:
        n = int(raw)
        if n < 1:
            raise ValueError
        return n
    except Exception:
        print("[Info] Invalid run count. Using 1 run.")
        return 1


NUM_LLM_RUNS = prompt_run_count()
shot_dt = datetime.now()
SHOT_TIMESTAMP = shot_dt.strftime("%Y_%m_%d_%H_%M")

# =========================
# LLM calls
# =========================
def single_shot_call() -> Optional[LLMVariantOutput]:
    temp_val = float(OPENAI_TEMPERATURE) if OPENAI_TEMPERATURE.strip() else None
    seed_val = int(OPENAI_SEED) if OPENAI_SEED.strip() else None
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_MAIN},
        {"role": "user", "content": json.dumps(USER_PROMPT_MAIN, ensure_ascii=False, separators=(",", ":"))},
    ]
    print(f"Calling {OPENAI_MODEL} with {len(variants_for_llm)} variants (single-shot)...")
    raw = openai_chat(
        messages,
        model=OPENAI_MODEL,
        response_json=True,
        temperature=temp_val,
        seed=seed_val,
        timeout=OPENAI_TIMEOUT,
    )
    parsed = try_parse_json(raw)
    return LLMVariantOutput(**parsed)


NARR_SYSTEM = """You are an expert in C. elegans mystery cell of male(MCM) neuron biology.
For each variant, write a 2-3 sentence narrative using provided fields, NCBI refs, and WormBase context if present.
Cite evidence in brackets, e.g., [Effect=missense; impact=MODERATE; GeneRef: PMID 12345]. When WB_Overview or MANUAL_DESCRIPTION_WB text is available, explicitly cite it in the bracketed evidence as WB_Overview="..." and/or MANUAL_DESCRIPTION_WB="..." (shortened as needed).
Return JSON: {"annotations":[{"variant_id":"string","gene":"string","narrative":"string","narrative_confidence":0.0}]}"""


def llm_narrative_chunk(chunk: List[Dict[str, Any]]) -> Dict[str, Tuple[str, float]]:
    user = {
        "variants": chunk,
        "schema": {
            "annotations": [
                {
                    "variant_id": "string",
                    "gene": "string",
                    "narrative": "string",
                    "narrative_confidence": 0.00,
                }
            ]
        },
    }
    messages = [
        {"role": "system", "content": NARR_SYSTEM},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, separators=(",", ":"))},
    ]
    raw = openai_chat(messages, model=OPENAI_MODEL, response_json=True, temperature=None, seed=None, timeout=OPENAI_TIMEOUT)
    parsed = try_parse_json(raw)
    anns = parsed.get("annotations", [])

    chunk_out: Dict[str, Tuple[str, float]] = {}
    for a in anns:
        try:
            va = VariantNarrative(**a)
            chunk_out[va.variant_id] = (va.narrative, float(va.narrative_confidence))
        except ValidationError:
            pass
    return chunk_out


def llm_narratives_in_chunks(items: List[Dict[str, Any]], chunk_size: int = 18) -> Dict[str, Tuple[str, float]]:
    out: Dict[str, Tuple[str, float]] = {}
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
    if not chunks:
        return out

    chunk_workers = stage_worker_count(len(chunks))
    print(
        f"[Parallel] Narrative stage: {len(chunks)} chunks with {chunk_workers} workers.",
        flush=True,
    )
    for chunk_result in parallel_map_ordered(chunks, llm_narrative_chunk):
        out.update(chunk_result)
    return out


RANK_SYSTEM = """Rank genes for causing mystery cell of male(MCM) neuron loss using provided summaries.
The variant_summaries list is already one entry per gene (deduplicated). Return exactly top_k ranking entries, each with a different gene; use the variant_id from the summary for that gene.
Heuristics: HIGH-impact > damaging missense > low/unknown; integrate NCBI refs, WormBase context (WB_Overview and MANUAL_DESCRIPTION_WB), and narrative confidence.
Within each ranking item, set rationale to a single concise sentence summarizing why this gene/variant is ranked at that position, without square brackets or inline evidence lists; place detailed evidence and citations into the key_evidence bullet strings instead.
Output JSON only with fields: summary, most_likely, ranking[]."""


def llm_ranking_from_summaries(
    summaries: List[Dict[str, Any]],
    required_top_k: Optional[int] = None,
    retry_attempt: int = 1,
    retry_max_attempts: int = 1,
) -> LLMRankingOnly:
    top_k = min(required_top_k or TOP_K, len(summaries))
    user = {
        "task": "Rank all candidate genes by likelihood of causing mystery cell of male(MCM) neuron loss. Input is one summary per gene (already deduplicated). Return exactly top_k ranking entries, one per gene, using the variant_id from each summary.",
        "top_k": top_k,
        "ranking_unit": "genes",
        "must_return_exact_top_k": True,
        "no_duplicate_genes": True,
        "retry_attempt": retry_attempt,
        "retry_max_attempts": retry_max_attempts,
        "variant_summaries": summaries,
        "schema": {
            "summary": "string",
            "most_likely": {
                "rank": 1,
                "variant_id": "string",
                "gene": "string",
                "causal_probability": 0.00,
                "confidence": 0.00,
                "rationale": "string",
                "key_evidence": ["short strings"],
            },
            "ranking": [
                {
                    "rank": 1,
                    "variant_id": "string",
                    "gene": "string",
                    "causal_probability": 0.00,
                    "confidence": 0.00,
                    "rationale": "string",
                    "key_evidence": ["short strings"],
                }
            ],
        },
    }
    messages = [
        {"role": "system", "content": RANK_SYSTEM},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, separators=(",", ":"))},
    ]
    raw = openai_chat(messages, model=OPENAI_MODEL, response_json=True, temperature=None, seed=None, timeout=OPENAI_TIMEOUT)
    parsed = try_parse_json(raw)
    return LLMRankingOnly(**parsed)


def chunked_fallback_pipeline() -> Tuple[pd.DataFrame, Dict[str, Tuple[str, float]]]:
    print("Falling back to chunked pipeline: generating narratives in chunks...")
    narr_map = llm_narratives_in_chunks(variants_for_llm, chunk_size=18)
    print("Chunked pipeline: ranking from compact summaries...")
    summaries = build_variant_summaries(narr_map)
    rank_df = rerank_exact_top_k_from_summaries(
        summaries=summaries,
        narr_map=narr_map,
        candidate_by_variant=candidate_by_variant,
        top_k=TOP_K,
    )
    return rank_df.sort_values(["rank", "variant_id"]), narr_map


# =========================
# Execute
# =========================
def enforce_probability_ranking(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    out["_orig_rank"] = pd.to_numeric(out.get("rank"), errors="coerce")
    out["_causal_probability"] = pd.to_numeric(out.get("causal_probability"), errors="coerce")
    out["_confidence"] = pd.to_numeric(out.get("confidence"), errors="coerce")

    out = out.sort_values(
        ["_causal_probability", "_confidence", "_orig_rank", "variant_id"],
        ascending=[False, False, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)

    out["rank"] = range(1, len(out) + 1)
    return out.drop(columns=["_orig_rank", "_causal_probability", "_confidence"])


def _gene_key(gene: Any, variant_id: Any) -> str:
    g = str(gene or "").strip().upper()
    if g:
        return g
    return f"__VAR__{str(variant_id or '').strip()}"


def build_variant_summaries(narr_map: Dict[str, Tuple[str, float]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for v in variants_for_llm:
        vid = v["variant_id"]
        narrative, narr_conf = narr_map.get(vid, ("", 0.0))
        summary_item = {
            "variant_id": vid,
            "gene": v.get("Gene_name", ""),
            "WBGene_ID": v.get("WBGene_ID", ""),
            "Effect": v.get("Effect", ""),
            "impact_bucket": v.get("impact_bucket", "UNKNOWN"),
            "QUAL": v.get("QUAL", None),
            "DP": v.get("DP", None),
            "WB_Overview": v.get("WB_Overview", ""),
            "MANUAL_DESCRIPTION_WB": v.get("MANUAL_DESCRIPTION_WB", ""),
            "gene_knowledge": v.get("gene_knowledge", {}),
            "narrative": narrative,
            "narrative_confidence": narr_conf,
            "_pre_score": v.get("_pre_score", 0.0),
        }
        if "user_annotation" in v:
            summary_item["user_annotation"] = v.get("user_annotation", "no annotation")
        summaries.append(summary_item)
    return summaries


def dedupe_rank_rows_by_gene(
    rank_df: pd.DataFrame,
    candidate_by_variant: Dict[str, Dict[str, Any]],
    narr_map: Dict[str, Tuple[str, float]],
) -> pd.DataFrame:
    if rank_df is None or rank_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    seen_gene_keys = set()

    ordered = rank_df.sort_values(["rank", "variant_id"], kind="mergesort").to_dict("records")
    for row in ordered:
        vid = str(row.get("variant_id", "")).strip()
        if not vid or vid not in candidate_by_variant:
            continue
        rec = candidate_by_variant.get(vid, {})
        gene = str(row.get("gene", "")).strip() or str(rec.get("gene", "")).strip()
        gkey = _gene_key(gene, vid)
        if gkey in seen_gene_keys:
            continue
        seen_gene_keys.add(gkey)

        row["gene"] = gene
        if not row.get("Narrative"):
            narrative, narr_conf = narr_map.get(vid, ("", 0.0))
            row["Narrative"] = narrative
            row["Narrative_confidence"] = round(float(narr_conf), 2) if narrative else None
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame()
    out["rank"] = range(1, len(out) + 1)
    return out


def rerank_exact_top_k_from_summaries(
    summaries: List[Dict[str, Any]],
    narr_map: Dict[str, Tuple[str, float]],
    candidate_by_variant: Dict[str, Dict[str, Any]],
    top_k: int,
    max_attempts: int = 2,
) -> pd.DataFrame:
    """Rank from summaries (one per gene). Returns up to top_k rows; uses best result across attempts if target not met."""
    target = min(top_k, len(summaries))
    if target <= 0:
        return pd.DataFrame()

    best_dedup_df: Optional[pd.DataFrame] = None
    for attempt in range(1, max_attempts + 1):
        ranking_only = llm_ranking_from_summaries(
            summaries,
            required_top_k=target,
            retry_attempt=attempt,
            retry_max_attempts=max_attempts,
        )
        rank_rows = []
        for it in sorted(ranking_only.ranking, key=lambda x: x.rank):
            rank_rows.append(
                {
                    "run_id": RUN_ID,
                    "variant_id": it.variant_id,
                    "gene": it.gene,
                    "rank": int(it.rank),
                    "causal_probability": round(float(it.causal_probability), 2),
                    "confidence": round(float(it.confidence), 2),
                    "rationale": it.rationale,
                    "key_evidence": "; ".join(it.key_evidence) if it.key_evidence else "",
                    "Narrative": narr_map.get(it.variant_id, ("", None))[0],
                    "Narrative_confidence": round(float(narr_map.get(it.variant_id, ("", 0.0))[1]), 2)
                    if it.variant_id in narr_map
                    else None,
                }
            )

        dedup_df = dedupe_rank_rows_by_gene(pd.DataFrame(rank_rows), candidate_by_variant, narr_map)
        if len(dedup_df) >= target:
            out = dedup_df.iloc[:target].copy()
            out["rank"] = range(1, len(out) + 1)
            return out

        if best_dedup_df is None or len(dedup_df) > len(best_dedup_df):
            best_dedup_df = dedup_df
        print(
            f"[Info] Ranking attempt {attempt}/{max_attempts} returned {len(dedup_df)} unique genes "
            f"(target {target}); using best so far.",
            flush=True,
        )

    if best_dedup_df is not None and len(best_dedup_df) > 0:
        out = best_dedup_df.iloc[:target].copy()
        out["rank"] = range(1, len(out) + 1)
        return out
    return pd.DataFrame()


orig_small = pd.DataFrame(packed_dedup).copy()
orig_small_cols = [
    "variant_id",
    "gene",
    "effect",
    "impact_bucket",
    "qual",
    "depth",
    "WB_Overview",
    "MANUAL_DESCRIPTION_WB",
]
if USE_MAPPING:
        orig_small_cols.extend(["parental_ratio", "ems_label_raw", "protein_change"])
orig_small_ren = orig_small[
    [c for c in orig_small_cols if c in orig_small.columns]
].rename(columns={"gene": "Gene_name_csv"})
user_annotation_map_by_variant = {r["variant_id"]: r.get("user_annotation", "no annotation") for r in packed_dedup}
candidate_by_variant = {r["variant_id"]: r for r in pref}
written_paths: List[pathlib.Path] = []
gene_ranks_per_run: List[Dict[str, int]] = []
all_genes_across_runs: set[str] = set()
gene_run_details_per_run: List[Dict[str, Dict[str, str]]] = []


def _rankdata(values: List[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    sorted_idx = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        v = values[sorted_idx[i]]
        while j + 1 < n and values[sorted_idx[j + 1]] == v:
            j += 1
        rank_start = i + 1
        rank_end = j + 1
        avg_rank = (rank_start + rank_end) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson_corr(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n == 0 or n != len(y):
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mean_x
        dy = yi - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    if den_x <= 0.0 or den_y <= 0.0:
        return 0.0
    return num / math.sqrt(den_x * den_y)


def _spearman_corr_from_ranks(x: List[float], y: List[float]) -> float:
    if not x or not y or len(x) != len(y):
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson_corr(rx, ry)

for shot_idx in range(1, NUM_LLM_RUNS + 1):
    print(f"\n=== LLM shot {shot_idx}/{NUM_LLM_RUNS} ===")
    rank_df: Optional[pd.DataFrame] = None
    narr_map: Dict[str, Tuple[str, float]] = {}

    try:
        llm_out = single_shot_call()
        narr_map = {a.variant_id: (a.narrative, float(a.narrative_confidence)) for a in llm_out.annotations}
        ranked = sorted(llm_out.ranking, key=lambda x: x.rank)
        if TOP_K and len(ranked) > TOP_K:
            ranked = ranked[:TOP_K]

        rows = []
        for it in ranked:
            rows.append(
                {
                    "run_id": RUN_ID,
                    "variant_id": it.variant_id,
                    "gene": it.gene,
                    "rank": int(it.rank),
                    "causal_probability": round(float(it.causal_probability), 2),
                    "confidence": round(float(it.confidence), 2),
                    "rationale": it.rationale,
                    "key_evidence": "; ".join(it.key_evidence) if it.key_evidence else "",
                    "Narrative": narr_map.get(it.variant_id, ("", None))[0],
                    "Narrative_confidence": round(narr_map.get(it.variant_id, ("", 0.0))[1], 2)
                    if it.variant_id in narr_map
                    else None,
                }
            )
        rank_df = pd.DataFrame(rows).sort_values(["rank", "variant_id"])
    except Exception as e:
        print(f"[Single-shot failed: {e.__class__.__name__}] {e}\n")
        rank_df, narr_map = chunked_fallback_pipeline()

    rank_df = dedupe_rank_rows_by_gene(rank_df, candidate_by_variant, narr_map)
    required_k = min(TOP_K, len(variants_for_llm))
    if len(rank_df) < required_k:
        print(
            f"[Info] Initial ranking returned {len(rank_df)} unique genes (target {required_k}). "
            f"Reranking from one-per-gene summaries to get top {required_k}.",
            flush=True,
        )
        summaries = build_variant_summaries(narr_map)
        rank_df = rerank_exact_top_k_from_summaries(
            summaries=summaries,
            narr_map=narr_map,
            candidate_by_variant=candidate_by_variant,
            top_k=required_k,
        )

    rank_df = enforce_probability_ranking(rank_df)
    rank_df["user_annotation"] = rank_df["variant_id"].map(user_annotation_map_by_variant).fillna("no annotation")
    rank_df["pubmed_ref_avaiable"] = rank_df["variant_id"].map(variant_pubmed_ref_avaiable_map).fillna("")
    rank_df["filtered_pubmed"] = rank_df["variant_id"].map(variant_filtered_pubmed_map).fillna("")
    rank_df["agent1_publication_comments"] = rank_df["variant_id"].map(
        variant_agent1_publication_comments_map
    ).fillna("")

    final_df = rank_df.merge(orig_small_ren, how="left", on="variant_id").sort_values(["rank", "variant_id"])

    view_cols = [
        "rank",
        "causal_probability",
        "confidence",
        "variant_id",
        "gene",
        "Gene_name_csv",
        "effect",
        "impact_bucket",
        "qual",
        "depth",
        "WB_Overview",
        "MANUAL_DESCRIPTION_WB",
        "user_annotation",
        "pubmed_ref_avaiable",
        "filtered_pubmed",
        "agent1_publication_comments",
        "Narrative",
        "Narrative_confidence",
        "rationale",
    ]
    if USE_MAPPING:
        idx_after_depth = view_cols.index("depth") + 1
        view_cols = (
            view_cols[:idx_after_depth]
            + ["parental_ratio", "ems_label_raw", "protein_change"]
            + view_cols[idx_after_depth:]
        )
    final_df = final_df[[c for c in view_cols if c in final_df.columns]]

    run_gene_ranks: Dict[str, int] = {}
    for _, row in final_df.iterrows():
        gene_name = str(row.get("gene") or "").strip()
        if not gene_name:
            continue
        rank_value = int(row.get("rank"))
        run_gene_ranks[gene_name] = rank_value
        all_genes_across_runs.add(gene_name)
    gene_ranks_per_run.append(run_gene_ranks)
    run_gene_details: Dict[str, Dict[str, str]] = {}
    for _, row in final_df.iterrows():
        gene_name = str(row.get("gene") or "").strip()
        if not gene_name:
            continue
        run_gene_details[gene_name] = {
            "Narrative": str(row.get("Narrative") or ""),
            "Rationale": str(row.get("rationale") or ""),
            "impact_bucket": str(row.get("impact_bucket") or ""),
            "WB_Overview": str(row.get("WB_Overview") or ""),
            "MANUAL_DESCRIPTION_WB": str(row.get("MANUAL_DESCRIPTION_WB") or ""),
            "effect": str(row.get("effect") or ""),
            "user_annotation": str(row.get("user_annotation") or ""),
            "pubmed_ref_avaiable": str(row.get("pubmed_ref_avaiable") or ""),
            "filtered_pubmed": str(row.get("filtered_pubmed") or ""),
            "agent1_publication_comments": str(row.get("agent1_publication_comments") or ""),
        }
    gene_run_details_per_run.append(run_gene_details)

    print("\n=== Top ranked genes (representative variants, preview) ===")
    _tab = tabulate(final_df.head(20).fillna(""), headers="keys", tablefmt="github", showindex=False)
    try:
        print(_tab)
    except UnicodeEncodeError:
        print("[Preview skipped (encoding); see CSV for full output]")

    if not rank_df.empty:
        top_row = rank_df.sort_values("rank").iloc[0]
        print("\n=== Most likely causal gene (representative variant, LLM) ===")
        _tab2 = tabulate(
                [
                    [
                        int(top_row["rank"]),
                        top_row["variant_id"],
                        top_row["gene"],
                        f"{float(top_row['causal_probability']):.2f}",
                        f"{float(top_row['confidence']):.2f}",
                        textwrap.shorten(str(narr_map.get(top_row["variant_id"], ("", 0.0))[0]), width=120),
                        f"{float(narr_map.get(top_row['variant_id'], ('', 0.0))[1]):.2f}",
                    ]
                ],
                headers=["rank", "variant_id", "gene", "prob", "conf", "narrative", "narr_conf"],
                tablefmt="github",
            )
        try:
            print(_tab2)
        except UnicodeEncodeError:
            print("[Preview skipped (encoding)]")

    merged_path = OUTPUT_DIR / f"multi_agent_shot{shot_idx}_{SHOT_TIMESTAMP}.csv"
    final_df.to_csv(merged_path, index=False, encoding="utf-8-sig")
    written_paths.append(merged_path)

if gene_ranks_per_run and all_genes_across_runs:
    genes_sorted = sorted(all_genes_across_runs)
    num_runs = len(gene_ranks_per_run)
    max_rank_for_missing = TOP_K + 1

    pairwise_corrs: List[float] = []
    if num_runs >= 2:
        for i in range(num_runs):
            x = [float(gene_ranks_per_run[i].get(g, max_rank_for_missing)) for g in genes_sorted]
            for j in range(i + 1, num_runs):
                y = [float(gene_ranks_per_run[j].get(g, max_rank_for_missing)) for g in genes_sorted]
                pairwise_corrs.append(_spearman_corr_from_ranks(x, y))
        avg_pairwise_spearman = sum(pairwise_corrs) / len(pairwise_corrs) if pairwise_corrs else 0.0
    else:
        avg_pairwise_spearman = 1.0

    gene_mean_ranks: Dict[str, float] = {}
    gene_variances: Dict[str, float] = {}
    gene_run_counts: Dict[str, int] = {}
    for g in genes_sorted:
        ranks = [float(run_map.get(g, max_rank_for_missing)) for run_map in gene_ranks_per_run]
        mean_rank = sum(ranks) / num_runs
        variance = sum((r - mean_rank) ** 2 for r in ranks) / num_runs
        gene_mean_ranks[g] = mean_rank
        gene_variances[g] = variance
        gene_run_counts[g] = sum(1 for run_map in gene_ranks_per_run if g in run_map)

    variability_path = OUTPUT_DIR / f"multi_agent_rank_variability_{SHOT_TIMESTAMP}.csv"
    with variability_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([f"{avg_pairwise_spearman:.6f}"])
        writer.writerow(["gene", "variance"])
        for g in genes_sorted:
            writer.writerow([g, f"{gene_variances[g]:.6f}"])
    written_paths.append(variability_path)

    integrated_rows = sorted(
        [
            (
                g,
                gene_mean_ranks.get(g, float("nan")),
                gene_variances.get(g, float("nan")),
                gene_run_counts.get(g, 0),
            )
            for g in genes_sorted
        ],
        key=lambda x: x[1],
    )

    integrated_csv_path = OUTPUT_DIR / f"multi_agent_integrated_ranking_{SHOT_TIMESTAMP}.csv"
    with integrated_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["gene", "average_rank", "rank_variance", "n_runs_seen"])
        for g, mean_rank, variance, run_count in integrated_rows:
            writer.writerow([g, f"{mean_rank:.6f}", f"{variance:.6f}", run_count])
    written_paths.append(integrated_csv_path)

    integrated_html_path = OUTPUT_DIR / f"multi_agent_integrated_ranking_{SHOT_TIMESTAMP}.html"
    integrated_data_genes: List[Dict[str, Any]] = []
    for g, mean_rank, variance, run_count in integrated_rows:
        runs_detail: List[Dict[str, Any]] = []
        gene_meta: Dict[str, Any] = {
            "impact_bucket": "",
            "WB_Overview": "",
            "MANUAL_DESCRIPTION_WB": "",
            "effect": "",
            "user_annotation": "",
            "pubmed_ref_avaiable": "",
            "filtered_pubmed": "",
            "agent1_publication_comments": "",
        }
        for idx in range(num_runs):
            run_idx = idx + 1
            ranks_map = gene_ranks_per_run[idx]
            details_map = gene_run_details_per_run[idx] if idx < len(gene_run_details_per_run) else {}
            d = (details_map.get(g, {}) or {})
            # capture per-run narrative/rationale
            runs_detail.append(
                {
                    "run_index": run_idx,
                    "rank": ranks_map.get(g),
                    "Narrative": d.get("Narrative", ""),
                    "Rationale": d.get("Rationale", ""),
                }
            )
            # capture static gene metadata once (same across runs)
            if d and not gene_meta["impact_bucket"]:
                gene_meta["impact_bucket"] = d.get("impact_bucket", "") or ""
                gene_meta["WB_Overview"] = d.get("WB_Overview", "") or ""
                gene_meta["MANUAL_DESCRIPTION_WB"] = d.get("MANUAL_DESCRIPTION_WB", "") or ""
                gene_meta["effect"] = d.get("effect", "") or ""
                gene_meta["user_annotation"] = d.get("user_annotation", "") or ""
                gene_meta["pubmed_ref_avaiable"] = d.get("pubmed_ref_avaiable", "") or ""
                gene_meta["filtered_pubmed"] = d.get("filtered_pubmed", "") or ""
                gene_meta["agent1_publication_comments"] = d.get("agent1_publication_comments", "") or ""
        integrated_data_genes.append(
            {
                "gene": g,
                "average_rank": mean_rank,
                "rank_variance": variance,
                "n_runs_seen": run_count,
                "impact_bucket": gene_meta["impact_bucket"],
                "WB_Overview": gene_meta["WB_Overview"],
                "MANUAL_DESCRIPTION_WB": gene_meta["MANUAL_DESCRIPTION_WB"],
                "effect": gene_meta["effect"],
                "user_annotation": gene_meta["user_annotation"],
                "pubmed_ref_avaiable": gene_meta["pubmed_ref_avaiable"],
                "filtered_pubmed": gene_meta["filtered_pubmed"],
                "agent1_publication_comments": gene_meta["agent1_publication_comments"],
                "runs": runs_detail,
            }
        )

    integrated_payload = {
        "timestamp": SHOT_TIMESTAMP,
        "num_runs": num_runs,
        "genes": integrated_data_genes,
    }
    data_json = json.dumps(integrated_payload, ensure_ascii=False).replace("</", "<\\/")

    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Integrated Ranking</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
    header { padding: 12px 16px; background: #1f2933; color: #f9fafb; }
    header h1 { margin: 0; font-size: 20px; }
    header p { margin: 4px 0 0 0; font-size: 13px; color: #d1d5db; }
    main { display: flex; height: calc(100vh - 64px); }
    #table-container { flex: 1; overflow: auto; padding: 12px 16px; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }
    th { background: #f3f4f6; position: sticky; top: 0; z-index: 1; }
    tr:hover { background: #eef2ff; cursor: pointer; }
    #detail-panel { width: 38%; max-width: 520px; border-left: 1px solid #e5e7eb; padding: 12px 16px; overflow: auto; background: #f9fafb; }
    #detail-title { font-weight: bold; margin-bottom: 8px; }
    #detail-table { border-collapse: collapse; width: 100%; font-size: 12px; }
    #detail-table th, #detail-table td { border: 1px solid #e5e7eb; padding: 4px 6px; vertical-align: top; }
    #detail-table th { background: #e5e7eb; }
    .muted { color: #6b7280; }
  </style>
</head>
<body>
  <header>
    <h1>Integrated Gene Ranking</h1>
    <p>Click a gene to inspect narratives and rationales across runs (timestamp: {timestamp}, runs: {num_runs}).</p>
  </header>
  <main>
    <section id="table-container">
      <table id="genes-table">
        <thead>
          <tr>
            <th>Gene</th>
            <th>Average rank</th>
            <th>Rank variance</th>
            <th>Runs seen</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </section>
    <aside id="detail-panel">
      <div id="detail-title" class="muted">Select a gene to see narratives, rationales, and gene-level context.</div>
      <div id="detail-meta" class="muted" style="font-size:12px; margin-bottom:8px;"></div>
      <table id="detail-table" style="display:none;">
        <thead>
          <tr>
            <th>Run</th>
            <th>Narrative</th>
            <th>Rationale</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </aside>
  </main>
  <script>
    const DATA = __DATA_JSON__;

    function formatNumber(x) {
      if (x === null || x === undefined || isNaN(x)) return '';
      return x.toFixed(3);
    }

    function renderTable() {
      const tbody = document.querySelector('#genes-table tbody');
      tbody.innerHTML = '';
      // DATA.genes is already sorted by average_rank ascending in the Python script.
      const genes = DATA.genes || [];
      for (const g of genes) {
        const tr = document.createElement('tr');
        tr.dataset.gene = g.gene;
        tr.innerHTML = `
          <td>${g.gene}</td>
          <td>${formatNumber(g.average_rank)}</td>
          <td>${formatNumber(g.rank_variance)}</td>
          <td>${g.n_runs_seen}</td>
        `;
        tr.addEventListener('click', () => renderDetail(g));
        tbody.appendChild(tr);
      }
    }

    function renderDetail(geneEntry) {
      const title = document.getElementById('detail-title');
      const meta = document.getElementById('detail-meta');
      const table = document.getElementById('detail-table');
      const tbody = table.querySelector('tbody');
      title.textContent = `Details for gene: ${geneEntry.gene}`;
      const metaLines = [];
      if (geneEntry.effect) metaLines.push(`<strong>Effect</strong>: ${geneEntry.effect}`);
      if (geneEntry.impact_bucket) metaLines.push(`<strong>impact_bucket</strong>: ${geneEntry.impact_bucket}`);
      if (geneEntry.user_annotation) metaLines.push(`<strong>user_annotation</strong>: ${geneEntry.user_annotation}`);
      if (geneEntry.WB_Overview) metaLines.push(`<strong>WB_Overview</strong>: ${geneEntry.WB_Overview}`);
      if (geneEntry.MANUAL_DESCRIPTION_WB) metaLines.push(`<strong>MANUAL_DESCRIPTION_WB</strong>: ${geneEntry.MANUAL_DESCRIPTION_WB}`);
      if (geneEntry.pubmed_ref_avaiable) metaLines.push(`<strong>pubmed_ref_avaiable</strong>: ${geneEntry.pubmed_ref_avaiable}`);
      if (geneEntry.filtered_pubmed) metaLines.push(`<strong>filtered_pubmed</strong>: ${geneEntry.filtered_pubmed}`);
      if (geneEntry.agent1_publication_comments) metaLines.push(`<strong>agent1_publication_comments</strong>: ${geneEntry.agent1_publication_comments}`);
      meta.innerHTML = metaLines.length ? metaLines.join('<br/>') : '';
      tbody.innerHTML = '';
      let hasRows = false;
      for (const r of geneEntry.runs) {
        const hasText = (r.Narrative && r.Narrative.trim()) || (r.Rationale && r.Rationale.trim());
        if (!hasText) continue;
        hasRows = true;
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${r.run_index}${r.rank ? ` (rank ${r.rank})` : ''}</td>
          <td>${r.Narrative || ''}</td>
          <td>${r.Rationale || ''}</td>
        `;
        tbody.appendChild(tr);
      }
      table.style.display = hasRows ? 'table' : 'none';
      if (!hasRows) {
        title.textContent += ' (no narratives/rationales recorded in the top list)';
      }
    }

    renderTable();
  </script>
</body>
</html>
""".replace("__DATA_JSON__", data_json).replace("{timestamp}", SHOT_TIMESTAMP).replace("{num_runs}", str(num_runs))

    integrated_html_path.write_text(html_template, encoding="utf-8")
    written_paths.append(integrated_html_path)

print("\nWROTE:")
for p in written_paths:
    print(f"- {p.resolve()}")

