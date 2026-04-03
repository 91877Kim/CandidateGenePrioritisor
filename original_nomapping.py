#!/usr/bin/env python3


# =========================
# Setup (installs + imports)
# =========================
# You may comment the next line on repeat runs.

import os, re, json, math, time, pathlib, textwrap, csv
from typing import List, Dict, Any, Optional, Tuple
import requests
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, conint, confloat, field_validator
from datetime import datetime, timezone
from tabulate import tabulate

# =========================
# API keys (script-style hard-coded)
# =========================
os.environ["OPENAI_API_KEY"] = "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"
os.environ["NCBI_API_KEY"] = "605418d3446353da6f9b17b37310caf65e08"
print("[Info] Loaded OPENAI_API_KEY and NCBI_API_KEY from script configuration.")

# =========================
# Paths
# =========================
CSV_PATH = os.getenv("CSV_PATH", "drp1.csv")

def _with_fallback(p: str, candidates: List[str]) -> str:
    pth = pathlib.Path(p)
    if pth.exists():
        return str(pth)
    for c in candidates:
        if pathlib.Path(c).exists():
            print(f"[Info] Using fallback file: {c}")
            return c
    return str(pth)

CSV_PATH = _with_fallback(
    CSV_PATH,
    [
        "drp1.csv",
        "/mnt/data/hu80_WS245_annotated-variants-mapping-region.csv",
        "/mnt/data/hu80_WS245_annotated-variants-mapping-region (1).csv",
    ],
)

# =========================
# Model and run parameters
# =========================
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-5-mini")

OPENAI_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE", "")   # leave empty to auto-adapt
OPENAI_SEED        = os.getenv("OPENAI_SEED", "")
OPENAI_TIMEOUT     = int(os.getenv("OPENAI_TIMEOUT", "360"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "5"))
OPENAI_BACKOFF_BASE= float(os.getenv("OPENAI_BACKOFF_BASE", "2.0"))

TOP_K                    = 30        # how many to keep in ranking
MAX_VARIANTS_TO_SEND     = 250
PREFER_EMS               = True
NUM_LLM_RUNS             = 10         # hard-coded number of LLM runs on this dataset
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

OUTPUT_DIR = pathlib.Path("llm_variant_prioritization_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Networking helpers (robust)
# =========================
def _requests_get(url: str, params: Dict[str, Any], timeout: int = 30, retries: int = 3, backoff: float = 2.0):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(backoff ** i)

def _requests_post_with_retries(url: str, headers: Dict[str, str], payload: Dict[str, Any],
                                timeout: int, max_retries: int, backoff_base: float) -> requests.Response:
    attempt = 0
    while True:
        try:
            return requests.post(url, headers=headers, json=payload, timeout=timeout)
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError):
            if attempt >= max_retries - 1:
                raise
            time.sleep(backoff_base ** attempt)
            attempt += 1

def _post_chat(payload: Dict[str, Any], timeout: int = OPENAI_TIMEOUT) -> str:
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"}
    r = _requests_post_with_retries(
        f"{OPENAI_BASE_URL}/chat/completions",
        headers=headers, payload=payload, timeout=timeout,
        max_retries=OPENAI_MAX_RETRIES, backoff_base=OPENAI_BACKOFF_BASE
    )
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text[:2000]}")
    data = r.json()
    return data["choices"][0]["message"]["content"]

def openai_chat(messages: List[Dict[str, str]],
                model: str,
                response_json: bool = True,
                temperature: Optional[float] = None,
                seed: Optional[int] = None,
                timeout: int = OPENAI_TIMEOUT) -> str:
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
        if ("param" in body and "temperature" in body) or ("Unsupported value" in body and "temperature" in body):
            payload.pop("temperature", None); retried = True
        if ("param" in body and "\"seed\"" in body):
            payload.pop("seed", None); retried = True
        if ("param" in body and "response_format" in body) or ("json_object" in body):
            payload.pop("response_format", None); retried = True
        if retried:
            return _post_chat(payload, timeout=timeout)
        raise

# =========================
# CSV loading + key builders
# =========================
def exists_or_raise(p: str):
    if not pathlib.Path(p).exists():
        raise FileNotFoundError(
            f"File not found: {p}\n"
            "• In Colab, upload via the left Files pane or mount Drive.\n"
            "• Then set CSV_PATH accordingly."
        )

def coalesce(colnames: List[str], df: pd.DataFrame) -> Optional[str]:
    for c in colnames:
        if c in df.columns: return c
    return None

def normalize_chr_label(chrom: Any) -> str:
    s = str(chrom)
    s = re.sub(r'(?i)^chromosome\s*', '', s)
    s = re.sub(r'(?i)^chr', '', s)
    s = s.strip()
    map_arabic = {"1":"I","2":"II","3":"III","4":"IV","5":"V","10":"X"}
    if s in map_arabic: return map_arabic[s]
    return s.upper()

def build_variant_id(chrom, pos, ref, alt) -> str:
    c = normalize_chr_label(chrom) if pd.notna(chrom) else "NA"
    p = str(int(pos)) if pd.notna(pos) else "NA"
    r = str(ref).upper() if pd.notna(ref) else "N"
    a = str(alt).upper() if pd.notna(alt) else "N"
    return f"{c}:{p} {r}>{a}"

def short(s: Any, maxlen: int = 60) -> str:
    t = "" if s is None else str(s)
    return t if len(t) <= maxlen else (t[:maxlen-1] + "…")

def normalize_float(s: pd.Series, clamp01: bool = False) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if clamp01:
        out = out.clip(lower=0.0, upper=1.0)
    return out

def _is_blank(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return str(x).strip() == ""

def parse_snpeff_ann_from_info(info_value: Any) -> Dict[str, str]:
    """
    Parse first SnpEff ANN entry from VCF INFO.
    ANN format:
    Allele|Annotation|Annotation_Impact|Gene_Name|Gene_ID|...
    """
    if _is_blank(info_value):
        return {"gene": "", "effect": "", "impact": ""}
    s = str(info_value)
    m = re.search(r"(?:^|;)ANN=([^;]+)", s)
    if not m:
        return {"gene": "", "effect": "", "impact": ""}
    ann_blob = m.group(1)
    first = ann_blob.split(",")[0].strip()
    parts = first.split("|")
    gene = parts[3].strip() if len(parts) > 3 else ""
    effect = parts[1].strip() if len(parts) > 1 else ""
    impact = parts[2].strip() if len(parts) > 2 else ""
    return {
        "gene": gene,
        "effect": effect,
        "impact": impact
    }

# =========================
# Load primary annotated CSV
# =========================
exists_or_raise(CSV_PATH)
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df):,} variants from: {CSV_PATH}")
# Normalize headers before column discovery (handles BOM/VCF-like headers)
df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

# Column discovery (primary)
chrom_col = coalesce(["CHROM","Chrom","chr","chrom","#CHROM","#chrom"], df)
pos_col   = coalesce(["POS","Position","POS_GRCh","pos"], df)
ref_col   = coalesce(["REF","Ref","reference_allele","ref"], df)
alt_col   = coalesce(["ALT","Alt","alt_allele","alt"], df)

if chrom_col is None or pos_col is None:
    raise KeyError(
        "Required columns not found in input CSV. "
        f"Detected columns: {list(df.columns)}. "
        "Expected one of CHROM/Chrom/chr/chrom/#CHROM/#chrom and POS/Position/POS_GRCh/pos."
    )

gene_col   = coalesce(["Gene_name","Gene","GENE","SYMBOL","Symbol","GeneName","ANN_Gene_Name","HGNC"], df)
effect_col = coalesce(["Effect","Consequence","ANN_Consequence","ANN_Annotation","VEP_CONSEQUENCE"], df)
impact_col = coalesce(["IMPACT","Impact","ANN_Annotation_Impact","VEP_IMPACT"], df)
sift_col   = coalesce(["SIFT","SIFT_pred","SIFT_PRED"], df)
pphen_col  = coalesce(["PolyPhen","PolyPhen_pred","PolyPhen_PRED"], df)
qual_col = coalesce(["QUAL","Qual","qual"], df)
dp_col = coalesce(["DP","Depth","depth","dp"], df)
ems_label_col = coalesce(["EMS_label","ems_label","EMS","ems"], df)
info_col = coalesce(["INFO","Info","info"], df)

# Build standardized keys for primary
df["_CHROM_N"] = df[chrom_col].apply(normalize_chr_label)
df["_POS_I"]   = pd.to_numeric(df[pos_col], errors="coerce").astype("Int64")
if ref_col: df["_REF_U"] = df[ref_col].astype(str).str.upper()
else:       df["_REF_U"] = "N"
if alt_col: df["_ALT_U"] = df[alt_col].astype(str).str.upper()
else:       df["_ALT_U"] = "N"
df["variant_id"] = df.apply(lambda r: build_variant_id(r["_CHROM_N"], r["_POS_I"], r["_REF_U"], r["_ALT_U"]), axis=1)
df["pos_key"]    = df["_CHROM_N"].astype(str) + ":" + df["_POS_I"].astype(str)

# Fallback extraction for VCF-like inputs: populate gene/effect/impact from INFO->ANN.
if info_col and (gene_col is None or effect_col is None or impact_col is None):
    ann_parsed = df[info_col].apply(parse_snpeff_ann_from_info).apply(pd.Series)
    ann_gene_col = "_ANN_GENE"
    ann_effect_col = "_ANN_EFFECT"
    ann_impact_col = "_ANN_IMPACT"
    df[ann_gene_col] = ann_parsed["gene"]
    df[ann_effect_col] = ann_parsed["effect"]
    df[ann_impact_col] = ann_parsed["impact"]
    print(
        f"[Info] Parsed ANN fallback from INFO for "
        f"{int((df[ann_gene_col].astype(str).str.strip() != '').sum()):,} rows "
        f"(gene/effect/impact)."
    )
else:
    ann_gene_col = ann_effect_col = ann_impact_col = None

# =========================
# Build records (using only values present in input CSV)
# =========================
def effect_severity_score(effect_text: str, impact_bucket: str) -> float:
    t = (effect_text or "").lower()
    base = {"HIGH": 3.0, "MODERATE": 2.0, "LOW": 1.0, "UNKNOWN": 0.5}.get((impact_bucket or "UNKNOWN").upper(), 0.5)
    if any(k in t for k in ["stop_gained","frameshift","splice_acceptor","splice_donor","start_lost","stop_lost","exon_loss"]):
        base = max(base, 3.2)
    elif any(k in t for k in ["missense","inframe_insertion","inframe_deletion","protein_altering"]):
        base = max(base, 2.2)
    elif any(k in t for k in ["synonymous","upstream","downstream","utr","intron"]):
        base = min(base, 1.0)
    return float(base)

def impact_bucket_from(effect_text: str, impact_bucket: Optional[str]) -> str:
    if isinstance(impact_bucket, str) and impact_bucket:
        ib = impact_bucket.strip().upper()
        if ib in {"HIGH","MODERATE","LOW"}:
            return ib
    t = (effect_text or "").lower()
    if any(k in t for k in ["stop_gained","frameshift","splice_acceptor","splice_donor","start_lost","stop_lost"]): return "HIGH"
    if any(k in t for k in ["missense","inframe_insertion","inframe_deletion","protein_altering"]): return "MODERATE"
    if any(k in t for k in ["synonymous","upstream","downstream","utr","intron"]): return "LOW"
    return "UNKNOWN"

def coerce_float(x):
    try: return float(x)
    except Exception: return None

packed_all: List[Dict[str, Any]] = []
for _, row in df.iterrows():
    vid = row["variant_id"]; pkey = row["pos_key"]
    gene   = row.get(gene_col) if gene_col else None
    effect = row.get(effect_col) if effect_col else None
    impact_raw = row.get(impact_col) if impact_col in df.columns else None

    if _is_blank(gene) and ann_gene_col:
        gene = row.get(ann_gene_col)
    if _is_blank(effect) and ann_effect_col:
        effect = row.get(ann_effect_col)
    if _is_blank(impact_raw) and ann_impact_col:
        impact_raw = row.get(ann_impact_col)

    ib     = impact_bucket_from(effect, impact_raw)

    qual           = coerce_float(row.get(qual_col)) if qual_col else None
    depth          = coerce_float(row.get(dp_col)) if dp_col else None
    ems_label_raw  = row.get(ems_label_col) if ems_label_col else None

    # Normalize EMS boolean
    ems_bool = False
    if isinstance(ems_label_raw, str):
        ems_bool = ems_label_raw.strip().lower().startswith("ems")

    rec = {
        "variant_id": vid,
        "pos_key": pkey,
        "chrom": row["_CHROM_N"],
        "pos": int(row["_POS_I"]) if pd.notna(row["_POS_I"]) else None,
        "gene": (gene if pd.notna(gene) else "") if gene is not None else "",
        "effect": effect if pd.notna(effect) else "",
        "impact_bucket": ib,
        "sift": short(row.get(sift_col), 28) if sift_col else "",
        "polyphen": short(row.get(pphen_col), 28) if pphen_col else "",
        "ems": ems_bool,
        "ems_label_raw": ems_label_raw if pd.notna(ems_label_raw) else "",
        "qual": qual,
        "depth": depth,
    }
    ems_bonus = 0.15 if (PREFER_EMS and rec["ems"]) else 0.0
    imp_w = {"HIGH":0.45,"MODERATE":0.25,"LOW":0.05,"UNKNOWN":0.0}.get(rec["impact_bucket"],0.0)
    q = rec["qual"]; q_w = 0.0 if (q is None or (isinstance(q,float) and math.isnan(q))) else min(float(q), 200)/200.0 * 0.05
    rec["_pre_score"] = ems_bonus + imp_w + q_w
    rec["_severity"]  = effect_severity_score(rec["effect"], rec["impact_bucket"])
    packed_all.append(rec)

# Deduplicate by variant_id, keeping most severe/relevant
def tie_break_key(r):
    ems = 1.0 if r.get("ems") else 0.0
    q = r.get("qual") or 0.0
    return (r["_severity"], ems, q)

packed_dedup: List[Dict[str, Any]] = []
for vid, grp in pd.DataFrame(packed_all).groupby("variant_id"):
    best = max(grp.to_dict("records"), key=tie_break_key)
    packed_dedup.append(best)
packed_df = pd.DataFrame(packed_dedup)
print(f"Prepared base records for {len(packed_df):,} variants after dedup by most severe effect.")

# =========================
# Pre-filter for LLM
# =========================
pref = []
for rec in packed_dedup:
    pref.append(rec)
if not pref:
    pref = packed_dedup
pref = sorted(pref, key=lambda r: (r["_pre_score"], r["_severity"]), reverse=True)[:MAX_VARIANTS_TO_SEND]
print(f"Will send {len(pref):,} variants to GPT‑5‑mini "
      f"(MAX_VARIANTS_TO_SEND={MAX_VARIANTS_TO_SEND}).")

# =========================
# Lightweight gene-knowledge fetch (NCBI Gene summary + PubMed refs)
# =========================
def fetch_gene_knowledge(gene: str) -> Dict[str, Any]:
    gene = (gene or "").strip()
    if not gene: return {"summary":"", "refs":[]}
    # NCBI Gene summary
    api_key = os.getenv("NCBI_API_KEY","")
    params_es = {"db":"gene","retmode":"json",
                 "term": f"{gene}[Gene Name] AND Caenorhabditis elegans[Organism]"}
    if api_key: params_es["api_key"] = api_key
    summary_text = ""
    refs = []
    try:
        es = _requests_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params_es, timeout=20)
        js = es.json()
        ids = js.get("esearchresult",{}).get("idlist",[])
        if ids:
            gid = ids[0]
            params_sum = {"db":"gene","retmode":"json","id":gid}
            if api_key: params_sum["api_key"] = api_key
            su = _requests_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", params_sum, timeout=20).json()
            doc = su.get("result",{}).get(gid,{})
            summary_text = doc.get("summary","") or doc.get("description","")
    except Exception:
        pass
    # PubMed refs (relevance to dopamine/neuron fate if available)
    try:
        q = f'({gene}) AND (Mystery cells of male(MCMs) OR neuron) AND elegans'
        params_pm = {"db":"pubmed","retmode":"json","retmax":"3","sort":"relevance","term": q}
        if api_key: params_pm["api_key"] = api_key
        pm = _requests_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params_pm, timeout=20).json()
        idlist = pm.get("esearchresult",{}).get("idlist",[])[:3]
        if idlist:
            params_sm = {"db":"pubmed","retmode":"json","id":",".join(idlist)}
            if api_key: params_sm["api_key"] = api_key
            sm = _requests_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", params_sm, timeout=20).json()
            for pid in idlist:
                it = sm.get("result",{}).get(pid,{})
                title = it.get("title","")
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
                if title:
                    refs.append({"pmid": pid, "title": title, "url": url})
    except Exception:
        pass
    return {"summary": summary_text.strip(), "refs": refs}

# Build knowledge map for unique genes in 'pref'
unique_genes = sorted({(r.get("gene") or "").strip() for r in pref if r.get("gene")})
gene_knowledge: Dict[str, Dict[str, Any]] = {}
for g in unique_genes:
    try:
        gene_knowledge[g] = fetch_gene_knowledge(g)
    except Exception:
        gene_knowledge[g] = {"summary":"", "refs":[]}

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
        if not v: raise ValueError("ranking must be non-empty")
        return v
    @field_validator("annotations")
    @classmethod
    def non_empty_ann(cls, v):
        if not v: raise ValueError("annotations must be non-empty")
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


def unique_rank_items_by_gene(
    ranking: List[VariantRankItem], top_k: int
) -> List[VariantRankItem]:
    """
    Enforce one ranking entry per gene in case the LLM returns duplicates.
    Keeps first (best-ranked) occurrence per gene.
    """
    ranked = sorted(ranking, key=lambda x: x.rank)
    seen: set[str] = set()
    unique: List[VariantRankItem] = []
    for it in ranked:
        gene_norm = (it.gene or "").strip().lower()
        variant_norm = (it.variant_id or "").strip().lower()
        dedup_key = f"gene:{gene_norm}" if gene_norm else f"variant:{variant_norm}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        unique.append(it)
        if len(unique) >= top_k:
            break
    removed = len(ranked) - len(unique)
    if removed > 0:
        print(f"[Info] Removed {removed} duplicate gene entries from LLM ranking output.")
    return unique


def sanitize_text_for_output(val: Any) -> Any:
    """
    Normalize unicode-heavy biology terms to ASCII-safe text for CSV compatibility
    across spreadsheet tools with mixed encoding auto-detection.
    """
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return val
    s = str(val)
    replacements = {
        "Wnt/尾-catenin": "Wnt/beta-catenin",
        "尾-catenin": "beta-catenin",
        "β-catenin": "beta-catenin",
        "β": "beta",
        "α": "alpha",
        "γ": "gamma",
        "δ": "delta",
        "κ": "kappa",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    return s

# =========================
# Build prompts (allow gene knowledge + references)
# =========================
SYSTEM_PROMPT_MAIN = """You are an expert in C. elegans forward genetics and mystery cells of male  fate.

Context:
- MCM neurons are a pair of male-specific interneurons in the head of C. elegans. They are normally produced during larval male sexual maturation by an asymmetric division of the amphid socket glial cell (AMso). In the mutant phenotype of interest, the AMso divides symmetrically to generate two glial cells instead of the MCM, so the MCM neurons are absent. We aim to identify the causal allele responsible for this defect.
- Screen detects loss of mystery cells of male neurons via a GFP reporter.
- Goal: within the provided candidate variant CSV, identify variants most likely to cause mystery cells of male loss.

Evidence you may use:
- Structured fields: Gene_name, Effect, impact_bucket, EMS_label (EMS mutagen), QUAL, DP, SIFT/PolyPhen.
- Gene knowledge: brief summaries and PubMed links (if provided) for the gene's known/putative function in C. elegans or related biology.

Instructions:
- Rank variants by causal probability (0–1) and provide confidence (0–1).
- For EACH variant, write a 2–5 sentence narrative connecting the mutation + gene function to MCM fate loss.
- **Cite evidence in brackets**, e.g., [Effect=stop_gained; EMS; GeneRef: PMID 12345, 67890].
- Prefer HIGH-impact (nonsense/frameshift/essential splice) > damaging missense > low/unknown; Synonymous variants should be aggressively penalized; EMS increases prior but is not required.
- Do not fabricate references; only cite provided PubMed links. Output JSON only; no chain-of-thought."""

def format_variant_for_llm(r: Dict[str, Any]) -> Dict[str, Any]:
    def safe_round(x, nd=3):
        try: return round(float(x), nd)
        except Exception: return None
    g = (r.get("gene") or "").strip()
    gk = gene_knowledge.get(g, {"summary":"", "refs":[]})
    return {
        "variant_id": r["variant_id"],
        "Gene_name": g,
        "Effect": short(r.get("effect"), 90),
        "impact_bucket": r.get("impact_bucket","UNKNOWN"),
        "ems_label": ("EMS" if r.get("ems") else "non-EMS"),
        "QUAL": safe_round(r.get("qual"), 1),
        "DP": (int(r["depth"]) if r.get("depth") is not None and not (isinstance(r["depth"],float) and math.isnan(r["depth"])) else None),
        "SIFT": r.get("sift",""),
        "PolyPhen": r.get("polyphen",""),
        "_pre_score": round(float(r.get("_pre_score", 0.0)), 3),
        "gene_knowledge": {
            "summary": short(gk.get("summary",""), 480),
            "pubmed_refs": gk.get("refs", [])
        }
    }

variants_for_llm = [format_variant_for_llm(r) for r in pref]

USER_PROMPT_MAIN = {
    "task": "Prioritize variants for MCM neuron loss resulting from the AMSo cells(progenitors for MCMs) divide symmetrically to generate two glial cells instead of the MCM, and annotate variant with a narrative + confidence leveraging gene function and provided PubMed links.",
    "instructions": {
        "ranking_size": min(TOP_K, len(variants_for_llm)),
        "ranking_unit": "genes",
        "must_return_exact_ranking_size": True,
        "no_duplicate_genes": True,
        "return_all_ranked": True,
        "notes": [
            "Use provided gene summaries and PubMed refs if present; otherwise rely on mutation class + general neuronal mechanisms.",
            "Narratives 2–5 sentences; include bracketed evidence and [GeneRef: PMID ...] when refs provided.",
            "Return one ranking entry per gene (no duplicate genes). If multiple variants map to the same gene, keep only the best-supported one."
        ]
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
            "key_evidence": ["short bullet strings"]
        },
        "ranking": [
            {
                "rank": 1,
                "variant_id": "string",
                "gene": "string",
                "causal_probability": 0.00,
                "confidence": 0.00,
                "rationale": "string",
                "key_evidence": ["short bullet strings"]
            }
        ],
        "annotations": [
            {
                "variant_id": "string",
                "gene": "string",
                "narrative": "string",
                "narrative_confidence": 0.00
            }
        ]
    },
    "variants": variants_for_llm
}

# =========================
# Call LLM (single-shot, then fallback)
# =========================
def single_shot_call() -> Optional[LLMVariantOutput]:
    temp_val = float(OPENAI_TEMPERATURE) if OPENAI_TEMPERATURE.strip() else None
    seed_val = int(OPENAI_SEED) if OPENAI_SEED.strip() else None
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_MAIN},
        {"role": "user", "content": json.dumps(USER_PROMPT_MAIN, ensure_ascii=False, separators=(",", ":"))}
    ]
    print(f"Calling {OPENAI_MODEL} with {len(variants_for_llm)} variants (single-shot)…")
    raw = openai_chat(messages, model=OPENAI_MODEL, response_json=True, temperature=temp_val, seed=seed_val, timeout=OPENAI_TIMEOUT)
    parsed = try_parse_json(raw)
    return LLMVariantOutput(**parsed)

# Fallback: chunked narratives + ranking from summaries
NARR_SYSTEM = """You are an expert in C. elegans MCM neuron biology.
For each variant, write a 2–5 sentence narrative using provided fields AND gene summaries/PMIDs if present.
Cite evidence in brackets, e.g., [Effect=missense; EMS; GeneRef: PMID 12345].
Return JSON: {"annotations":[{variant_id, gene, narrative, narrative_confidence}, ...]}"""
def llm_narratives_in_chunks(items: List[Dict[str, Any]], chunk_size: int = 18) -> Dict[str, Tuple[str, float]]:
    out: Dict[str, Tuple[str, float]] = {}
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i+chunk_size]
        user = {"variants": chunk, "schema": {"annotations":[{"variant_id":"string","gene":"string","narrative":"string","narrative_confidence":0.00}]}}
        messages = [
            {"role": "system", "content": NARR_SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False, separators=(",", ":"))}
        ]
        raw = openai_chat(messages, model=OPENAI_MODEL, response_json=True, temperature=None, seed=None, timeout=OPENAI_TIMEOUT)
        parsed = try_parse_json(raw)
        anns = parsed.get("annotations", [])
        for a in anns:
            try:
                va = VariantNarrative(**a)
                out[va.variant_id] = (va.narrative, float(va.narrative_confidence))
            except ValidationError:
                pass
    return out

RANK_SYSTEM = """Rank genes for causing MCM neuron fate loss using ONLY provided summaries (fields + gene refs).
The variant_summaries may include multiple variants per gene; return one ranking entry per gene (no duplicate genes), using the best-supported variant_id for that gene.
Heuristics: HIGH-impact > damaging missense > low/unknown; EMS prior; integrate narratives & their confidence.
Output JSON only with fields: summary, most_likely, ranking[]."""
def llm_ranking_from_summaries(summaries: List[Dict[str, Any]]) -> LLMRankingOnly:
    user = {
        "task": "Rank all candidate genes by likelihood of causing MCM neuron fate loss. Return one entry per gene (no duplicate genes); if a gene appears in multiple summaries, keep the best-supported variant_id for that gene.",
        "top_k": min(TOP_K, len(summaries)),
        "ranking_unit": "genes",
        "must_return_exact_top_k": True,
        "no_duplicate_genes": True,
        "variant_summaries": summaries,
        "schema": {
            "summary": "string",
            "most_likely": {
                "rank": 1, "variant_id": "string", "gene": "string",
                "causal_probability": 0.00, "confidence": 0.00,
                "rationale": "string", "key_evidence": ["short strings"]
            },
            "ranking": [
                {"rank": 1, "variant_id": "string", "gene": "string",
                 "causal_probability": 0.00, "confidence": 0.00,
                 "rationale": "string", "key_evidence": ["short strings"]}
            ]
        }
    }
    messages = [
        {"role": "system", "content": RANK_SYSTEM},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, separators=(",", ":"))}
    ]
    raw = openai_chat(messages, model=OPENAI_MODEL, response_json=True, temperature=None, seed=None, timeout=OPENAI_TIMEOUT)
    parsed = try_parse_json(raw)
    return LLMRankingOnly(**parsed)

def chunked_fallback_pipeline() -> Tuple[pd.DataFrame, Dict[str, Tuple[str,float]]]:
    print("Falling back to chunked pipeline: generating narratives in chunks…")
    narr_map = llm_narratives_in_chunks(variants_for_llm, chunk_size=18)
    print("Chunked pipeline: ranking from compact summaries…")
    summaries = []
    for v in variants_for_llm:
        vid = v["variant_id"]
        narrative, narr_conf = narr_map.get(vid, ("", 0.0))
        summaries.append({
            "variant_id": vid,
            "gene": v.get("Gene_name",""),
            "Effect": v.get("Effect",""),
            "impact_bucket": v.get("impact_bucket","UNKNOWN"),
            "ems_label": v.get("ems_label","non-EMS"),
            "QUAL": v.get("QUAL", None),
            "DP": v.get("DP", None),
            "gene_knowledge": v.get("gene_knowledge", {}),
            "narrative": narrative,
            "narrative_confidence": narr_conf,
            "_pre_score": v.get("_pre_score", 0.0)
        })
    ranking_only = llm_ranking_from_summaries(summaries)
    ranked_unique = unique_rank_items_by_gene(ranking_only.ranking, TOP_K)
    rank_rows = []
    for rank_idx, it in enumerate(ranked_unique, start=1):
        rank_rows.append({
            "run_id": RUN_ID,
            "variant_id": it.variant_id,
            "gene": sanitize_text_for_output(it.gene),
            "rank": rank_idx,
            "causal_probability": round(float(it.causal_probability), 2),
            "confidence": round(float(it.confidence), 2),
            "rationale": sanitize_text_for_output(it.rationale),
            "key_evidence": sanitize_text_for_output("; ".join(it.key_evidence) if it.key_evidence else ""),
            "Narrative": sanitize_text_for_output(narr_map.get(it.variant_id, ("", None))[0]),
            "Narrative_confidence": round(narr_map.get(it.variant_id, ("", 0.0))[1], 2) if it.variant_id in narr_map else None
        })
    return pd.DataFrame(rank_rows).sort_values(["rank","variant_id"]), narr_map

# =========================
# Execute: multi-shot with fixed run count
# =========================
SHOT_TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M")
written_paths: List[pathlib.Path] = []
gene_ranks_per_run: List[Dict[str, int]] = []
all_genes_across_runs: set[str] = set()
gene_run_details_per_run: List[Dict[str, Dict[str, str]]] = []
final_df: pd.DataFrame = pd.DataFrame()
last_rank_df: pd.DataFrame = pd.DataFrame()

# Merge back selected original fields for convenient viewing (from packed_dedup)
orig_small = pd.DataFrame(packed_dedup).copy()
orig_small_ren = orig_small[[
    "variant_id", "gene", "effect", "impact_bucket", "ems", "ems_label_raw",
    "qual", "depth"
]].rename(columns={"gene": "Gene_name_csv"})


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
        ranked = unique_rank_items_by_gene(llm_out.ranking, TOP_K)
        rows = []
        for rank_idx, it in enumerate(ranked, start=1):
            rows.append({
                "run_id": RUN_ID,
                "variant_id": it.variant_id,
                "gene": sanitize_text_for_output(it.gene),
                "rank": rank_idx,
                "causal_probability": round(float(it.causal_probability), 2),
                "confidence": round(float(it.confidence), 2),
                "rationale": sanitize_text_for_output(it.rationale),
                "key_evidence": sanitize_text_for_output("; ".join(it.key_evidence) if it.key_evidence else ""),
                "Narrative": sanitize_text_for_output(narr_map.get(it.variant_id, ("", None))[0]),
                "Narrative_confidence": round(narr_map.get(it.variant_id, ("", 0.0))[1], 2)
                if it.variant_id in narr_map
                else None
            })
        rank_df = pd.DataFrame(rows).sort_values(["rank", "variant_id"])
    except Exception as e:
        print(f"[Single-shot failed: {e.__class__.__name__}] {e}\n")
        rank_df, narr_map = chunked_fallback_pipeline()

    shot_raw_csv_path = OUTPUT_DIR / f"llm_variant_ranking_shot{shot_idx}_{SHOT_TIMESTAMP}_original.csv"
    shot_raw_jsonl_path = OUTPUT_DIR / f"llm_variant_ranking_shot{shot_idx}_{SHOT_TIMESTAMP}_original.jsonl"
    rank_df.to_csv(shot_raw_csv_path, index=False, encoding="utf-8-sig")
    with open(shot_raw_jsonl_path, "w", encoding="utf-8") as f:
        for _, r in rank_df.iterrows():
            f.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
    written_paths.extend([shot_raw_csv_path, shot_raw_jsonl_path])

    final_df = (rank_df.merge(
        orig_small_ren,
        how="left", on="variant_id"
    ).sort_values(["rank", "variant_id"]))

    # If LLM omits gene in ranking rows, fill from parsed CSV gene column.
    if "gene" in final_df.columns and "Gene_name_csv" in final_df.columns:
        gene_blank = final_df["gene"].isna() | (final_df["gene"].astype(str).str.strip() == "")
        final_df.loc[gene_blank, "gene"] = final_df.loc[gene_blank, "Gene_name_csv"]

    view_cols = [
        "rank", "causal_probability", "confidence",
        "variant_id", "gene", "Gene_name_csv", "effect", "impact_bucket", "ems", "ems_label_raw",
        "qual", "depth",
        "Narrative", "Narrative_confidence", "rationale"
    ]
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
            "effect": str(row.get("effect") or ""),
        }
    gene_run_details_per_run.append(run_gene_details)

    print("\n=== Top ranked variants (preview) ===")
    print(tabulate(final_df.head(20).fillna(""), headers="keys", tablefmt="github", showindex=False))

    if not rank_df.empty:
        top_row = rank_df.sort_values("rank").iloc[0]
        print("\n=== Most likely causal variant (LLM) ===")
        print(tabulate([[
            int(top_row["rank"]), top_row["variant_id"], top_row["gene"],
            f"{float(top_row['causal_probability']):.2f}", f"{float(top_row['confidence']):.2f}",
            textwrap.shorten(str(narr_map.get(top_row["variant_id"], ("", 0.0))[0]), width=120),
            f"{float(narr_map.get(top_row['variant_id'], ('', 0.0))[1]):.2f}"
        ]], headers=["rank", "variant_id", "gene", "prob", "conf", "narrative", "narr_conf"], tablefmt="github"))

    shot_merged_path = OUTPUT_DIR / f"original_nomapping_shot{shot_idx}_{SHOT_TIMESTAMP}.csv"
    final_df.to_csv(shot_merged_path, index=False, encoding="utf-8-sig")
    written_paths.append(shot_merged_path)
    last_rank_df = rank_df

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
            "effect": "",
        }
        for idx in range(num_runs):
            run_idx = idx + 1
            ranks_map = gene_ranks_per_run[idx]
            details_map = gene_run_details_per_run[idx] if idx < len(gene_run_details_per_run) else {}
            d = (details_map.get(g, {}) or {})
            runs_detail.append(
                {
                    "run_index": run_idx,
                    "rank": ranks_map.get(g),
                    "Narrative": d.get("Narrative", ""),
                    "Rationale": d.get("Rationale", ""),
                }
            )
            if d and not gene_meta["impact_bucket"]:
                gene_meta["impact_bucket"] = d.get("impact_bucket", "") or ""
                gene_meta["effect"] = d.get("effect", "") or ""
        integrated_data_genes.append(
            {
                "gene": g,
                "average_rank": mean_rank,
                "rank_variance": variance,
                "n_runs_seen": run_count,
                "impact_bucket": gene_meta["impact_bucket"],
                "effect": gene_meta["effect"],
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
    <div id="table-container">
      <table id="rank-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Gene</th>
            <th>Average rank</th>
            <th>Rank variance</th>
            <th>Runs seen</th>
            <th>Impact bucket</th>
            <th>Effect</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
    <aside id="detail-panel">
      <div id="detail-title">Select a gene</div>
      <div class="muted">Per-run narrative and rationale details will appear here.</div>
      <table id="detail-table" style="margin-top:10px;">
        <thead>
          <tr><th>Run</th><th>Narrative</th><th>Rationale</th></tr>
        </thead>
        <tbody></tbody>
      </table>
    </aside>
  </main>
  <script>
    const DATA = __DATA_JSON__;
    const tbody = document.querySelector('#rank-table tbody');
    const detailTitle = document.getElementById('detail-title');
    const detailBody = document.querySelector('#detail-table tbody');

    function formatNumber(x) {
      if (x === null || x === undefined || Number.isNaN(Number(x))) return '';
      return Number(x).toFixed(3);
    }

    DATA.genes.forEach((g, idx) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${idx + 1}</td>
        <td>${g.gene || ''}</td>
        <td>${formatNumber(g.average_rank)}</td>
        <td>${formatNumber(g.rank_variance)}</td>
        <td>${g.n_runs_seen}</td>
        <td>${g.impact_bucket || ''}</td>
        <td>${g.effect || ''}</td>
      `;
      tr.addEventListener('click', () => {
        detailTitle.textContent = `Gene: ${g.gene} | avg rank=${formatNumber(g.average_rank)} | var=${formatNumber(g.rank_variance)}`;
        detailBody.innerHTML = '';
        for (const r of g.runs) {
          const rtr = document.createElement('tr');
          rtr.innerHTML = `
            <td>${r.run_index}${r.rank ? ` (rank ${r.rank})` : ''}</td>
            <td>${(r.Narrative || '').replace(/</g,'&lt;')}</td>
            <td>${(r.Rationale || '').replace(/</g,'&lt;')}</td>
          `;
          detailBody.appendChild(rtr);
        }
      });
      tbody.appendChild(tr);
    });
  </script>
</body>
</html>
""".replace("__DATA_JSON__", data_json).replace("{timestamp}", SHOT_TIMESTAMP).replace("{num_runs}", str(num_runs))

    integrated_html_path.write_text(html_template, encoding="utf-8")
    written_paths.append(integrated_html_path)

# Keep legacy output filenames from the final shot for compatibility.
if not last_rank_df.empty:
    csv_path = OUTPUT_DIR / f"llm_variant_ranking_{RUN_ID}_original.csv"
    jsonl_path = OUTPUT_DIR / f"llm_variant_ranking_{RUN_ID}_original.jsonl"
    merged_path = OUTPUT_DIR / f"llm_variant_ranking_merged_{RUN_ID}_original.csv"
    last_rank_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, r in last_rank_df.iterrows():
            f.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
    final_df.to_csv(merged_path, index=False, encoding="utf-8-sig")
    written_paths.extend([csv_path, jsonl_path, merged_path])

print("\nWROTE:")
for p in written_paths:
    print(f"- {p.resolve()}")
print(f"(Input CSV source: {pathlib.Path(CSV_PATH).resolve()})")

# =========================
# Mapping diagnostics (explicit)
# =========================
missing_counts = {
    "ems_label_raw": int(final_df["ems_label_raw"].eq("").sum() if "ems_label_raw" in final_df.columns else 0),
    "qual": int(final_df["qual"].isna().sum() if "qual" in final_df.columns else 0),
    "depth": int(final_df["depth"].isna().sum() if "depth" in final_df.columns else 0)
}
print("\n[Diagnostics] Missing-by-column for key ranking fields:")
print(missing_counts)
