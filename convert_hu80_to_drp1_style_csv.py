#!/usr/bin/env python3
"""
Convert `hu80.csv` (CloudMap3 export style with effect/gene fields as columns)
into a drp1.csv-like table that `multi_agent_with_mapping.py` can consume.

`multi_agent_with_mapping.py` requires an `INFO` column containing an `ANN=...`
payload so it can parse:
  effect  = cols[1]
  impact  = cols[2]
  gene    = cols[3]
  wbgene  = cols[4] (must contain WBGene########)

This script builds:
  INFO = "ANN=<ALT>|<effect>|<impact>|<gene>|<wbgene_id>|transcript|<transcript_id>"
"""

from __future__ import annotations

import argparse
import re
from typing import Optional

import pandas as pd


def _clean_str(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def effect_to_vep_effect(effect_text: str) -> str:
    """
    Map hu80.csv `Effect` strings into VEP-ish tokens so downstream heuristics work.
    The exact strings are designed to trigger substring checks in:
      - effect_severity_score()
      - impact_bucket_from()
    """
    t = (effect_text or "").strip().upper()

    # Common VEP-like targets expected by multi_agent_with_mapping scoring
    if "STOP_GAIN" in t:
        return "stop_gained"
    if "FRAMESHIFT" in t:
        return "frameshift_variant"
    if "MISSENSE" in t:
        return "missense_variant"
    if "INFRAME" in t and "INSER" in t:
        return "inframe_insertion_variant"
    if "INFRAME" in t and "DELET" in t:
        return "inframe_deletion_variant"

    if "SYNONYMOUS" in t:
        return "synonymous_variant"

    # hu80 uses "UPSTREAM: <n> bases" and "DOWNSTREAM: <n> bases"
    if "UPSTREAM" in t:
        return "upstream_gene_variant"
    if "DOWNSTREAM" in t:
        return "downstream_gene_variant"

    # hu80 uses plain "INTRON" in many rows
    if "INTRON" in t:
        return "intron_variant"

    # UTR tokens
    if "UTR" in t or "3_PRIME" in t or "5_PRIME" in t:
        # Multi_agent heuristics look for "utr" substring
        # Normalize everything to a generic UTR label.
        return "UTR_variant"

    # Fallback: keep it as-is, but ensure something recognizable
    # (and avoid semicolons, which would break ANN extraction).
    if t:
        return re.sub(r"[;]+", "", t).lower()
    return "intergenic_region"


def build_ann(alt: str, effect: str, gene: str, wbgene_id: str, transcript_id: str) -> str:
    """
    Build a minimal ANN item with at least 5 '|' columns.
    Indices used by parse_info_fields():
      cols[1] = effect
      cols[2] = impact (we set UNKNOWN)
      cols[3] = gene
      cols[4] = wbgene_id (must contain WBGene########)
    """
    alt = _clean_str(alt)
    gene = _clean_str(gene)
    wbgene_id = _clean_str(wbgene_id)
    transcript_id = _clean_str(transcript_id)
    effect = (effect or "").replace(";", "")

    impact = "UNKNOWN"
    # Keep the ANN payload comma-free; multi_agent splits by comma to take the first.
    return f"{alt}|{effect}|{impact}|{gene}|{wbgene_id}|transcript|{transcript_id}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="hu80.csv", help="Input hu80.csv path")
    ap.add_argument(
        "--output",
        default="hu80_drp1_style.csv",
        help="Output CSV path (drp1-like schema)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input, engine="python")
    df.columns = [c.strip() for c in df.columns]

    required = ["CHROM", "POS", "Reference", "Change", "Quality", "Coverage", "Gene_ID", "Gene_name", "Effect"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in hu80.csv: {missing}")

    # Transcript id column is optional-ish in your hu80.csv export, but we try to use it.
    transcript_col = "Trancript_ID" if "Trancript_ID" in df.columns else ("Transcript_ID" if "Transcript_ID" in df.columns else None)

    # Standardize REF/ALT naming for multi_agent_with_mapping
    df_out = pd.DataFrame()
    df_out["CHROM"] = df["CHROM"]
    df_out["POS"] = df["POS"]
    df_out["REF"] = df["Reference"].astype(str).str.upper()
    df_out["ALT"] = df["Change"].astype(str).str.upper()
    df_out["QUAL"] = pd.to_numeric(df["Quality"], errors="coerce")
    df_out["DP"] = pd.to_numeric(df["Coverage"], errors="coerce")

    # Build INFO with ANN payload parsed by multi_agent_with_mapping.parse_info_fields()
    def _make_info(row) -> str:
        effect_raw = _clean_str(row.get("Effect"))
        gene = _clean_str(row.get("Gene_name"))
        wbgene_id = _clean_str(row.get("Gene_ID"))
        transcript_id = _clean_str(row.get(transcript_col)) if transcript_col else ""
        effect = effect_to_vep_effect(effect_raw)
        ann = build_ann(
            alt=_clean_str(row.get("Change")),
            effect=effect,
            gene=gene,
            wbgene_id=wbgene_id,
            transcript_id=transcript_id,
        )
        # Keep INFO minimal to avoid ';' issues in regex ANN=([^;]+)
        return f"ANN={ann}"

    df_out["INFO"] = df.apply(_make_info, axis=1)

    # Keep row order stable
    df_out.to_csv(args.output, index=False)
    print(f"Wrote {len(df_out):,} rows -> {args.output}")


if __name__ == "__main__":
    main()

