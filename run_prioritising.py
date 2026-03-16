#!/usr/bin/env python3
"""
Run the prioritisation pipeline from prioritising_input.txt.

Reads prioritising_input.txt for:
  - Lines 1–3: Optional mapping inputs. Leave all three empty to run
    prioritisation without mapping.
    * Lines 1–2: VCF paths (homozygous SNP, Hawaiian ratio) for mapping.
    * Line 3: Optional mapping table path. If non-empty and file exists,
      runs multi_agent_with_mapping.py with that table only. Otherwise runs
      the mapping code on the two VCFs to generate a variant table, then
      runs multi_agent_with_mapping.py with the generated table.
  - Line 4: Main variant CSV path (e.g. drp1.csv).
  - Line 5: Optional user annotation CSV filename in script directory.
  - Line 6: Agent1 prompt (literature evaluation; phenotype + mechanisms).
  - Line 7: Agent2 prompt (reasoning / synthesis for the gene set).
  - Line 8: How many LLM runs on this gene set (integer).

Lines 4–8 are always required; lines 1–3 are only required when using mapping.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PRIORITISING_INPUT = SCRIPT_DIR / "prioritising_input.txt"
MAPPING_INPUT = SCRIPT_DIR / "mapping_input.txt"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
MULTI_AGENT_SCRIPT = SCRIPT_DIR / "multi_agent_with_mapping.py"
RUN_CLOUDMAP = SCRIPT_DIR / "run_cloudmap.py"


def read_prioritising_lines() -> list[str]:
    """Read non-comment lines from prioritising_input.txt (order preserved; empty lines kept)."""
    if not PRIORITISING_INPUT.exists():
        print(f"Error: {PRIORITISING_INPUT} not found.", file=sys.stderr)
        sys.exit(1)
    lines = []
    with open(PRIORITISING_INPUT, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            if line.lstrip().startswith("#"):
                continue
            lines.append(line)
    return lines


def main():
    lines = read_prioritising_lines()
    if len(lines) < 8:
        print(
            "Error: prioritising_input.txt must have at least 8 non-comment lines.",
            file=sys.stderr,
        )
        print(
            "  Lines 1–3: mapping inputs (optional); 4: variant CSV path; "
            "5: annotation CSV; 6: Agent1 prompt; 7: Agent2 prompt; 8: LLM run count.",
            file=sys.stderr,
        )
        sys.exit(1)

    homo_vcf = lines[0].strip()
    ratio_vcf = lines[1].strip()
    mapping_table_path = lines[2].strip()
    variant_csv_path = lines[3].strip()
    annotation_csv = lines[4].strip() if len(lines) > 4 else ""
    agent1_prompt = lines[5].strip() if len(lines) > 5 else ""
    agent2_prompt = lines[6].strip() if len(lines) > 6 else ""
    run_count = lines[7].strip() if len(lines) > 7 else "1"

    # Build stdin tail for multi_agent_with_mapping:
    #   1) variant CSV path
    #   2) annotation CSV path
    #   3) Agent1Prompt
    #   4) Phenotype description (blank -> reuse Agent1Prompt inside multi_agent)
    #   5) Target mechanisms (blank -> reuse Agent1Prompt inside multi_agent)
    #   6) Agent2Prompt (fed into PROMPT)
    #   7) Number of LLM runs
    multi_agent_tail = [
        variant_csv_path,
        annotation_csv,
        agent1_prompt,
        "",
        "",
        agent2_prompt,
        run_count,
    ]

    # Resolve paths relative to script dir
    if homo_vcf and not Path(homo_vcf).is_absolute():
        homo_vcf = str(SCRIPT_DIR / homo_vcf)
    if ratio_vcf and not Path(ratio_vcf).is_absolute():
        ratio_vcf = str(SCRIPT_DIR / ratio_vcf)
    if mapping_table_path and not Path(mapping_table_path).is_absolute():
        mapping_table_path = str(SCRIPT_DIR / mapping_table_path)
    if variant_csv_path and not Path(variant_csv_path).is_absolute():
        variant_csv_path = str(SCRIPT_DIR / variant_csv_path)
    if annotation_csv and not Path(annotation_csv).is_absolute():
        annotation_csv = str(SCRIPT_DIR / annotation_csv)

    skip_mapping = not homo_vcf and not ratio_vcf and not mapping_table_path
    use_existing_table = bool(mapping_table_path) and Path(mapping_table_path).exists()

    if skip_mapping:
        print("Lines 1–3 empty; running prioritisation without mapping.")
        mapping_path_for_agent = ""
    elif use_existing_table:
        print(f"Using existing mapping table: {mapping_table_path}")
        mapping_path_for_agent = mapping_table_path
    else:
        if not mapping_table_path:
            print("No mapping table path provided; will run mapping on the two VCFs.")
        else:
            print(f"Mapping table not found or empty: {mapping_table_path}; will run mapping.")
        if not homo_vcf or not Path(homo_vcf).exists():
            print(f"Error: Homozygous VCF not found: {homo_vcf}", file=sys.stderr)
            sys.exit(1)
        if not ratio_vcf or not Path(ratio_vcf).exists():
            print(f"Error: Hawaiian ratio VCF not found: {ratio_vcf}", file=sys.stderr)
            sys.exit(1)
        # Write mapping_input.txt for run_cloudmap
        with open(MAPPING_INPUT, "w", encoding="utf-8") as f:
            f.write(homo_vcf + "\n")
            f.write(ratio_vcf + "\n")
        print(f"Running mapping: {RUN_CLOUDMAP}")
        result = subprocess.run(
            [sys.executable, str(RUN_CLOUDMAP)],
            cwd=str(SCRIPT_DIR),
        )
        if result.returncode != 0:
            print("Mapping failed.", file=sys.stderr)
            sys.exit(result.returncode)
        # Find generated variant table (e.g. outputs/mapping_interval_chrIII_variants.csv)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        generated = sorted(OUTPUTS_DIR.glob("mapping_interval_*_variants.csv"))
        if not generated:
            print("Error: No mapping_interval_*_variants.csv found in outputs/ after mapping.", file=sys.stderr)
            sys.exit(1)
        mapping_path_for_agent = str(generated[0])
        print(f"Using generated mapping table: {mapping_path_for_agent}")

    # Build stdin for multi_agent_with_mapping: first line = mapping path, then lines 4–9 from file
    stdin_lines = [mapping_path_for_agent] + multi_agent_tail
    stdin_text = "\n".join(stdin_lines) + "\n"

    print(f"Running {MULTI_AGENT_SCRIPT}")
    result = subprocess.run(
        [sys.executable, str(MULTI_AGENT_SCRIPT)],
        cwd=str(SCRIPT_DIR),
        input=stdin_text,
        encoding="utf-8",
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
