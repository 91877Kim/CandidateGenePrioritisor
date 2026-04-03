#!/usr/bin/env python3
"""
Run CloudMap3 mapping pipeline using input paths from mapping_input.txt.

mapping_input.txt format (one path per line, lines starting with # ignored):
  Line 1: Homozygous SNP VCF path
  Line 2: Hawaiian allele-ratio VCF path
"""
from pathlib import Path
import sys
import io

# Avoid Windows console UnicodeEncodeError (gbk) when printing from mapping_code
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    script_dir = Path(__file__).resolve().parent
    mapping_input = script_dir / "mapping_input.txt"

    if not mapping_input.exists():
        print("Error: mapping_input.txt not found.", file=sys.stderr)
        print("Create it with two lines: homo VCF path, ratio VCF path.", file=sys.stderr)
        sys.exit(1)

    with open(mapping_input) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    if len(lines) < 2:
        print("Error: mapping_input.txt must contain at least 2 lines.", file=sys.stderr)
        print("  Line 1: Homozygous SNP VCF path", file=sys.stderr)
        print("  Line 2: Hawaiian allele-ratio VCF path", file=sys.stderr)
        sys.exit(1)

    homo_vcf = Path(lines[0])
    ratio_vcf = Path(lines[1])

    if not homo_vcf.exists():
        print(f"Error: Homozygous VCF not found: {homo_vcf}", file=sys.stderr)
        sys.exit(1)
    if not ratio_vcf.exists():
        print(f"Error: Ratio VCF not found: {ratio_vcf}", file=sys.stderr)
        sys.exit(1)

    print(f"Running CloudMap3 mapping with:")
    print(f"  Homozygous SNPs: {homo_vcf}")
    print(f"  Hawaiian ratios: {ratio_vcf}")
    print()

    # Run from script dir so relative paths in mapping_input.txt resolve
    import os
    os.chdir(script_dir)
    sys.path.insert(0, str(script_dir))
    import mapping_code  # noqa: F401  # runs pipeline on import


if __name__ == "__main__":
    main()
