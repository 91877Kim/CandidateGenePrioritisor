#!/usr/bin/env python3
"""
Ensure the INFO column in a VCF-style CSV is always quoted so that commas inside
INFO (e.g. between ANN annotations) do not break column alignment.

Use when run_prioritising.py / multi_agent_with_mapping.py reports only one gene
from a file like drp1_nonsubstracted_drp1format.csv. Some rows have unquoted INFO
with commas, causing CSV readers to split one row into many columns and corrupt
the INFO cell.

Usage:
  python fix_csv_info_quoting.py drp1_nonsubstracted_drp1format.csv -o drp1_nonsubstracted_fixed.csv
"""
import argparse
import sys
from pathlib import Path

PASS_PREFIX = ",PASS,AB="
FORMAT_MARKER = ",GT:"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Input CSV path (VCF-style with #CHROM,POS,...,INFO,...)")
    ap.add_argument("-o", "--output", default=None, help="Output CSV path (default: overwrite input)")
    args = ap.parse_args()
    inp = Path(args.input)
    out = Path(args.output) if args.output else inp
    if not inp.exists():
        print(f"Error: not found: {inp}", file=sys.stderr)
        sys.exit(1)

    fixed = 0
    lines_out = []
    with open(inp, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                lines_out.append(line)
                continue
            if ",PASS,\"" in line:
                # Already quoted
                lines_out.append(line)
                continue
            start = line.find(PASS_PREFIX)
            if start == -1:
                lines_out.append(line)
                continue
            # INFO runs from after ",PASS," until the comma before ",GT:"
            end_gt = line.rfind(FORMAT_MARKER)
            if end_gt == -1:
                lines_out.append(line)
                continue
            info_start = start + len(PASS_PREFIX)
            info_end = end_gt  # INFO is line[info_start:info_end]
            info_val = line[info_start:info_end].replace('"', '""')
            new_line = (
                line[:info_start] + '"' + info_val + '"' + line[info_end:]
            )
            lines_out.append(new_line)
            fixed += 1

    with open(out, "w", encoding="utf-8") as f:
        f.writelines(lines_out)
    print(f"Wrote {out} ({fixed} rows had INFO field quoted).")


if __name__ == "__main__":
    main()
