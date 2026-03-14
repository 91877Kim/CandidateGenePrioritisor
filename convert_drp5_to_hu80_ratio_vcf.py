#!/usr/bin/env python3
"""
Convert drp5_HA_SNP_positions_for_cloudmap.vcf (freeBayes) to hu80-style ratio VCF
so mapping_code.py can use it. Output: GT:AD:DP:GQ:PL FORMAT; headers describe acronyms.
Uses only standard library (no cyvcf2 required).
"""
import sys
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DRP5_VCF = SCRIPT_DIR / "drp5_HA_SNP_positions_for_cloudmap.vcf"
OUT_VCF = SCRIPT_DIR / "drp5_HA_SNP_positions_hu80format.vcf"

# hu80-style header: FORMAT/INFO descriptions so acronyms are documented
HU80_HEADER = """##fileformat=VCFv4.1
##FORMAT=<ID=AD,Number=.,Type=Integer,Description="Allelic depths for the ref and alt alleles in the order listed">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth (reads with MQ=255 or with bad mates are filtered)">
##FORMAT=<ID=GQ,Number=1,Type=Float,Description="Genotype Quality">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Normalized, Phred-scaled likelihoods for genotypes as defined in the VCF specification">
##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in genotypes, for each ALT allele, in the same order as listed">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency, for each ALT allele, in the same order as listed">
##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles in called genotypes">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth; some reads may have been filtered">
##contig=<ID=chrI,length=15072434>
##contig=<ID=chrII,length=15279421>
##contig=<ID=chrIII,length=13783801>
##contig=<ID=chrIV,length=17493829>
##contig=<ID=chrMtDNA,length=13794>
##contig=<ID=chrV,length=20924180>
##contig=<ID=chrX,length=17718942>
##source=convert_drp5_to_hu80_ratio_vcf (from freeBayes; GT:AD:DP:GQ:PL for mapping_code.py ratio VCF)
"""


def parse_info(info_str, key):
    """Get first value for key from INFO string (e.g. AF=0.5 or AC=1)."""
    if not info_str or info_str == ".":
        return None
    m = re.search(rf"{re.escape(key)}=([^;\s]+)", info_str)
    return m.group(1) if m else None


def gl_to_pl(gl_str):
    """Convert log10 genotype likelihood string (e.g. '-34.04,0,-34.27') to Phred PL."""
    if not gl_str or gl_str == ".":
        return "0,0,0"
    try:
        parts = gl_str.split(",")
        if len(parts) < 3:
            return "0,0,0"
        pl = [max(0, int(round(-10 * float(p)))) for p in parts[:3]]
        return ",".join(map(str, pl))
    except (TypeError, ValueError):
        return "0,0,0"


def main():
    if not DRP5_VCF.exists():
        print(f"Not found: {DRP5_VCF}", file=sys.stderr)
        sys.exit(1)

    format_keys = None
    sample_name = "SAMPLE"
    n_written = 0

    with open(DRP5_VCF, "r", encoding="utf-8", errors="replace") as fin, open(
        OUT_VCF, "w", encoding="utf-8"
    ) as out:
        out.write(HU80_HEADER.strip())
        out.write("\n")

        for line in fin:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                # #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	RGSM
                parts = line[1:].split("\t")
                if len(parts) >= 10:
                    format_keys = parts[8].split(":")
                    sample_name = parts[9]
                out.write(f"#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	{sample_name}\n")
                continue

            if format_keys is None:
                continue
            col = line.split("\t")
            if len(col) < 10:
                continue
            chrom, pos, id_, ref, alt, qual, filt, info_str, fmt_str, sample_str = (
                col[0], col[1], col[2], col[3], col[4], col[5], col[6], col[7], col[8], col[9]
            )
            if len(ref) != 1 or (alt and len(alt) != 1):
                continue
            if not alt or alt == ".":
                continue

            sample_parts = sample_str.split(":")
            fmt_parts = fmt_str.split(":")
            by_key = {}
            for i, k in enumerate(fmt_parts):
                if i < len(sample_parts):
                    by_key[k] = sample_parts[i]

            gt = by_key.get("GT", "0/0")
            ad_str = by_key.get("AD")
            dp_str = by_key.get("DP")
            ro_str = by_key.get("RO")
            ao_str = by_key.get("AO")
            gl_str = by_key.get("GL")

            ref_c, alt_c = 0, 0
            if ad_str and "," in ad_str:
                a, b = ad_str.split(",", 1)
                try:
                    ref_c = int(a)
                    alt_c = int(sum(int(x) for x in b.split(",")))
                except ValueError:
                    pass
            if ref_c == 0 and alt_c == 0 and ro_str is not None and ao_str is not None:
                try:
                    ref_c = int(ro_str)
                    alt_c = int(ao_str)
                except ValueError:
                    pass
            if ref_c == 0 and alt_c == 0:
                continue

            dp = ref_c + alt_c
            if dp_str:
                try:
                    dp = int(dp_str)
                except ValueError:
                    pass

            pl_str = gl_to_pl(gl_str) if gl_str else gl_to_pl(None)
            gq = 99
            if gl_str:
                try:
                    gl_vals = [float(x) for x in gl_str.split(",")[:3]]
                    if len(gl_vals) >= 3:
                        pl_vals = [max(0, int(round(-10 * x))) for x in gl_vals]
                        best = min(pl_vals)
                        second = sorted(pl_vals)[1]
                        gq = min(99, max(0, second - best))
                except (ValueError, IndexError):
                    pass

            info_parts = []
            for key in ("AC", "AF", "AN", "DP"):
                val = parse_info(info_str, key)
                if val is not None:
                    info_parts.append(f"{key}={val}")
            info_out = ";".join(info_parts) if info_parts else "."

            chrom_out = "chrMtDNA" if chrom == "chrM" else chrom
            qual_out = qual if qual else "."
            filt_out = filt if filt else "."
            alt_out = alt.split(",")[0] if "," in alt else alt

            out.write(
                f"{chrom_out}\t{pos}\t.\t{ref}\t{alt_out}\t{qual_out}\t{filt_out}\t{info_out}\tGT:AD:DP:GQ:PL\t{gt}:{ref_c},{alt_c}:{dp}:{gq}:{pl_str}\n"
            )
            n_written += 1

    print(f"Wrote {n_written} variants to {OUT_VCF}")


if __name__ == "__main__":
    main()
