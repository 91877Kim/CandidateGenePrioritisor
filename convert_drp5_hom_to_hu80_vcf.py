#!/usr/bin/env python3
"""
Convert drp5 hom mutant variants.vcf to hu80-style homozygous SNPs VCF format
so mapping_code.py can use it. Output: single sample, FORMAT GT:AD:DP:GQ:PL,
only biallelic SNPs with at least one 1/1 genotype. Headers describe acronyms.
"""
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DRP5_HOM_VCF = SCRIPT_DIR / "drp5 hom mutant variants.vcf"
OUT_VCF = SCRIPT_DIR / "drp5_hom_mutant_variants_hu80format.vcf"

# hu80-style header for homozygous SNPs VCF (mapping_code parse_homozygous_snps)
HU80_HOM_HEADER = """##fileformat=VCFv4.1
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
##source=convert_drp5_hom_to_hu80_vcf (single-sample GT:AD:DP:GQ:PL for mapping_code.py homozygous SNPs VCF)
"""


def parse_info(info_str, key):
    if not info_str or info_str == ".":
        return None
    m = re.search(rf"{re.escape(key)}=([^;\s]+)", info_str)
    return m.group(1) if m else None


def is_biallelic_snp(ref, alt):
    """True if single REF base and single ALT base (SNP)."""
    if not ref or not alt:
        return False
    ref = ref.strip()
    alt = alt.strip()
    if "," in alt:
        return False
    return len(ref) == 1 and len(alt) == 1 and ref.upper() in "ACGT" and alt.upper() in "ACGT"


def main():
    if not DRP5_HOM_VCF.exists():
        print(f"Not found: {DRP5_HOM_VCF}", file=sys.stderr)
        sys.exit(1)

    format_keys = None
    sample_names = []
    n_written = 0

    with open(DRP5_HOM_VCF, "r", encoding="utf-8", errors="replace") as fin, open(
        OUT_VCF, "w", encoding="utf-8"
    ) as out:
        out.write(HU80_HOM_HEADER.strip())
        out.write("\n")

        for line in fin:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                parts = line[1:].split("\t")
                if len(parts) >= 9:
                    format_keys = parts[8].split(":")
                    sample_names = parts[9:] if len(parts) > 9 else []
                out.write("#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	rgSM\n")
                continue

            if format_keys is None:
                continue

            col = line.split("\t")
            if len(col) < 9:
                continue
            chrom, pos, id_, ref, alt, qual, filt, info_str, fmt_str = col[:9]
            samples_col = col[9:] if len(col) > 9 else []

            if not is_biallelic_snp(ref, alt):
                continue
            alt_clean = alt.split(",")[0].strip() if "," in alt else alt.strip()

            by_key = {}
            found_gt_11 = False
            gt_out = "1/1"
            ad_out = "0,0"
            dp_out = "0"
            gq_out = "0"
            pl_out = "0,0,0"

            for si, sample_val in enumerate(samples_col):
                sample_val = sample_val.strip().strip('"')
                if not sample_val or sample_val == ".":
                    continue
                s_parts = sample_val.split(":")
                fmt_parts = fmt_str.split(":")
                for i, k in enumerate(fmt_parts):
                    if i < len(s_parts):
                        by_key[k] = s_parts[i]
                gt = by_key.get("GT", "")
                if gt != "1/1":
                    continue
                found_gt_11 = True
                ad_out = by_key.get("AD", "0,0")
                dp_out = by_key.get("DP", "0")
                gq_out = by_key.get("GQ", "0")
                pl_out = by_key.get("PL", "0,0,0")
                break

            if not found_gt_11:
                continue

            chrom_out = "chrMtDNA" if chrom == "chrM" else chrom
            qual_out = qual if qual else "."
            filt_out = filt if filt else "."
            info_parts = []
            for key in ("AC", "AF", "AN", "DP"):
                val = parse_info(info_str, key)
                if val is not None:
                    info_parts.append(f"{key}={val}")
            info_out = ";".join(info_parts) if info_parts else "."

            out.write(
                f"{chrom_out}\t{pos}\t.\t{ref}\t{alt_clean}\t{qual_out}\t{filt_out}\t{info_out}\tGT:AD:DP:GQ:PL\t{gt_out}:{ad_out}:{dp_out}:{gq_out}:{pl_out}\n"
            )
            n_written += 1

    print(f"Wrote {n_written} homozygous SNP variants to {OUT_VCF}")


if __name__ == "__main__":
    main()
