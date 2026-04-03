#@title CloudMap-style BSA Classifier + Extra Features + Bootstrap & Permutation CIs (single block)

# =================== Install deps (run in Colab: !pip install ...) ===================
# !pip -q install "xgboost>=2.0,<3" "lightgbm>=4.1,<5" pyarrow tqdm cyvcf2

# =================== Imports & environment hardening ===================
import os, random, warnings, unittest, itertools, math
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb

from scipy.ndimage import gaussian_filter1d
from scipy.stats import binned_statistic
try:
    from cyvcf2 import VCF
    _USE_CYVCF2 = True
    pysam = None
except Exception:
    VCF = None
    _USE_CYVCF2 = False
    try:
        import pysam
    except Exception:
        pysam = None

# ---- Avoid Drive FUSE issues: stay in /content and disable joblib multiprocessing ----
try:
    os.chdir("/content")
except Exception:
    pass
os.environ["JOBLIB_TEMP_FOLDER"] = "/content"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# Optional GPU for XGBoost (enable GPU runtime in Colab, then set True)
USE_GPU = False
XGB_DEVICE_KW = {"tree_method": "hist"}
if USE_GPU:
    XGB_DEVICE_KW.update({"device": "cuda"})

# =================== Mount Drive and set HOME_DIR ===================
try:
    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=True)
except Exception:
    pass

_script_dir = Path(__file__).resolve().parent
HOME_DIR = str(_script_dir)
os.makedirs(HOME_DIR, exist_ok=True)

# =================== File inputs (defaults; overridden by mapping_input.txt if present) ===================
_base = _script_dir.parent
HOMO_VCF = str(_base / "drp5_hom_mutant_variants_hu80format.vcf")
RATIO_VCF = str(_base / "drp5_HA_SNP_positions_hu80format.vcf")
_mapping_input = _script_dir / "mapping_input.txt"
TARGET_CHROM_RAW = ""  # Optional third line: e.g. X / III / chrX
if _mapping_input.exists():
    with open(_mapping_input) as _f:
        _lines = [ln.strip() for ln in _f if ln.strip() and not ln.strip().startswith("#")]
    if len(_lines) >= 2:
        HOMO_VCF, RATIO_VCF = _lines[0], _lines[1]
    if len(_lines) >= 3:
        TARGET_CHROM_RAW = _lines[2]

# =================== Causal variant (UPDATED) ===================
CAUSAL_CHR_LABEL = "chrIII"
CAUSAL_POS = 2_401_031

# =================== Helpers ===================
def normalize_chr(chrom):
    """Normalize chromosome labels to C. elegans roman numerals (I, II, III, IV, V, X)."""
    s = str(chrom).strip()
    s = s.replace("Chromosome", "").replace("chromosome", "")
    s = s.replace("CHR","").replace("chr","").replace("Chr","")
    s = s.strip()
    arabic_to_roman = {'1':'I','2':'II','3':'III','4':'IV','5':'V','10':'X'}
    roman_set = {'I','II','III','IV','V','X'}
    if s in roman_set: return s
    if s in arabic_to_roman: return arabic_to_roman[s]
    su = s.upper()
    return su if su in roman_set else su

def exists_or_raise(p):
    if not Path(p).exists():
        raise FileNotFoundError(
            f"File not found: {p}\n"
            f"-> Make sure Drive is mounted and the path is correct."
        )

def is_ems_transition(ref, alt):
    """EMS mutations are G/C -> A/T transitions (i.e., G->A or C->T)."""
    if ref is None or alt is None: return False
    r = str(ref).upper(); a = str(alt).upper()
    return (r == 'G' and a == 'A') or (r == 'C' and a == 'T')

def is_transition(ref, alt):
    """Transitions: A<->G, C<->T."""
    if ref is None or alt is None: return False
    r = str(ref).upper(); a = str(alt).upper()
    return (r, a) in {('A','G'),('G','A'),('C','T'),('T','C')}

def longest_run(bool_array):
    """Return longest consecutive True run length."""
    if len(bool_array)==0: return 0
    b = np.asarray(bool_array, dtype=int)
    d = np.diff(np.r_[0, b, 0])
    starts = np.where(d==1)[0]
    ends   = np.where(d==-1)[0]
    if len(starts)==0: return 0
    return int(np.max(ends - starts))

def nearest_ratio_for_positions(ratio_df_chr, query_pos, tol_bp=1000):
    """
    For an array-like of positions on one chromosome, return nearest ratio values and distances.
    Returns: (ratios, nearest_pos, dist_bp, src) where src in {'exact','nearest','none'}
    """
    if len(ratio_df_chr) == 0:
        return [np.nan]*len(query_pos), [np.nan]*len(query_pos), [np.nan]*len(query_pos), ['none']*len(query_pos)
    pos_arr = ratio_df_chr['POS'].values
    rat_arr = ratio_df_chr['RATIO'].values
    out_ratio, out_pos, out_dist, out_src = [], [], [], []
    for p in query_pos:
        idx = np.searchsorted(pos_arr, p)
        candidates = []
        if idx < len(pos_arr): candidates.append((abs(pos_arr[idx]-p), idx))
        if idx > 0:            candidates.append((abs(pos_arr[idx-1]-p), idx-1))
        if candidates:
            d, j = min(candidates, key=lambda x: x[0])
            if d == 0:
                out_ratio.append(float(rat_arr[j])); out_pos.append(int(pos_arr[j])); out_dist.append(0); out_src.append('exact')
            elif d <= tol_bp:
                out_ratio.append(float(rat_arr[j])); out_pos.append(int(pos_arr[j])); out_dist.append(int(d)); out_src.append('nearest')
            else:
                out_ratio.append(np.nan); out_pos.append(np.nan); out_dist.append(np.nan); out_src.append('none')
        else:
            out_ratio.append(np.nan); out_pos.append(np.nan); out_dist.append(np.nan); out_src.append('none')
    return out_ratio, out_pos, out_dist, out_src

def debug_causal_presence(homo_vcf, ratio_vcf, causal_chr_label, causal_pos, window_bp=2000):
    """
    Explain why the causal site may be absent from the homozygous SNP table.
    Reports exact/nearby entries in both VCFs around the causal coordinate.
    """
    if not _USE_CYVCF2:
        print("\n[Diagnostics] Skipped (requires cyvcf2).")
        return
    print("\n[Diagnostics] Checking causal site presence in VCFs...")
    c = normalize_chr(causal_chr_label)

    # Homozygous VCF (exact and nearby)
    homo_found_exact = False
    homo_nearby = []
    for v in VCF(homo_vcf):
        if not v.is_snp: continue
        chrom = normalize_chr(v.CHROM)
        if chrom != c: continue
        pos = int(v.POS)
        if pos == causal_pos:
            homo_found_exact = True
            gt = None
            if len(v.samples) > 0 and v.genotypes is not None and v.genotypes[0] is not None and len(v.genotypes[0]) >= 2:
                gt = f"{v.genotypes[0][0]}/{v.genotypes[0][1]}"
            print(f"  Homo VCF: FOUND exact at {c}:{pos:,}  REF={v.REF} ALT={v.ALT[0] if v.ALT else None} GT={gt} QUAL={v.QUAL} DP={v.INFO.get('DP')}")
        elif abs(pos - causal_pos) <= window_bp:
            homo_nearby.append((pos, v.REF, v.ALT, v.QUAL, v.INFO.get('DP')))
    if not homo_found_exact:
        print(f"  Homo VCF: NOT found at exact position {c}:{causal_pos:,}.")
        if homo_nearby:
            print(f"  Homo VCF: Nearby variants within +/- {window_bp} bp:")
            for pos, REF, ALT, QUAL, DP in sorted(homo_nearby, key=lambda x: abs(x[0]-causal_pos))[:6]:
                print(f"    {c}:{pos:,}  REF={REF} ALT={ALT[0] if ALT else None} QUAL={QUAL} DP={DP}")
        else:
            print(f"  Homo VCF: No nearby homozygous SNPs within +/- {window_bp} bp.")

    # Ratio VCF (exact and nearest)
    ratio_found_exact = False
    nearest = None
    for v in VCF(ratio_vcf):
        if not v.is_snp: continue
        chrom = normalize_chr(v.CHROM)
        if chrom != c: continue
        pos = int(v.POS)
        if pos == causal_pos:
            ratio_found_exact = True
            ratio_val = None; depth = None
            try:
                AD = v.format('AD')
                if AD is not None and len(AD)>0 and AD[0] is not None and len(AD[0])>=2:
                    refc=float(AD[0][0] or 0.0); altc=float(sum([a for a in AD[0][1:] if a is not None])); total=refc+altc
                    ratio_val = altc/total if total>0 else None; depth=int(total)
            except Exception:
                pass
            if ratio_val is None:
                try:
                    AO=v.format('AO'); RO=v.format('RO')
                    if AO is not None and RO is not None and len(AO)>0 and len(RO)>0:
                        refc=float(RO[0] or 0.0); altc=float(sum([a for a in AO[0] if a is not None])); total=refc+altc
                        ratio_val = altc/total if total>0 else None; depth=int(total)
                except Exception:
                    pass
            if ratio_val is None:
                af = None
                try:
                    AF = v.format('AF')
                    if AF is not None and len(AF)>0 and len(AF[0])>=1: af = float(AF[0][0])
                except Exception:
                    af = v.INFO.get('AF')
                    af = float(af) if af is not None else None
                if af is not None: ratio_val = max(0.0, min(1.0, af))
            dp_info = v.INFO.get('DP')
            if depth is None and dp_info is not None: depth=int(dp_info)
            print(f"  Ratio VCF: FOUND exact at {c}:{pos:,}  Hawaiian_ratio={ratio_val:.3f}  DP={depth}")
        else:
            d = abs(pos - causal_pos)
            if nearest is None or d < nearest[0]:
                rv = None
                af = v.INFO.get('AF')
                if af is not None:
                    try: rv = float(af)
                    except: rv = None
                nearest = (d, pos, rv)
    if not ratio_found_exact:
        if nearest is not None:
            d, pos, rv = nearest
            print(f"  Ratio VCF: No exact site at {c}:{causal_pos:,}. Nearest at {c}:{pos:,} (Δ={d} bp), AF≈{rv if rv is not None else 'NA'}.")
        else:
            print("  Ratio VCF: No sites on that chromosome (unexpected).")

# =================== Classifier ===================
class BulkedSegregantClassifier:
    """Classifier for bulked-segregant mapping in C. elegans (mutant vs Hawaiian)."""
    def __init__(self, window_size=250_000, step_size=50_000, classifier_type='xgboost'):
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.classifier_type = classifier_type
        self.feature_names = []
        self.classifier = None
        self.trained = False
        self.optimal_threshold = 0.5
        self._last_homo_df = None
        self._last_ratio_df = None

    # ----- VCF parsing -----
    def parse_homozygous_snps(self, vcf_path):
        """Parse homozygous SNPs VCF (mutant SNPs with Hawaiian SNPs subtracted)."""
        if _USE_CYVCF2:
            return self._parse_homozygous_snps_cyvcf2(vcf_path)
        if pysam is not None:
            return self._parse_homozygous_snps_pysam(vcf_path)
        return self._parse_homozygous_snps_native(vcf_path)

    def _parse_homozygous_snps_cyvcf2(self, vcf_path):
        rows = []
        vcf = VCF(vcf_path)
        has_sample = len(vcf.samples) > 0
        for v in vcf:
            if not v.is_snp: continue
            if v.ALT is None or len(v.ALT) != 1: continue
            if has_sample:
                gts = v.genotypes[0] if v.genotypes is not None else None
                if gts is None or len(gts) < 2: continue
                if not ((gts[0] == 1) and (gts[1] == 1)): continue
            chrom = normalize_chr(v.CHROM); pos = int(v.POS)
            ref = (v.REF or "").upper(); alt = (v.ALT[0] or "").upper()
            qual = float(v.QUAL) if v.QUAL is not None else 30.0
            dp = None
            try:
                dp_fmt = v.format('DP')
                if dp_fmt is not None and len(dp_fmt) > 0: dp = int(dp_fmt[0][0])
            except Exception: dp = None
            if dp is None: dp = int(v.INFO.get('DP') or 20)
            rows.append({'CHROM': chrom, 'POS': pos, 'REF': ref, 'ALT': alt, 'QUAL': qual, 'DP': dp})
        df = pd.DataFrame(rows)
        if df.empty: raise ValueError(f"No usable SNPs parsed from {vcf_path}.")
        return df

    def _parse_homozygous_snps_pysam(self, vcf_path):
        rows = []
        vcf = pysam.VariantFile(vcf_path)
        samples = list(vcf.header.samples)
        has_sample = len(samples) > 0
        for rec in vcf:
            if len(rec.ref) != 1 or not rec.alts or len(rec.alts) != 1: continue
            if has_sample:
                s = samples[0]
                gt = rec.samples[s].get('GT')
                if gt is None or None in gt: continue
                if not (gt[0] == 1 and gt[1] == 1): continue
            chrom = normalize_chr(rec.chrom); pos = int(rec.pos); ref = (rec.ref or "").upper(); alt = (rec.alts[0] or "").upper()
            qual = float(rec.qual) if rec.qual is not None else 30.0
            dp = rec.samples[samples[0]].get('DP') if has_sample else None
            if dp is None: dp = rec.info.get('DP')
            dp = int(dp) if dp is not None else 20
            rows.append({'CHROM': chrom, 'POS': pos, 'REF': ref, 'ALT': alt, 'QUAL': qual, 'DP': dp})
        vcf.close()
        df = pd.DataFrame(rows)
        if df.empty: raise ValueError(f"No usable SNPs parsed from {vcf_path}.")
        return df

    def _parse_homozygous_snps_native(self, vcf_path):
        """Pure-Python VCF parser fallback (no cyvcf2/pysam)."""
        rows = []
        with open(vcf_path) as f:
            fmt_idx = None
            for line in f:
                if line.startswith('##'): continue
                if line.startswith('#CHROM'):
                    parts = line.strip().split('\t')
                    fmt_idx = parts.index('FORMAT') if 'FORMAT' in parts else None
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 9: continue
                chrom, pos_s, ref, alt, qual_s = parts[0], parts[1], parts[3], parts[4], parts[5]
                if len(ref) != 1 or ',' in alt: continue  # SNP only, single alt
                alts = alt.split(',')
                if len(alts) != 1: continue
                if fmt_idx is not None and len(parts) > fmt_idx + 1:
                    fmt_parts = parts[fmt_idx].split(':')
                    samp = parts[fmt_idx + 1]
                    gt_idx = fmt_parts.index('GT') if 'GT' in fmt_parts else -1
                    ad_idx = fmt_parts.index('AD') if 'AD' in fmt_parts else -1
                    dp_idx = fmt_parts.index('DP') if 'DP' in fmt_parts else -1
                    samp_parts = samp.split(':')
                    if gt_idx >= 0 and len(samp_parts) > gt_idx:
                        gt = samp_parts[gt_idx]
                        if gt != '1/1' and gt != '1|1': continue
                qual = float(qual_s) if qual_s != '.' else 30.0
                dp = 20
                if fmt_idx is not None and len(parts) > fmt_idx + 1:
                    fmt_parts = parts[fmt_idx].split(':')
                    samp_parts = parts[fmt_idx + 1].split(':')
                    if 'DP' in fmt_parts:
                        i = fmt_parts.index('DP')
                        if i < len(samp_parts) and samp_parts[i].isdigit(): dp = int(samp_parts[i])
                rows.append({'CHROM': normalize_chr(chrom), 'POS': int(pos_s), 'REF': ref.upper(), 'ALT': alts[0].upper(), 'QUAL': qual, 'DP': dp})
        df = pd.DataFrame(rows)
        if df.empty: raise ValueError(f"No usable SNPs parsed from {vcf_path}.")
        return df

    def parse_allele_ratios(self, vcf_path):
        """Parse allele ratios at Hawaiian SNP sites VCF. RATIO from AD/AO/RO/AF; DEPTH from DP/AD."""
        if _USE_CYVCF2:
            return self._parse_allele_ratios_cyvcf2(vcf_path)
        if pysam is not None:
            return self._parse_allele_ratios_pysam(vcf_path)
        return self._parse_allele_ratios_native(vcf_path)

    def _parse_allele_ratios_cyvcf2(self, vcf_path):
        rows = []
        vcf = VCF(vcf_path)
        has_sample = len(vcf.samples) > 0
        for v in vcf:
            if not v.is_snp: continue
            if v.ALT is None or len(v.ALT) < 1: continue
            chrom = normalize_chr(v.CHROM); pos = int(v.POS)
            ratio = None; depth = None
            try:
                AD = v.format('AD') if has_sample else None
                if AD is not None and len(AD) > 0 and AD[0] is not None and len(AD[0]) >= 2:
                    ref_count = float(AD[0][0] or 0.0)
                    alt_count = float(sum([a for a in AD[0][1:] if a is not None]))
                    total = ref_count + alt_count
                    if total > 0: ratio = alt_count / total; depth = int(total)
            except Exception: pass
            if ratio is None:
                try:
                    AO = v.format('AO') if has_sample else None
                    RO = v.format('RO') if has_sample else None
                    if AO and RO and len(AO) > 0 and len(RO) > 0:
                        ref_count = float(RO[0] or 0.0)
                        alt_count = float(sum([a for a in AO[0] if a is not None]))
                        total = ref_count + alt_count
                        if total > 0: ratio = alt_count / total; depth = int(total)
                except Exception: pass
            if ratio is None:
                af = None
                try:
                    AF_fmt = v.format('AF') if has_sample else None
                    if AF_fmt and len(AF_fmt) > 0 and len(AF_fmt[0]) >= 1: af = float(AF_fmt[0][0])
                except: af = v.INFO.get('AF'); af = float(af) if af is not None else None
                if af is not None: ratio = float(max(0.0, min(1.0, af)))
            try:
                DP = v.format('DP') if has_sample else None
                if DP and len(DP) > 0: depth = int(DP[0][0])
            except: pass
            if depth is None: depth = int(v.INFO.get('DP')) if v.INFO.get('DP') is not None else 100
            if ratio is None: continue
            rows.append({'CHROM': chrom, 'POS': pos, 'RATIO': float(ratio), 'DEPTH': int(depth)})
        df = pd.DataFrame(rows)
        if df.empty: raise ValueError(f"No usable ratio sites parsed from {vcf_path}.")
        return df

    def _parse_allele_ratios_pysam(self, vcf_path):
        rows = []
        vcf = pysam.VariantFile(vcf_path)
        samples = list(vcf.header.samples)
        has_sample = len(samples) > 0
        for rec in vcf:
            if len(rec.ref) != 1 or not rec.alts: continue
            chrom = normalize_chr(rec.chrom); pos = int(rec.pos)
            ratio = None; depth = None
            if has_sample:
                s = samples[0]
                ad = rec.samples[s].get('AD')
                if ad is not None and len(ad) >= 2:
                    refc = float(ad[0] or 0); altc = float(sum(ad[1:]) if len(ad) > 1 else 0)
                    total = refc + altc
                    if total > 0: ratio = altc / total; depth = int(total)
                if ratio is None:
                    ao = rec.samples[s].get('AO'); ro = rec.samples[s].get('RO')
                    if ao is not None and ro is not None:
                        refc = float(ro); altc = float(sum(ao) if isinstance(ao, (list, tuple)) else ao)
                        total = refc + altc
                        if total > 0: ratio = altc / total; depth = int(total)
                if ratio is None:
                    af = rec.samples[s].get('AF') or rec.info.get('AF')
                    if af is not None:
                        av = af[0] if isinstance(af, (tuple, list)) else af
                        ratio = max(0.0, min(1.0, float(av)))
                if depth is None: depth = rec.samples[s].get('DP')
            if not has_sample and ratio is None:
                af = rec.info.get('AF')
                if af is not None:
                    av = af[0] if isinstance(af, (tuple, list)) else af
                    ratio = max(0.0, min(1.0, float(av)))
            if depth is None: depth = rec.info.get('DP')
            depth = int(depth) if depth is not None else 100
            if ratio is None: continue
            rows.append({'CHROM': chrom, 'POS': pos, 'RATIO': float(ratio), 'DEPTH': int(depth)})
        vcf.close()
        df = pd.DataFrame(rows)
        if df.empty: raise ValueError(f"No usable ratio sites parsed from {vcf_path}.")
        return df

    def _parse_allele_ratios_native(self, vcf_path):
        """Pure-Python VCF parser fallback for ratio VCF (AD or AF)."""
        rows = []
        with open(vcf_path) as f:
            fmt_idx = None
            for line in f:
                if line.startswith('##'): continue
                if line.startswith('#CHROM'):
                    parts = line.strip().split('\t')
                    fmt_idx = parts.index('FORMAT') if 'FORMAT' in parts else None
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 8: continue
                chrom, pos_s, ref, alt, info = parts[0], parts[1], parts[3], parts[4], parts[7]
                if len(ref) != 1 or not alt or alt == '.': continue
                ratio = None; depth = 100
                if fmt_idx is not None and len(parts) > fmt_idx + 1:
                    fmt_parts = parts[fmt_idx].split(':')
                    samp_parts = parts[fmt_idx + 1].split(':')
                    if 'AD' in fmt_parts:
                        i = fmt_parts.index('AD')
                        if i < len(samp_parts):
                            ad_vals = samp_parts[i].split(',')
                            if len(ad_vals) >= 2 and all(x.isdigit() for x in ad_vals[:2]):
                                refc = float(ad_vals[0]); altc = float(ad_vals[1])
                                total = refc + altc
                                if total > 0: ratio = altc / total; depth = int(total)
                    if ratio is None and 'AF' in fmt_parts:
                        i = fmt_parts.index('AF')
                        if i < len(samp_parts):
                            try: ratio = float(samp_parts[i].split(',')[0]); ratio = max(0, min(1, ratio))
                            except: pass
                    if 'DP' in fmt_parts and depth == 100:
                        i = fmt_parts.index('DP')
                        if i < len(samp_parts) and samp_parts[i].isdigit(): depth = int(samp_parts[i])
                if ratio is None and 'AF=' in info:
                    for kv in info.split(';'):
                        if kv.startswith('AF='):
                            try: ratio = float(kv.split('=')[1].split(',')[0]); ratio = max(0, min(1, ratio))
                            except: pass
                            break
                if ratio is None: continue
                rows.append({'CHROM': normalize_chr(chrom), 'POS': int(pos_s), 'RATIO': float(ratio), 'DEPTH': int(depth)})
        df = pd.DataFrame(rows)
        if df.empty: raise ValueError(f"No usable ratio sites parsed from {vcf_path}.")
        return df

    # ----- features -----
    def _ensure_feature_template(self):
        if self.feature_names:
            return
        dummy_h = pd.DataFrame({'CHROM':['I'],'POS':[1000],'REF':['A'],'ALT':['G'],'QUAL':[30],'DP':[20]})
        dummy_r = pd.DataFrame({'CHROM':['I'],'POS':[1000],'RATIO':[0.5],'DEPTH':[50]})
        f = self.extract_window_features(dummy_h, dummy_r, 'I', 0, 250_000)
        self.feature_names = list(f.keys())

    def extract_window_features(self, homo_snps_df, allele_ratios_df, chrom, start, end):
        """
        Enriched feature set: homozygous SNP stats (+EMS, Ti/Tv, QUAL summaries),
        allele-ratio stats (+depth-weighted fractions, IQR/MAD, longest low run, inter-marker gaps),
        and combined indicators.
        """
        features = {}
        window_bp = max(1, end - start)
        window_kb = window_bp / 1000.0

        ws = homo_snps_df[(homo_snps_df['CHROM']==chrom) & (homo_snps_df['POS']>=start) & (homo_snps_df['POS']<end)]
        wr = allele_ratios_df[(allele_ratios_df['CHROM']==chrom) & (allele_ratios_df['POS']>=start) & (allele_ratios_df['POS']<end)]

        # mutant-specific homozygous SNPs -----------------------------
        features['n_homo_snps']      = len(ws)
        features['homo_snp_density'] = len(ws) / window_kb
        if len(ws) > 0:
            # QUAL stats
            q = ws['QUAL'].astype(float).values
            features['mean_snp_qual'] = float(np.mean(q))
            features['std_snp_qual']  = float(np.std(q, ddof=0))
            features['min_snp_qual']  = float(np.min(q))
            features['median_snp_qual'] = float(np.median(q))
            q25, q75 = np.percentile(q, [25,75])
            features['iqr_snp_qual']  = float(q75 - q25)
            features['sum_snp_qual']  = float(np.sum(q))
            # spacing
            if len(ws) > 1:
                pos = np.sort(ws['POS'].values)
                d = np.diff(pos)
                features['mean_snp_spacing'] = float(np.mean(d))
                features['std_snp_spacing']  = float(np.std(d))
                features['min_snp_spacing']  = float(np.min(d))
                features['max_snp_spacing']  = float(np.max(d))
                expected = window_bp / max(1, len(ws))
                features['snp_clustering']   = float(np.std(d) / (expected + 1e-9))
                sd = np.sort(d); n = len(sd); idx = np.arange(1, n+1)
                gini = (2*np.sum(idx*sd))/(n*np.sum(sd)+1e-9) - (n+1)/n
                features['snp_gini']         = float(gini)
            else:
                for k in ['mean_snp_spacing','std_snp_spacing','min_snp_spacing','max_snp_spacing','snp_clustering','snp_gini']:
                    features[k] = 0.0
            # EMS & Ti/Tv
            ems_mask = ws.apply(lambda r: is_ems_transition(r['REF'], r['ALT']), axis=1).values
            ti_mask  = ws.apply(lambda r: is_transition(r['REF'], r['ALT']), axis=1).values
            ems_count = int(np.sum(ems_mask))
            ti_count  = int(np.sum(ti_mask))
            tv_count  = int(len(ws) - ti_count)
            features['ems_count'] = ems_count
            features['ems_fraction'] = float(ems_count / len(ws)) if len(ws)>0 else 0.0
            features['ti_count'] = ti_count
            features['tv_count'] = tv_count
            features['ti_tv_ratio'] = float(ti_count / (tv_count + 1e-9))
        else:
            for k in ['mean_snp_qual','std_snp_qual','min_snp_qual','median_snp_qual','iqr_snp_qual','sum_snp_qual',
                      'mean_snp_spacing','std_snp_spacing','min_snp_spacing','max_snp_spacing','snp_clustering','snp_gini',
                      'ems_count','ems_fraction','ti_count','tv_count','ti_tv_ratio']:
                features[k] = 0.0

        # allele ratios (Hawaiian sites) ------------------------------
        features['n_ratio_sites']      = len(wr)
        features['ratio_site_density'] = len(wr)/window_kb
        if len(wr)>0:
            ratios = wr['RATIO'].astype(float).values
            mu = float(np.mean(ratios))
            sd = float(np.std(ratios, ddof=0))
            features['mean_ratio']   = mu
            features['median_ratio'] = float(np.median(ratios))
            features['std_ratio']    = sd
            features['min_ratio']    = float(np.min(ratios))
            features['max_ratio']    = float(np.max(ratios))
            features['ratio_range']  = features['max_ratio'] - features['min_ratio']
            features['ratio_skew']   = float(np.mean((ratios-mu)**3) / ((sd**3)+1e-10))
            features['ratio_kurtosis']= float(np.mean((ratios-mu)**4) / ((sd**4)+1e-10) - 3.0)
            # Bins and weighted bins
            low  = ratios < 0.25
            mid  = (ratios>=0.25) & (ratios<0.75)
            high = ratios >= 0.75
            features['n_low_ratio']  = int(np.sum(low))
            features['n_mid_ratio']  = int(np.sum(mid))
            features['n_high_ratio'] = int(np.sum(high))
            features['frac_low_ratio']  = float(np.mean(low))
            features['frac_mid_ratio']  = float(np.mean(mid))
            features['frac_high_ratio'] = float(np.mean(high))
            depths = wr['DEPTH'].astype(float).values if 'DEPTH' in wr.columns else np.ones_like(ratios)
            w = depths / max(1.0, np.sum(depths))
            features['weighted_mean_ratio'] = float(np.sum(w*ratios))
            features['mean_depth'] = float(np.mean(depths))
            features['std_depth']  = float(np.std(depths, ddof=0))
            features['w_frac_low_ratio']  = float(np.sum(w*low))
            features['w_frac_mid_ratio']  = float(np.sum(w*mid))
            features['w_frac_high_ratio'] = float(np.sum(w*high))
            # Robust dispersion
            q25, q75 = np.percentile(ratios, [25,75])
            features['iqr_ratio'] = float(q75 - q25)
            features['mad_ratio'] = float(np.median(np.abs(ratios - np.median(ratios))))
            features['ratio_cv']  = float(sd / (mu + 1e-3))
            # Runs of low ratios (ordered by POS)
            rpos = wr.sort_values('POS')
            low_sorted = (rpos['RATIO'].values < 0.25)
            features['longest_low_streak'] = float(longest_run(low_sorted))
            # Inter-marker gap stats
            if len(rpos) > 1:
                gaps = np.diff(rpos['POS'].values.astype(int))
                features['mean_inter_ratio_gap'] = float(np.mean(gaps))
                features['std_inter_ratio_gap']  = float(np.std(gaps))
            else:
                features['mean_inter_ratio_gap'] = 0.0
                features['std_inter_ratio_gap']  = 0.0
            # distances to canonical points
            features['dist_from_mutant']   = float(np.mean(np.abs(ratios - 0.0)))
            features['dist_from_hawaiian'] = float(np.mean(np.abs(ratios - 1.0)))
            features['dist_from_het']      = float(np.mean(np.abs(ratios - 0.5)))
        else:
            for k in ['mean_ratio','median_ratio','std_ratio','min_ratio','max_ratio','ratio_range','ratio_skew','ratio_kurtosis',
                      'n_low_ratio','n_mid_ratio','n_high_ratio','frac_low_ratio','frac_mid_ratio','frac_high_ratio',
                      'weighted_mean_ratio','mean_depth','std_depth','w_frac_low_ratio','w_frac_mid_ratio','w_frac_high_ratio',
                      'iqr_ratio','mad_ratio','ratio_cv','longest_low_streak','mean_inter_ratio_gap','std_inter_ratio_gap',
                      'dist_from_mutant','dist_from_hawaiian','dist_from_het']:
                features[k] = 0.0 if k.startswith(('n_','std_','min_','max_','ratio_range','ratio_skew','ratio_kurtosis','ratio_cv')) else (0.5 if 'mean' in k or 'median' in k else 0.0)

        # combined + positional ---------------------------------------
        features['snp_to_ratio_density'] = float(features['homo_snp_density'] / (features['ratio_site_density'] + 1e-9))
        features['mutant_score']         = float(features['homo_snp_density'] * (1.0 - features['mean_ratio']))
        features['hawaiian_score']       = float((1.0 / (features['homo_snp_density'] + 1.0)) * features['mean_ratio'])
        features['recomb_signal']        = float(features['frac_mid_ratio'] * (1.0 / (features['homo_snp_density'] + 1.0)))
        features['window_start_mb']      = float(start / 1e6)
        features['window_center_mb']     = float((start + end) / 2e6)
        return features

    def _features_to_vector(self, features_dict):
        if not self.feature_names:
            self.feature_names = list(features_dict.keys())
        return np.array([features_dict[k] for k in self.feature_names], dtype=float)

    # ----- model -----
    def _make_estimator(self, scale_pos_weight=1.0):
        # IMPORTANT: n_jobs=1 everywhere to avoid subprocess spawn
        if self.classifier_type == 'xgboost':
            base = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=1.0,
                scale_pos_weight=float(scale_pos_weight), objective='binary:logistic',
                eval_metric='auc', random_state=SEED, n_jobs=1, **XGB_DEVICE_KW
            )
        elif self.classifier_type == 'lightgbm':
            base = lgb.LGBMClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                class_weight=None, random_state=SEED, n_jobs=1, verbose=-1
            )
        else:  # random_forest
            base = RandomForestClassifier(
                n_estimators=600, max_depth=None, min_samples_split=4, min_samples_leaf=2,
                random_state=SEED, n_jobs=1
            )
        clf = CalibratedClassifierCV(base, method='isotonic', cv=3, n_jobs=1)
        pipe = Pipeline([("clf", clf)])
        return pipe

    def create_synthetic_training_data(self, n_mutants=40, pos_windows=12, neg_windows=18):
        """Synthetic features consistent with feature names (fast demo)."""
        self._ensure_feature_template()
        n_features = len(self.feature_names)
        name2idx = {n:i for i,n in enumerate(self.feature_names)}
        idx_hsnpden = name2idx.get('homo_snp_density')
        idx_meanrat = name2idx.get('mean_ratio')
        idx_frclow  = name2idx.get('frac_low_ratio')
        idx_frchigh = name2idx.get('frac_high_ratio')
        idx_mutscr  = name2idx.get('mutant_score')

        X_list, y_list, g_list = [], [], []
        for m in range(n_mutants):
            P = np.random.normal(0,1,(pos_windows, n_features))
            N = np.random.normal(0,1,(neg_windows, n_features))
            if idx_hsnpden is not None: P[:,idx_hsnpden]+=2.0;  N[:,idx_hsnpden]-=1.5
            if idx_meanrat is not None: P[:,idx_meanrat]-=1.2; N[:,idx_meanrat]+=1.2
            if idx_frclow  is not None: P[:,idx_frclow ]+=1.0;  N[:,idx_frclow ]-=0.8
            if idx_frchigh is not None: P[:,idx_frchigh]-=1.0;  N[:,idx_frchigh]+=0.8
            if idx_mutscr  is not None: P[:,idx_mutscr ]+=2.0;  N[:,idx_mutscr ]-=2.0
            X_list.extend([P,N]); y_list.extend([np.ones(pos_windows), np.zeros(neg_windows)])
            g_list.extend([np.full(pos_windows, m), np.full(neg_windows, m)])
        X = np.vstack(X_list).astype(float)
        y = np.concatenate(y_list).astype(int)
        groups = np.concatenate(g_list).astype(int)
        return X, y, groups

    def train(self, X, y, groups=None, optimize_threshold=True):
        """Grouped CV (by mutant), sequential to avoid joblib spawn; prints confusion matrix."""
        pos, neg = int(y.sum()), int((y==0).sum())
        spw = max(1.0, neg / max(1, pos))
        cv = GroupKFold(n_splits=5) if groups is not None else None

        # Sequential CV loop
        oof = np.zeros_like(y, dtype=float)
        aucs = []
        for fold, (tr, te) in enumerate(cv.split(X, y, groups), 1):
            est_fold = self._make_estimator(scale_pos_weight=spw)
            est_fold.fit(X[tr], y[tr])
            p = est_fold.predict_proba(X[te])[:,1]
            oof[te] = p
            aucs.append(roc_auc_score(y[te], p))
        print(f"CV ROC-AUC (grouped by mutant): {np.mean(aucs):.3f} +/-  {np.std(aucs):.3f}")

        # Threshold by PR-F1 on OOF
        precision, recall, thr = precision_recall_curve(y, oof)
        f1 = 2*precision*recall / (precision + recall + 1e-10)
        thr_cv = float(thr[np.nanargmax(f1[:-1])])
        yhat = (oof >= thr_cv).astype(int)

        print(f"Optimal decision threshold (CV, PR-F1): {thr_cv:.3f}")
        print("\nCONFUSION MATRIX (OOF, grouped CV):")
        cm = confusion_matrix(y, yhat, labels=[0,1])
        print(pd.DataFrame(cm, index=['True Hawaiian(0)','True Mutant(1)'], columns=['Pred Hawaiian(0)','Pred Mutant(1)']))
        print("\nCLASSIFICATION REPORT (OOF):\n", classification_report(y, yhat, target_names=['Hawaiian','Mutant Parent']))

        # Fit final model on all data (sequential)
        self.classifier = self._make_estimator(scale_pos_weight=spw)
        self.classifier.fit(X, y)
        self.trained = True

        if optimize_threshold:
            p_all = self.classifier.predict_proba(X)[:,1]
            precision2, recall2, thr2 = precision_recall_curve(y, p_all)
            f12 = 2*precision2*recall2 / (precision2 + recall2 + 1e-10)
            self.optimal_threshold = float(thr2[np.nanargmax(f12[:-1])])
            print(f"Optimal decision threshold (classifier-only): {self.optimal_threshold:.3f}")
        return self.classifier

    # ----- smoothing (Gaussian + HMM) -----
    def _gaussian_average(self, p, sigma_windows=2.0):
        return gaussian_filter1d(p, sigma=float(sigma_windows), mode="nearest")

    def _viterbi_two_state(self, p, stay=0.995):
        """Two-state HMM decode with stay probability (0=Hawaiian, 1=Mutant)."""
        T = len(p)
        logA = np.log([[stay, 1.0-stay],[1.0-stay, stay]])
        logB = np.vstack([np.log(1.0-p + 1e-12), np.log(p + 1e-12)])
        dp  = np.zeros((2, T)); ptr = np.zeros((2, T), dtype=np.int8)
        dp[:,0] = np.log(0.5) + logB[:,0]
        for t in range(1,T):
            prev_to_0 = dp[:,t-1] + logA[:,0]
            prev_to_1 = dp[:,t-1] + logA[:,1]
            ptr[0,t] = int(np.argmax(prev_to_0))
            ptr[1,t] = int(np.argmax(prev_to_1))
            dp[0,t]  = prev_to_0[ptr[0,t]] + logB[0,t]
            dp[1,t]  = prev_to_1[ptr[1,t]] + logB[1,t]
        states = np.zeros(T, dtype=np.int8)
        states[-1] = int(np.argmax(dp[:, -1]))
        for t in range(T-2,-1,-1):
            states[t] = ptr[states[t+1], t+1]
        return states

    # ----- genome prediction -----
    def predict_genome(self, snp_vcf, ratio_vcf, smooth=True, sigma_windows=1.8, stay=0.99, alpha_heuristic=0.8):
        if not self.trained:
            raise RuntimeError("Train the classifier first (even synthetic).")
        homo = self.parse_homozygous_snps(snp_vcf)
        ratio = self.parse_allele_ratios(ratio_vcf)
        allpos = pd.concat([homo[['CHROM','POS']], ratio[['CHROM','POS']]], ignore_index=True)
        chrom_sizes = allpos.groupby('CHROM')['POS'].max().to_dict()

        predictions = []
        for chrom in sorted(chrom_sizes.keys()):
            size = int(chrom_sizes[chrom])
            tmp = []
            for start in range(0, max(1, size), self.step_size):
                end = min(start + self.window_size, size)
                f = self.extract_window_features(homo, ratio, chrom, start, end)
                x = self._features_to_vector(f).reshape(1,-1)
                p_clf = float(self.classifier.predict_proba(x)[0][1])
                tmp.append({
                    'chrom': chrom, 'start': start, 'end': end, 'center': (start+end)//2,
                    'p_clf': p_clf,
                    'homo_snp_density': f['homo_snp_density'],
                    'n_ratio_sites': f['n_ratio_sites'],
                    'ratio_site_density': f['ratio_site_density'],
                    'mean_ratio': f['mean_ratio'],
                    'frac_low_ratio': f['frac_low_ratio'],
                    'frac_mid_ratio': f['frac_mid_ratio'],
                    'frac_high_ratio': f['frac_high_ratio'],
                    'mutant_score': f['mutant_score'],
                })
            dfc = pd.DataFrame(tmp)
            # heuristic combination
            def rank01(a):
                if len(a)==0: return a
                r = pd.Series(a).rank(method='average').values
                return (r - 1) / max(1, len(a) - 1)
            r_mut = rank01(dfc['mutant_score'].values)
            r_den = rank01(dfc['homo_snp_density'].values)
            one_minus_mean = 1.0 - np.clip(dfc['mean_ratio'].values, 0.0, 1.0)
            frac_low = np.clip(dfc['frac_low_ratio'].values, 0.0, 1.0)
            p_heur = 0.45*r_mut + 0.25*one_minus_mean + 0.20*frac_low + 0.10*r_den
            p_heur = np.clip(p_heur, 0.0, 1.0)
            p_comb = alpha_heuristic*p_heur + (1.0 - alpha_heuristic)*dfc['p_clf'].values
            p_comb = np.clip(p_comb, 0.0, 1.0)
            # smoothing + HMM
            if smooth:
                p_smooth = self._gaussian_average(p_comb, sigma_windows=sigma_windows)
                states = self._viterbi_two_state(p_smooth, stay=stay)
            else:
                p_smooth = p_comb
                states = (p_comb >= 0.5).astype(int)
            for i in range(len(dfc)):
                predictions.append({
                    **dfc.iloc[i].to_dict(),
                    'prob_mutant_heur': float(p_heur[i]),
                    'prob_mutant': float(p_comb[i]),
                    'prob_mutant_smooth': float(p_smooth[i]),
                    'prediction_smoothed': int(states[i])
                })
        # Keep parsed dataframes for downstream tables/plots
        self._last_homo_df = homo
        self._last_ratio_df = ratio
        return pd.DataFrame(predictions).sort_values(['chrom','start']).reset_index(drop=True)

    # ----- region calling -----
    def _trim_region_edges(self, sub, a, b, p_edge_min=None, peak_frac=None):
        p = sub['prob_mutant_smooth'].values
        segment = p[a:b+1]
        if len(segment)==0: return a, b
        if peak_frac is not None:
            thr = peak_frac * float(np.max(segment))
        elif p_edge_min is not None:
            thr = float(p_edge_min)
        else:
            return a, b
        good = segment >= thr
        if not np.any(good):
            peak_idx = int(np.argmax(segment))
            return a+peak_idx, a+peak_idx
        peak_idx = int(np.argmax(segment))
        g = good.astype(int)
        starts = np.where(np.diff(np.r_[0,g,0])==1)[0]
        ends   = np.where(np.diff(np.r_[0,g,0])==-1)[0]-1
        for s,e in zip(starts, ends):
            if s <= peak_idx <= e:
                return a+s, a+e
        lengths = (ends - starts + 1)
        k = int(np.argmax(lengths))
        return a+int(starts[k]), a+int(ends[k])

    def find_mapping_regions(self, predictions_df, min_region_size=100_000, min_confidence=0.6,
                             max_gap_windows=0, p_edge_min=None, peak_frac=None):
        df = predictions_df.copy()
        out = []
        for chrom in df['chrom'].unique():
            sub = df[df['chrom']==chrom].sort_values('start').reset_index(drop=True)
            runs = []
            s = None
            for i, r in sub.iterrows():
                if r['prediction_smoothed']==1 and s is None:
                    s = i
                elif r['prediction_smoothed']==0 and s is not None:
                    runs.append((s, i-1)); s=None
            if s is not None: runs.append((s, len(sub)-1))
            merged=[]
            for a,b in runs:
                if merged and (a - merged[-1][1] - 1) <= max_gap_windows:
                    merged[-1] = (merged[-1][0], b)
                else:
                    merged.append((a,b))
            for a,b in merged:
                ta, tb = self._trim_region_edges(sub, a, b, p_edge_min=p_edge_min, peak_frac=peak_frac)
                start, end = int(sub.loc[ta,'start']), int(sub.loc[tb,'end'])
                size = end - start
                if size < min_region_size: continue
                probs = sub.loc[ta:tb, 'prob_mutant_smooth']
                conf = float(np.mean(probs))
                if conf < min_confidence: continue
                out.append({
                    'chrom': chrom, 'start': start, 'end': end, 'size': size, 'size_mb': size/1e6,
                    'confidence': conf,
                    'p5': float(np.percentile(probs,5)), 'p95': float(np.percentile(probs,95)),
                    'n_windows': int(tb-ta+1),
                    'peak_center': int(sub.loc[ta:tb].iloc[np.argmax(sub.loc[ta:tb,'prob_mutant_smooth'])]['center']),
                    'edge_left_p': float(sub.loc[ta,'prob_mutant_smooth']),
                    'edge_right_p': float(sub.loc[tb,'prob_mutant_smooth'])
                })
        return sorted(out, key=lambda x: (x["chrom"], -x["confidence"], x["size"]))

    # ----- region diagnostics -----
    def describe_region(self, predictions_df, region, flank_bp=250_000):
        chrom = region['chrom']
        sub = predictions_df[predictions_df['chrom']==chrom].sort_values('start').reset_index(drop=True)
        inside = (sub['start']>=region['start']) & (sub['end']<=region['end'])
        flankL = (sub['end']>=max(0, region['start']-flank_bp)) & (sub['end']<region['start'])
        flankR = (sub['start']<=region['end']+flank_bp) & (sub['start']>region['end'])

        def summary(mask, label):
            blk = sub[mask]
            if blk.empty:
                print(f"  {label}: (no windows)")
                return
            print(f"  {label}: n={len(blk)}  P_mutant_smooth mean={blk['prob_mutant_smooth'].mean():.3f} "
                  f"min={blk['prob_mutant_smooth'].min():.2f} median={blk['prob_mutant_smooth'].median():.2f} "
                  f"max={blk['prob_mutant_smooth'].max():.2f}")
            print(f"          homo_snp_density mean={blk['homo_snp_density'].mean():.3f}/kb "
                  f"ratio_sites mean={blk['n_ratio_sites'].mean():.1f} "
                  f"mean_ratio={blk['mean_ratio'].mean():.3f}  "
                  f"frac_low={blk['frac_low_ratio'].mean():.3f}  frac_mid={blk['frac_mid_ratio'].mean():.3f}  frac_high={blk['frac_high_ratio'].mean():.3f}")

        print(f"\n=== Region Diagnostics: Chr {chrom}: {region['start']:,}-{region['end']:,} "
              f"| size={region['size']/1e6:.3f} Mb | mean P={region['confidence']:.3f} "
              f"| p5={region['p5']:.2f} p95={region['p95']:.2f} | peak@{region['peak_center']:,} ===")
        print(f"  Edge probs: left={region['edge_left_p']:.3f}, right={region['edge_right_p']:.3f}")
        summary(inside, "Inside")
        summary(flankL, "Left flank (250 kb)")
        summary(flankR, "Right flank (250 kb)")

    # ----- variant table in mapping interval (with allele ratios + CAUSAL LABELING) -----
    def variants_in_interval_table(self, region, save_prefix=f"{HOME_DIR}/outputs/mapping_interval",
                                   nearest_tol_bp=1000, causal_chr_label=None, causal_pos=None):
        """
        Build and print a table of homozygous SNPs inside the mapping interval with:
          - EMS tagging (G/C->A/T)
          - Hawaiian allele ratio at the same position (or nearest within +/- nearest_tol_bp)
          - Causal labeling (IS_CAUSAL, DIST_TO_CAUSAL_BP) if causal is provided
          - SORTED BY POSITION (do not bump causal to the top)
        """
        Path(save_prefix).parent.mkdir(parents=True, exist_ok=True)
        if self._last_homo_df is None or self._last_ratio_df is None:
            raise RuntimeError("Call predict_genome() first to populate parsed dataframes.")
        chrom = region['chrom']
        start, end = region['start'], region['end']
        df = self._last_homo_df
        ratio = self._last_ratio_df

        sub = df[(df['CHROM']==chrom) & (df['POS']>=start) & (df['POS']<=end)].copy()
        if sub.empty:
            print("\n[Variant Table] No homozygous SNPs found inside the mapping interval.")
            return sub

        # EMS
        sub['EMS'] = sub.apply(lambda r: is_ems_transition(r['REF'], r['ALT']), axis=1)
        sub['EMS_label'] = np.where(sub['EMS'], 'EMS (G/C->A/T)', 'non-EMS')

        # Join nearest Hawaiian allele ratio
        ratio_chr = ratio[ratio['CHROM']==chrom].sort_values('POS').reset_index(drop=True)
        sub = sub.sort_values('POS').reset_index(drop=True)
        sub['Hawaiian_ratio'] = np.nan
        sub['Ratio_source'] = 'none'
        sub['Ratio_nearest_pos'] = np.nan
        sub['Ratio_nearest_dist_bp'] = np.nan

        if len(ratio_chr) > 0:
            qpos = sub['POS'].values
            nearest_ratio, nearest_pos, nearest_dist, src = nearest_ratio_for_positions(
                ratio_chr, qpos, tol_bp=nearest_tol_bp
            )
            sub['Hawaiian_ratio'] = nearest_ratio
            sub['Ratio_nearest_pos'] = nearest_pos
            sub['Ratio_nearest_dist_bp'] = nearest_dist
            sub['Ratio_source'] = src
            exact_mask = sub['Ratio_nearest_dist_bp'].fillna(1e9).astype(float) == 0.0
            sub.loc[exact_mask, 'Ratio_source'] = 'exact'

        sub['Parental_ratio'] = 1.0 - sub['Hawaiian_ratio']

        # Causal labeling (do NOT sort by IS_CAUSAL; keep chromosomal order)
        causal_chr_norm = normalize_chr(causal_chr_label) if causal_chr_label else None
        if causal_chr_norm and (causal_pos is not None) and (causal_chr_norm == chrom):
            sub['IS_CAUSAL'] = (sub['POS'].astype(int) == int(causal_pos))
            sub['DIST_TO_CAUSAL_BP'] = (sub['POS'].astype(int) - int(causal_pos)).abs()
        else:
            sub['IS_CAUSAL'] = False
            sub['DIST_TO_CAUSAL_BP'] = np.nan

        # Final sort strictly by POS (requested behavior)
        sub = sub.sort_values('POS').reset_index(drop=True)

        out_cols = ['CHROM','POS','REF','ALT','QUAL','DP','EMS_label',
                    'Hawaiian_ratio','Parental_ratio','Ratio_source','Ratio_nearest_pos','Ratio_nearest_dist_bp',
                    'IS_CAUSAL','DIST_TO_CAUSAL_BP']
        sub = sub[out_cols]

        # Summary + print (safe formatting)
        total = len(sub)
        ems_count = int((sub['EMS_label']=='EMS (G/C->A/T)').sum())
        non_count = total - ems_count
        ems_frac = ems_count / total if total>0 else float('nan')
        non_frac = non_count / total if total>0 else float('nan')
        ems_frac_str = f"{ems_frac:.2%}" if total>0 else "NA"
        non_frac_str = f"{non_frac:.2%}" if total>0 else "NA"

        print("\n[Variant Table] Homozygous SNPs inside mapping interval "
              f"{chrom}:{start:,}-{end:,} (n={total}) — sorted by chromosomal position")
        print(f"  EMS (G/C->A/T): {ems_count} ({ems_frac_str})  |  non-EMS: {non_count} ({non_frac_str})")

        show_rows = min(60, total)
        if total > show_rows:
            print(f"(Showing first {show_rows} of {total} variants; full table saved below)")
            print(sub.head(show_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        else:
            print(sub.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        out_csv = f"{save_prefix}_variants.csv"
        sub.to_csv(out_csv, index=False, na_rep='nan')
        print(f"Saved full variant table to: {out_csv}")
        return sub

    # ----- plot ALL chromosomes with highlight on chosen interval -----
    def plot_genome_predictions(self, predictions_df, mapping_regions=None, ratio_df=None, highlight_region=None):
        chroms = ['I','II','III','IV','V','X']
        chroms = [c for c in chroms if c in predictions_df['chrom'].unique()]
        n = len(chroms)
        if n==0:
            print("No predictions to plot."); return
        fig, axes = plt.subplots(n, 1, figsize=(15, 2.8*n), sharex=False)
        if n==1: axes=[axes]
        for i, chrom in enumerate(chroms):
            ax = axes[i]
            sub = predictions_df[predictions_df['chrom']==chrom].sort_values('center')
            ax.plot(sub['center'], sub['prob_mutant'], alpha=0.25, label='P(mutant) combined')
            ax.plot(sub['center'], sub['prob_mutant_smooth'], linewidth=2, label='P(mutant) smoothed')
            if ratio_df is not None and (ratio_df['CHROM']==chrom).any():
                rchr = ratio_df[ratio_df['CHROM']==chrom]
                binsize = 100_000
                bins = np.arange(0, max(1, sub['center'].max())+binsize, binsize)
                if len(rchr)>0 and len(bins) > 1:
                    br, _, _ = binned_statistic(rchr['POS'].values, rchr['RATIO'].values, statistic='mean', bins=bins)
                    centers = (bins[:-1] + bins[1:]) / 2
                    mask = ~np.isnan(br)
                    ax.plot(centers[mask], br[mask], linestyle='--', alpha=0.6, label='Allele ratio (binned)')
            if mapping_regions:
                for r in mapping_regions:
                    if r['chrom']==chrom:
                        ax.axvspan(r['start'], r['end'], alpha=0.10, color='gray')
            if highlight_region and highlight_region['chrom']==chrom:
                ax.axvspan(highlight_region['start'], highlight_region['end'], alpha=0.30, color='orange')
            ax.set_ylim(0,1); ax.set_ylabel(f"Chr {chrom}"); ax.grid(alpha=0.2)
            if i==0: ax.legend(fontsize=8)
            ax.set_title(f"Chromosome {chrom}")
        axes[-1].set_xlabel("Position (bp)")
        plt.suptitle("Genome-wide mutant vs. Hawaiian probability (all chromosomes)\n"
                     "Shading: light=all mapping regions; orange=chosen interval containing causal site", y=1.02)
        plt.tight_layout(); plt.show()

    # ----- UPDATED: compact plot — y-axis is Parental_ratio (0-1), variants as colored bars on x-axis; no probability lines -----
    def plot_compact_regions_with_variants(self, mapping_regions, nearest_tol_bp=1000):
        """
        For each chromosome, show ONLY the top-confidence called mapping region, overlaying:
          - Vertical bars at each homozygous variant position (x = POS).
              - Bar height = Parental_ratio = 1 - Hawaiian_ratio (from nearest ratio site within tolerance).
              - EMS variants (G/C->A/T): red bars
              - non-EMS variants: blue bars
          - A downward arrow labeled 'predicted peak' at the region's peak_center, pointing to the bar height at the
            nearest variant to the peak_center (or the top of the axis if no nearby ratio is available).
        No region-diagnostic probability lines or axis are shown.
        """
        if mapping_regions is None or len(mapping_regions) == 0:
            print("No mapping regions to plot."); return
        if self._last_homo_df is None or self._last_ratio_df is None:
            print("No cached variants/ratios; run predict_genome() first."); return

        # top-confidence region per chromosome
        top_by_chr = {}
        for r in mapping_regions:
            c = r['chrom']
            if (c not in top_by_chr) or (r['confidence'] > top_by_chr[c]['confidence']):
                top_by_chr[c] = r

        chroms_all = ['I','II','III','IV','V','X']
        n = len(chroms_all)
        fig, axes = plt.subplots(n, 1, figsize=(14, 1.9*n), sharex=False)
        if n == 1: axes = [axes]

        for i, chrom in enumerate(chroms_all):
            ax = axes[i]
            if chrom not in top_by_chr:
                ax.axis('off')
                ax.text(0.5, 0.5, f"Chr {chrom}: no mapping region called", transform=ax.transAxes,
                        ha='center', va='center', fontsize=10)
                continue

            r = top_by_chr[chrom]
            start, end, peak_center = r['start'], r['end'], r['peak_center']
            ax.axvspan(start, end, color='lightgray', alpha=0.35, zorder=0)

            # Collect variants in region
            subv = self._last_homo_df[(self._last_homo_df['CHROM']==chrom) &
                                      (self._last_homo_df['POS']>=start) &
                                      (self._last_homo_df['POS']<=end)].copy()

            # Compute EMS flag and parental ratio via nearest Hawaiian ratio site
            if not subv.empty:
                subv['EMS'] = subv.apply(lambda row: is_ems_transition(row['REF'], row['ALT']), axis=1)
                rchr = self._last_ratio_df[self._last_ratio_df['CHROM']==chrom].sort_values('POS').reset_index(drop=True)
                subv = subv.sort_values('POS').reset_index(drop=True)
                subv['Hawaiian_ratio'] = np.nan
                if len(rchr) > 0:
                    nr, np_pos, ndist, src = nearest_ratio_for_positions(rchr, subv['POS'].values, tol_bp=nearest_tol_bp)
                    subv['Hawaiian_ratio'] = nr
                subv['Parental_ratio'] = 1.0 - subv['Hawaiian_ratio']
                # Drop rows lacking a usable ratio
                plot_df = subv[~subv['Parental_ratio'].isna()].copy()
            else:
                plot_df = pd.DataFrame(columns=['POS','Parental_ratio','EMS'])

            # Choose bar width in bp (scaled to region span)
            span = max(1, end - start)
            width_bp = max(int(span * 0.0015), 100)  # ~0.15% of region span, at least 100 bp

            # Plot bars by class
            if not plot_df.empty:
                non = plot_df[~plot_df['EMS']]
                ems = plot_df[plot_df['EMS']]
                if len(non) > 0:
                    ax.bar(non['POS'].values, non['Parental_ratio'].values, width=width_bp,
                           align='center', color='blue', alpha=0.9, edgecolor='none', label='non-EMS', zorder=2)
                if len(ems) > 0:
                    ax.bar(ems['POS'].values, ems['Parental_ratio'].values, width=width_bp,
                           align='center', color='red', alpha=0.95, edgecolor='none', label='EMS (G/C->A/T)', zorder=3)

            # Arrow to predicted peak (downwards): target y = parental ratio of nearest variant to peak_center
            if not plot_df.empty:
                idx_near = int(np.argmin(np.abs(plot_df['POS'].values - peak_center)))
                peak_x = float(plot_df.iloc[idx_near]['POS'])
                peak_y = float(plot_df.iloc[idx_near]['Parental_ratio'])
            else:
                peak_x = peak_center
                peak_y = 1.0
            y_text = min(0.98, max(peak_y + 0.10, 0.90))
            ax.annotate("predicted peak",
                        xy=(peak_x, peak_y), xytext=(peak_x, y_text),
                        arrowprops=dict(arrowstyle='-|>', lw=1.2),
                        ha='center', va='bottom', zorder=5)

            # Axes cosmetics
            ax.set_xlim(start, end)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_ylabel("Parental ratio (1 - Hawaiian)", fontsize=9)
            title = f"Chromosome {chrom}: {start:,}-{end:,}  conf={r['confidence']:.2f}"
            ax.set_title(title, fontsize=10)
            ax.grid(axis='y', alpha=0.2)
            if i == 0:
                ax.legend(loc='upper right', fontsize=8, frameon=False)

        axes[-1].set_xlabel("Position (bp)")
        plt.suptitle("Compact view: called mapping regions — variant bars colored by class\n"
                     "Height = Parental ratio (0-1). Arrow points DOWN to nearest variant at predicted peak.", y=1.02)
        plt.tight_layout()
        plt.show()

# =================== Statistical uncertainty: bootstrap & permutation ===================
def _region_indices_and_arrays(pred_df, region):
    """Return chromosome sub-DF, and region index range [a,b] inclusive, and arrays."""
    chrom = region['chrom']
    sub = pred_df[pred_df['chrom']==chrom].sort_values('start').reset_index(drop=True)
    inside = (sub['start']>=region['start']) & (sub['end']<=region['end'])
    idx = np.where(inside.values)[0]
    if len(idx)==0:
        return sub, None, None, None, None
    a, b = int(idx.min()), int(idx.max())
    p_chrom = sub['prob_mutant_smooth'].values.astype(float)
    p_region = p_chrom[a:b+1]
    return sub, a, b, p_chrom, p_region

def moving_block_bootstrap_mean(arr, B=2000, block_size=3, seed=SEED):
    """Circular moving-block bootstrap of the mean for a 1D array arr."""
    rng = np.random.default_rng(seed)
    m = len(arr)
    if m == 0:
        return np.array([])
    block_size = max(1, min(block_size, m))
    n_blocks = int(np.ceil(m / block_size))
    out = np.empty(B, dtype=float)
    idx_all = np.arange(m)
    for i in range(B):
        sel = []
        for _ in range(n_blocks):
            s = int(rng.integers(0, m))  # start
            # circular block [s, s+1, ..., s+block_size-1] mod m
            blk = (s + np.arange(block_size)) % m
            sel.extend(blk.tolist())
        sel = sel[:m]
        out[i] = float(np.mean(arr[sel]))
    return out

def circular_permutation_means(p_chrom, a, b, n_perm=2000, seed=SEED):
    """Circularly shift the chromosome series and compute region means at the same index range."""
    rng = np.random.default_rng(seed)
    T = len(p_chrom)
    if T == 0 or a is None or b is None: return np.array([])
    L = b - a + 1
    base = np.arange(a, b+1)  # shape (L,)
    shifts = rng.integers(0, T, size=n_perm)[:, None]  # (n_perm,1)
    idxs = (base[None, :] + shifts) % T               # (n_perm, L)
    means = p_chrom[idxs].mean(axis=1)
    return means

def region_uncertainty(pred_df, region, B_boot=2000, block_size=3, n_perm=2000, seed=SEED):
    """Compute bootstrap CI and permutation p-value for a region's mean prob."""
    sub, a, b, p_chrom, p_region = _region_indices_and_arrays(pred_df, region)
    if p_region is None:
        print("[Uncertainty] No windows inside region; skipping.")
        return None
    obs_mean = float(np.mean(p_region))
    boot = moving_block_bootstrap_mean(p_region, B=B_boot, block_size=block_size, seed=seed)
    ci_lo, ci_hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if len(boot)>0 else (np.nan, np.nan)
    null_means = circular_permutation_means(p_chrom, a, b, n_perm=n_perm, seed=seed)
    p_perm = float((1 + np.sum(null_means >= obs_mean)) / (1 + len(null_means))) if len(null_means)>0 else np.nan
    print(f"\n[Region uncertainty] Chr {region['chrom']}:{region['start']:,}-{region['end']:,}")
    print(f"  Observed mean P = {obs_mean:.3f}  |  Bootstrap 95% CI = [{ci_lo:.3f}, {ci_hi:.3f}] (B={B_boot}, block={block_size})")
    if len(null_means)>0:
        print(f"  Permutation p-value (circular shift on chromosome, n={len(null_means)}): p = {p_perm:.4f}")
    return {'obs_mean': obs_mean, 'ci': (ci_lo, ci_hi), 'boot': boot, 'p_perm': p_perm, 'null_means': null_means}

def bootstrap_difference_ci(pred_df, region1, region2, B=2000, block_size=3, seed=SEED):
    """Bootstrap CI for difference of means: mean(region1) - mean(region2) with independent block resampling per region."""
    _, _, _, _, arr1 = _region_indices_and_arrays(pred_df, region1)
    _, _, _, _, arr2 = _region_indices_and_arrays(pred_df, region2)
    if arr1 is None or arr2 is None:
        print("[Diff CI] One or both regions missing; skipping.")
        return None
    boot1 = moving_block_bootstrap_mean(arr1, B=B, block_size=block_size, seed=seed)
    boot2 = moving_block_bootstrap_mean(arr2, B=B, block_size=block_size, seed=seed+1)
    diff = boot1 - boot2
    ci_lo, ci_hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
    obs = float(np.mean(arr1) - np.mean(arr2))
    p_two_sided = 2.0 * min(np.mean(diff >= 0.0), np.mean(diff <= 0.0))
    print(f"\n[Difference (III - V)] Observed = {obs:.3f}  |  Bootstrap 95% CI = [{ci_lo:.3f}, {ci_hi:.3f}] (B={B}, block={block_size})")
    print(f"  Two-sided bootstrap p (diff ≠ 0): {p_two_sided:.4f}")
    return {'obs_diff': obs, 'ci': (ci_lo, ci_hi), 'p_two_sided': p_two_sided, 'boot_diff': diff}

# =================== Printing helpers (explanations) ===================
def explain_labeling_and_thresholds(p_edge_min=0.65, min_conf=0.60, min_size=100_000):
    print("\n[How windows/regions are labeled]")
    print("  - Window classifier positive class (1)  = 'Mutant Parent / linked' (parental).")
    print("    Negative class (0) = 'Hawaiian / unlinked'.")
    print("  - Features: enriched set: homozygous SNP density/spacing/quality (+EMS, Ti/Tv),")
    print("    Hawaiian allele-ratio stats (+depth-weighted fractions, IQR/MAD, longest low run, inter-marker gaps),")
    print("    and combined indicators (e.g., mutant_score = homo_snp_density * (1 - mean_ratio)).")
    print("  - Pipeline: calibrated XGBoost probability -> 80/20 blend with heuristic -> Gaussian smoothing -> 2-state HMM.")
    print(f"  - Region call: contiguous HMM=1 windows; edges trimmed to >= {p_edge_min:.2f}; require mean conf >= {min_conf:.2f} and size >= {min_size/1e3:.0f} kb.")

def explain_top_region_fields():
    print("\n[Explanation of 'Top predicted mapping regions' fields]")
    print("  size      = physical length of the trimmed interval")
    print("  conf      = mean(prob_mutant_smooth) across windows inside the interval")
    print("  p5,p95    = 5th and 95th percentiles of prob_mutant_smooth inside the interval")
    print("  edges     = (left_edge_p, right_edge_p) after trimming; these are >= p_edge_min by construction")

def pick_chr_region_by_coords_or_top(regions, chrom, start=None, end=None):
    """Prefer exact coordinate match; else return the top-confidence region on that chromosome."""
    regs = [r for r in regions if r['chrom']==normalize_chr(chrom)]
    if start is not None and end is not None:
        for r in regs:
            if r['start']==int(start) and r['end']==int(end):
                return r
        # if exact not found, pick region with maximal overlap with the target span
        target = (int(start), int(end))
        def iou(a, b):
            inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
            union = (a[1]-a[0]) + (b[1]-b[0]) - inter
            return inter/union if union>0 else 0.0
        if regs:
            r_best = max(regs, key=lambda r: iou((r['start'], r['end']), target))
            return r_best
        return None
    # no coords provided: pick top by confidence
    if regs:
        return sorted(regs, key=lambda x: (-x['confidence'], x['size']))[0]
    return None

def compare_regions_confidence(r1, label1, r2, label2, p_edge_min=0.65):
    if (r1 is None) or (r2 is None):
        print("\n[Confidence comparison] One or both regions are missing; cannot compare.")
        return
    diff = r1['confidence'] - r2['confidence']
    rel = r1['confidence'] / max(r2['confidence'], 1e-9)
    print(f"\n[Confidence comparison] {label1} vs {label2}")
    print(f"  conf_{label1} = {r1['confidence']:.3f} ; conf_{label2} = {r2['confidence']:.3f}")
    print(f"  absolute difference = {diff:.3f}  (~{diff*100:.1f} percentage points)")
    print(f"  relative lift       = {rel:.2f}x")
    print("  Interpretation:")
    print(f"    - Both edges trimmed to >= p_edge_min={p_edge_min:.2f}: "
          f"{label1} edges=({r1['edge_left_p']:.2f},{r1['edge_right_p']:.2f}), "
          f"{label2} edges=({r2['edge_left_p']:.2f},{r2['edge_right_p']:.2f}).")
    print(f"    - Interior strength: {label1} p95={r1['p95']:.2f} vs {label2} p95={r2['p95']:.2f}.")
    print("    - Higher conf and higher p95 indicate a stronger, more consistent parental-like signal in the region with higher values.\n")

# =================== Pipeline ===================
def run_pipeline(homo_vcf, ratio_vcf, causal_chr_label, causal_pos,
                 window_size=250_000, step_size=50_000,
                 p_edge_min=0.65, min_region_conf=0.60, min_region_size=100_000,
                 B_BOOT=2000, BLOCK_SIZE=3, N_PERM=2000):
    for p in [homo_vcf, ratio_vcf]: exists_or_raise(p)

    causal_chr_norm = normalize_chr(causal_chr_label)
    print(f"HOME_DIR: {HOME_DIR}")
    print(f"Inputs:\n  Homozygous SNPs: {homo_vcf}\n  Hawaiian-site ratios: {ratio_vcf}")
    print(f"Causal site: {causal_chr_label}:{causal_pos:,} (normalized -> {causal_chr_norm}:{causal_pos:,})")
    print(f"Windowing: {window_size/1000:.0f} kb windows, {step_size/1000:.0f} kb step\n")

    clf = BulkedSegregantClassifier(window_size=window_size, step_size=step_size, classifier_type='xgboost')

    # Seed feature order by parsing once (also verifies VCF readability)
    homo_df_tmp = clf.parse_homozygous_snps(homo_vcf)
    ratio_df_tmp = clf.parse_allele_ratios(ratio_vcf)
    _ = clf.extract_window_features(homo_df_tmp, ratio_df_tmp, normalize_chr('I'), 0, window_size)

    # Synthetic training (grouped CV), sequential to avoid joblib spawn
    X, y, groups = clf.create_synthetic_training_data(n_mutants=40, pos_windows=12, neg_windows=18)
    clf.train(X, y, groups=groups, optimize_threshold=True)

    # Explain labeling & thresholds
    explain_labeling_and_thresholds(p_edge_min=p_edge_min, min_conf=min_region_conf, min_size=min_region_size)

    # Prediction + HMM smoothing (tighter defaults)
    print("\n=== Prediction + HMM smoothing (tighter defaults) ===")
    preds = clf.predict_genome(homo_vcf, ratio_vcf, smooth=True,
                               sigma_windows=1.8, stay=0.99, alpha_heuristic=0.8)

    # Region calling (stricter): no gap merge, edge trim @p_edge_min, min_conf
    regions = clf.find_mapping_regions(preds, min_region_size=min_region_size, min_confidence=min_region_conf,
                                       max_gap_windows=0, p_edge_min=p_edge_min, peak_frac=None)

    print("\nTop predicted mapping regions (stricter defaults):")
    for r in regions[:10]:
        print(f"Chr {r['chrom']}: {r['start']:,}-{r['end']:,}  size={r['size']/1e6:.3f} Mb  "
              f"conf={r['confidence']:.3f} (p5={r['p5']:.2f}, p95={r['p95']:.2f})  edges=({r['edge_left_p']:.2f},{r['edge_right_p']:.2f})")

    explain_top_region_fields()

    # Region(s) for chrIII and chrV
    regs_III = [r for r in regions if r['chrom']==causal_chr_norm and r['start'] <= causal_pos <= r['end']]
    best_chrIII = None
    if regs_III:
        best_chrIII = sorted(regs_III, key=lambda z: (-z['confidence'], z['size']))[0]
        print(f"\nCausal variant {causal_chr_norm}:{causal_pos:,} is inside mapping interval:")
        print(f"  interval = {best_chrIII['chrom']}:{best_chrIII['start']:,}-{best_chrIII['end']:,}")
        print(f"  length   = {best_chrIII['size']:,} bp ({best_chrIII['size']/1e6:.3f} Mb)")
    else:
        print(f"\nCausal variant {causal_chr_norm}:{causal_pos:,} is NOT inside any called mapping interval.")
        # pick the top chrIII region anyway for comparison/table
        best_chrIII = pick_chr_region_by_coords_or_top(regions, 'III')

    # chrV: prefer the slightly gray interval 17,550,000-18,000,000; else best overlap / top
    target_chrV = (17_550_000, 18_000_000)
    best_chrV = pick_chr_region_by_coords_or_top(regions, 'V', start=target_chrV[0], end=target_chrV[1])
    if best_chrV:
        print(f"\nSelected chrV interval for table: Chr V: {best_chrV['start']:,}-{best_chrV['end']:,}  "
              f"size={best_chrV['size']/1e6:.3f} Mb  conf={best_chrV['confidence']:.3f}  "
              f"edges=({best_chrV['edge_left_p']:.2f},{best_chrV['edge_right_p']:.2f})")
    else:
        print("\nNo chrV mapping interval found.")

    best_target = None
    target_chr_norm = None
    if TARGET_CHROM_RAW:
        target_chr_norm = normalize_chr(TARGET_CHROM_RAW)
        best_target = pick_chr_region_by_coords_or_top(regions, target_chr_norm)
        if best_target:
            print(f"\nTarget chromosome interval (from mapping_input.txt): Chr {best_target['chrom']} "
                  f"{best_target['start']:,}-{best_target['end']:,} "
                  f"(conf={best_target['confidence']:.3f}, size={best_target['size']/1e6:.3f} Mb)")
        else:
            print(f"\nNo mapping interval found for requested target chromosome: {target_chr_norm}")

    # Diagnostics + variant tables
    if best_chrIII:
        clf.describe_region(preds, best_chrIII, flank_bp=250_000)
        tbl_chrIII = clf.variants_in_interval_table(
            best_chrIII,
            save_prefix=f"{HOME_DIR}/outputs/mapping_interval_chrIII",
            nearest_tol_bp=1000,
            causal_chr_label=CAUSAL_CHR_LABEL,
            causal_pos=CAUSAL_POS
        )
        if (tbl_chrIII is None) or (not (tbl_chrIII['POS'] == CAUSAL_POS).any()):
            print("\n[Explanation] The causal site is not in the homozygous SNP table because:")
            print("  - It may not be present in the homozygous VCF at exactly that POS (filtered by caller).")
            print("  - It may not be a homozygous ALT call (we require GT=1/1), or it might be heterozygous.\n"
                  "  - The homozygous VCF has Hawaiian SNPs subtracted; the causal site may not pass those filters.")
            debug_causal_presence(homo_vcf, ratio_vcf, causal_chr_label, causal_pos, window_bp=2000)

    if best_chrV:
        clf.describe_region(preds, best_chrV, flank_bp=250_000)
        _ = clf.variants_in_interval_table(
            best_chrV,
            save_prefix=f"{HOME_DIR}/outputs/mapping_interval_chrV",
            nearest_tol_bp=1000,
            causal_chr_label=None,
            causal_pos=None
        )

    # If the user requested a specific chromosome via mapping_input.txt (3rd line),
    # always output the mapping-interval variant table for that chromosome too.
    if best_target and target_chr_norm:
        clf.describe_region(preds, best_target, flank_bp=250_000)
        causal_chr_norm = normalize_chr(CAUSAL_CHR_LABEL)
        causal_label_for_table = CAUSAL_CHR_LABEL if target_chr_norm == causal_chr_norm else None
        causal_pos_for_table = CAUSAL_POS if target_chr_norm == causal_chr_norm else None
        _ = clf.variants_in_interval_table(
            best_target,
            save_prefix=f"{HOME_DIR}/outputs/mapping_interval_{target_chr_norm}_0",
            nearest_tol_bp=1000,
            causal_chr_label=causal_label_for_table,
            causal_pos=causal_pos_for_table,
        )

    # Standard genome plot (probabilities)
    clf.plot_genome_predictions(preds, mapping_regions=regions, ratio_df=clf._last_ratio_df, highlight_region=best_chrIII if best_chrIII else None)

    # UPDATED compact per-chromosome plot:
    #   - y-axis is Parental_ratio (0-1)
    #   - variants drawn as colored barplots on the x-axis
    #   - no probability lines/axis; arrow points DOWN to nearest-variant height at predicted peak
    clf.plot_compact_regions_with_variants(mapping_regions=regions, nearest_tol_bp=1000)

    # Confidence comparison: chrIII vs chrV (deterministic)
    if best_chrIII and best_chrV:
        compare_regions_confidence(best_chrIII, "III", best_chrV, "V", p_edge_min=p_edge_min)

    # Uncertainty quantification (bootstrap CI + permutation p-value)
    if best_chrIII:
        statsIII = region_uncertainty(preds, best_chrIII, B_boot=B_BOOT, block_size=BLOCK_SIZE, n_perm=N_PERM, seed=SEED)
    if best_chrV:
        statsV = region_uncertainty(preds, best_chrV, B_boot=B_BOOT, block_size=BLOCK_SIZE, n_perm=N_PERM, seed=SEED+7)

    # Bootstrap CI for difference (III - V)
    if best_chrIII and best_chrV:
        _ = bootstrap_difference_ci(preds, best_chrIII, best_chrV, B=B_BOOT, block_size=BLOCK_SIZE, seed=SEED+13)

    # Unit test to confirm causal inside chosen interval (only if we found such an interval)
    if regs_III:
        class TestCausalInMappingInterval(unittest.TestCase):
            def test_causal_in_interval(self):
                hits = [r for r in regions if r['chrom']==causal_chr_norm and r['start'] <= causal_pos <= r['end']]
                self.assertTrue(len(hits)>0, msg=f"Causal {causal_chr_norm}:{causal_pos:,} not inside any interval.")
        print("\nRunning unit test for causal variant inclusion...")
        unittest.main(argv=['first-arg-is-ignored'], exit=False)

    return clf, preds, regions, (causal_chr_norm, causal_pos), best_chrIII, best_chrV

# =================== Run pipeline ===================
clf, PREDICTIONS_DF, REGIONS, CAUSAL, BEST_CHRIII, BEST_CHRV = run_pipeline(
    HOMO_VCF, RATIO_VCF, CAUSAL_CHR_LABEL, CAUSAL_POS,
    window_size=250_000, step_size=50_000,
    p_edge_min=0.65, min_region_conf=0.60, min_region_size=100_000,
    B_BOOT=2000, BLOCK_SIZE=3, N_PERM=2000  # adjust for speed/precision as needed
)

# =================== Notes ===================
print("""
WHAT'S NEW (compact view)
- Y-axis is Parental ratio (1 - Hawaiian), fixed 0-1.
- Variants are plotted as vertical bars at their genomic positions:
    - EMS (G/C->A/T): red bars
    - non-EMS: blue bars
- Removed region-diagnostic probability lines/axis entirely.
- 'predicted peak' arrow now points DOWN to the height of the nearest variant bar at the region's peak_center.
""")
