#!/usr/bin/env python3
"""Cross-check the hardcoded numbers in the paper prose against the JSON results."""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
SEC = ROOT / "regional_bias_paper/paper/sections"
MODELS = ["resnet","convnext","deit","swin","dino","mae","clip","siglip"]
DS = ["dollarstreet","geode"]

def load(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

m = {}
for d in DS:
    for k in MODELS:
        ft = load(R/d/k/"test_results.json")
        pf = load(R/"imagenet"/f"{d}_{k}.json")
        pe = pf["splits"]["test"] if "splits" in pf else pf
        m[(d,k)] = dict(fa=ft["object_acc"]*100, fg=ft["region_bias_gap"]*100,
                        pa=pe["object_acc"]*100, pg=pe["region_bias_gap"]*100)

checks = []
def chk(name, got, exp, tol=0.01):
    ok = abs(got-exp) <= tol
    checks.append((name, got, exp, ok))

# headline numbers used across abstract/results/conclusion
chk("ConvNeXt DS ft acc", m[("dollarstreet","convnext")]["fa"], 76.67)
chk("ConvNeXt GeoDE ft acc", m[("geode","convnext")]["fa"], 96.61)
chk("DeiT GeoDE ft acc", m[("geode","deit")]["fa"], 96.28)
chk("DeiT GeoDE pre acc", m[("geode","deit")]["pa"], 1.77, 0.02)
chk("ConvNeXt DS pre acc", m[("dollarstreet","convnext")]["pa"], 1.60, 0.02)
chk("ConvNeXt DS ft gap", m[("dollarstreet","convnext")]["fg"], 8.48)
chk("ConvNeXt GeoDE ft gap", m[("geode","convnext")]["fg"], 2.29)
chk("MAE DS ft acc", m[("dollarstreet","mae")]["fa"], 66.74)
chk("MAE GeoDE ft acc", m[("geode","mae")]["fa"], 92.86)
chk("MAE DS ft gap", m[("dollarstreet","mae")]["fg"], 6.88)
chk("MAE GeoDE ft gap", m[("geode","mae")]["fg"], 1.87)
chk("ResNet DS ft gap (max DS)", m[("dollarstreet","resnet")]["fg"], 12.20)
chk("CLIP GeoDE ft gap (max GeoDE)", m[("geode","clip")]["fg"], 3.06)

# dataset-level means (8 models)
for d, ea, eg in [("dollarstreet",72.02,8.62),("geode",95.01,2.39)]:
    chk(f"{d} mean ft acc", sum(m[(d,k)]["fa"] for k in MODELS)/8, ea)
    chk(f"{d} mean ft gap", sum(m[(d,k)]["fg"] for k in MODELS)/8, eg)

# paradigm means
PAR = {"Conv":["resnet","convnext"],"Trans":["deit","swin"],
       "Self":["dino","mae"],"VLM":["clip","siglip"]}
exp_acc = {"Conv":80.75,"Trans":85.93,"Self":82.24,"VLM":85.14}
exp_gap = {"Conv":6.41,"Trans":5.07,"Self":4.71,"VLM":5.84}
for p, ks in PAR.items():
    a = sum(m[(d,k)]["fa"] for k in ks for d in DS)/4
    g = sum(m[(d,k)]["fg"] for k in ks for d in DS)/4
    chk(f"paradigm {p} mean acc", a, exp_acc[p])
    chk(f"paradigm {p} mean gap", g, exp_gap[p])

# all 16 pairs: ft acc > pre acc, and ft gap > pre gap
for (d,k),v in m.items():
    chk(f"{d}/{k} acc rises", 1.0 if v["fa"]>v["pa"] else 0.0, 1.0)
    chk(f"{d}/{k} gap rises", 1.0 if v["fg"]>v["pg"] else 0.0, 1.0)

fails = [c for c in checks if not c[3]]
for name, got, exp, ok in checks:
    if not ok:
        print(f"  MISMATCH {name}: got {got:.4f} expected {exp:.4f}")
print(f"{len(checks)-len(fails)}/{len(checks)} checks passed")
sys.exit(1 if fails else 0)
