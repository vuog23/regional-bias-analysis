#!/usr/bin/env python3
"""Parse the generated LaTeX tables cell-by-cell and verify against the JSON results."""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
T = ROOT / "regional_bias_paper/paper/tables"
MODELS = ["resnet","convnext","deit","swin","dino","mae","clip","siglip"]
NAME = {"resnet":"ResNet50","convnext":"ConvNeXt V2","deit":"DeiT III","swin":"SwinV2",
        "dino":"DINOv2","mae":"MAE","clip":"CLIP","siglip":"SigLIP"}
DS = [("dollarstreet","Dollar Street"),("geode","GeoDE")]
RP = {"af":"Africa","am":"Americas","as":"Asia","eu":"Europe","Africa":"Africa",
      "Americas":"Americas","EastAsia":"East Asia","Europe":"Europe",
      "SouthEastAsia":"Southeast Asia","WestAsia":"West Asia"}

def load(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

m = {}
for dk,_ in DS:
    for k in MODELS:
        ft = load(R/dk/k/"test_results.json")
        pf = load(R/"imagenet"/f"{dk}_{k}.json")
        pe = pf["splits"]["test"] if "splits" in pf else pf
        m[(dk,k)] = dict(
            fa=ft["object_acc"]*100, fg=ft["region_bias_gap"]*100,
            pa=pe["object_acc"]*100, pg=pe["region_bias_gap"]*100,
            fbest=RP[ft["best_region"]], fba=ft["best_region_acc"]*100,
            fworst=RP[ft["worst_region"]], fwa=ft["worst_region_acc"]*100,
            pbest=RP[pe["best_region"]], pba=pe["best_region_acc"]*100,
            pworst=RP[pe["worst_region"]], pwa=pe["worst_region_acc"]*100)

fails = []
def eq(name, a, b, tol=0.005):
    if isinstance(a,str) or isinstance(b,str):
        if str(a) != str(b): fails.append(f"{name}: table={a!r} json={b!r}")
    elif abs(a-b) > tol:
        fails.append(f"{name}: table={a:.4f} json={b:.4f}")

def rows(fn):
    txt = (T/fn).read_text(encoding="utf-8")
    out = []
    for ln in txt.splitlines():
        if "\\quad" in ln and "&" in ln:
            cells = [c.strip().replace("\\quad","").strip()
                     for c in ln.split("\\\\")[0].split("&")]
            out.append(cells)
    return out

# ---- main comparison: Model, Paradigm, preAcc, ftAcc, dAcc, preGap, ftGap, dGap
cur = None
order = iter([(d,k) for d,_ in DS for k in MODELS])
txt = (T/"final_main_comparison_table.tex").read_text(encoding="utf-8")
seq = []
for ln in txt.splitlines():
    if "\\textit{Dollar Street}" in ln: cur="dollarstreet"
    elif "\\textit{GeoDE}" in ln: cur="geode"
    elif "\\quad" in ln and "&" in ln:
        c = [x.strip().replace("\\quad","").strip() for x in ln.split("\\\\")[0].split("&")]
        k = [kk for kk in MODELS if NAME[kk]==c[0]][0]
        v=m[(cur,k)]
        eq(f"main {cur}/{k} preAcc", float(c[2]), v["pa"], 0.01)
        eq(f"main {cur}/{k} ftAcc", float(c[3]), v["fa"], 0.01)
        eq(f"main {cur}/{k} dAcc", float(c[4]), v["fa"]-v["pa"], 0.01)
        eq(f"main {cur}/{k} preGap", float(c[5]), v["pg"], 0.01)
        eq(f"main {cur}/{k} ftGap", float(c[6]), v["fg"], 0.01)
        eq(f"main {cur}/{k} dGap", float(c[7]), v["fg"]-v["pg"], 0.01)

# ---- best/worst: Model, State, Best, BestAcc, Worst, WorstAcc, Gap
cur=None
txt = (T/"final_best_worst_regions_table.tex").read_text(encoding="utf-8")
for ln in txt.splitlines():
    if "\\textit{Dollar Street}" in ln: cur="dollarstreet"
    elif "\\textit{GeoDE}" in ln: cur="geode"
    elif "\\quad" in ln and "&" in ln:
        c=[x.strip().replace("\\quad","").strip() for x in ln.split("\\\\")[0].split("&")]
        k=[kk for kk in MODELS if NAME[kk]==c[0]][0]; v=m[(cur,k)]
        if c[1]=="Pretrained":
            eq(f"bw {cur}/{k} pre best", c[2], v["pbest"]); eq(f"bw {cur}/{k} pre bestacc", float(c[3]), v["pba"],0.01)
            eq(f"bw {cur}/{k} pre worst", c[4], v["pworst"]); eq(f"bw {cur}/{k} pre worstacc", float(c[5]), v["pwa"],0.01)
            eq(f"bw {cur}/{k} pre gap", float(c[6]), v["pg"],0.01)
        else:
            eq(f"bw {cur}/{k} ft best", c[2], v["fbest"]); eq(f"bw {cur}/{k} ft bestacc", float(c[3]), v["fba"],0.01)
            eq(f"bw {cur}/{k} ft worst", c[4], v["fworst"]); eq(f"bw {cur}/{k} ft worstacc", float(c[5]), v["fwa"],0.01)
            eq(f"bw {cur}/{k} ft gap", float(c[6]), v["fg"],0.01)

# ---- tradeoff: Model, Paradigm, ftAcc, accRank, ftGap, gapRank
cur=None
txt=(T/"final_tradeoff_ranking_table.tex").read_text(encoding="utf-8")
buf={}
for ln in txt.splitlines():
    if "\\textit{Dollar Street}" in ln: cur="dollarstreet"; buf[cur]=[]
    elif "\\textit{GeoDE}" in ln: cur="geode"; buf[cur]=[]
    elif "\\quad" in ln and "&" in ln:
        c=[x.strip().replace("\\quad","").strip() for x in ln.split("\\\\")[0].split("&")]
        buf[cur].append(c)
for dk,_ in DS:
    rs=buf[dk]
    acc_sorted=sorted(rs,key=lambda c:-float(c[2]))
    gap_sorted=sorted(rs,key=lambda c:float(c[4]))
    for c in rs:
        k=[kk for kk in MODELS if NAME[kk]==c[0]][0]; v=m[(dk,k)]
        eq(f"trade {dk}/{k} ftAcc", float(c[2]), v["fa"],0.01)
        eq(f"trade {dk}/{k} ftGap", float(c[4]), v["fg"],0.01)
        eq(f"trade {dk}/{k} accRank", int(c[3]), acc_sorted.index(c)+1)
        eq(f"trade {dk}/{k} gapRank", int(c[5]), gap_sorted.index(c)+1)

# ---- paradigm summary means
PAR={"Convolutional Vision":["resnet","convnext"],"Transformer Vision":["deit","swin"],
     "Self-supervised Foundation":["dino","mae"],"Vision-language Foundation":["clip","siglip"]}
txt=(T/"final_paradigm_summary_table.tex").read_text(encoding="utf-8")
for ln in txt.splitlines():
    for p,ks in PAR.items():
        if ln.startswith(p+" &"):
            c=[x.strip() for x in ln.split("\\\\")[0].split("&")]
            a=sum(m[(d,k)]["fa"] for k in ks for d,_ in DS)/4
            g=sum(m[(d,k)]["fg"] for k in ks for d,_ in DS)/4
            eq(f"paradigm {p} meanAcc", float(c[2]), a, 0.01)
            eq(f"paradigm {p} meanGap", float(c[3]), g, 0.01)

for f in fails: print("  MISMATCH", f)
total = "all cells" if not fails else f"{len(fails)} mismatches"
print(f"Table cross-check: {total}")
sys.exit(1 if fails else 0)
