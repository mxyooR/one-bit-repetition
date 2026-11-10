"""Render the recorded adaptive-design study without generating observations."""

import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT/"experiments/adaptive_results/results.json").read_text(encoding="utf-8"))
ROWS = DATA["results"]
FIG = ROOT/"output/figures"
BUILD = ROOT/"output/previews"
FIG.mkdir(parents=True,exist_ok=True)
BUILD.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({
    "font.family":"serif", "font.serif":["Times New Roman"],
    "mathtext.fontset":"stix", "font.size":10, "axes.labelsize":10,
    "legend.fontsize":10, "xtick.labelsize":10, "ytick.labelsize":10,
    "pdf.fonttype":42, "ps.fonttype":42, "axes.spines.top":False,
    "axes.spines.right":False, "lines.linewidth":1.3,
})
METHODS = ["two_stage","dyadic","fixed_2","fixed_100","known_a"]
LABELS = ["Two-stage","Predetermined multiscale","Fixed $r=2$",
          "Fixed $r=100$","Known reliabilities"]
COLORS = ["#0072B2","#D55E00","#666666","#AA4499","#009E73"]
MARKERS = ["o","s","v","x","^"]
STYLES = ["-","--",":","-.", "-"]


def finish(fig,name):
    for ext in ["pdf","svg"]:
        fig.savefig(FIG/f"{name}.{ext}")
    fig.savefig(BUILD/f"{name}.png",dpi=180)
    plt.close(fig)


def legend(fig):
    handles = [Line2D([],[],color=c,marker=m,linestyle=s,markersize=4,label=l)
               for c,m,s,l in zip(COLORS,MARKERS,STYLES,LABELS)]
    fig.legend(handles=handles,loc="lower center",ncol=3,frameon=False,
               bbox_to_anchor=(.53,-.005),columnspacing=1.4,handlelength=2.1)


def plot_rows(ax,rows,xfield):
    for method,c,m,s in zip(METHODS,COLORS,MARKERS,STYLES):
        rr = sorted([r for r in rows if r["method"]==method],key=lambda r:r[xfield])
        x = [r[xfield] for r in rr]
        y = [r["normalized_mse"] for r in rr]
        e = [1.96*r["B"]*min(r["a1"],r["a2"])**2*r["mcse"] for r in rr]
        assert np.isfinite(y).all() and min(y)>0
        ax.errorbar(x,y,yerr=e,color=c,marker=m,linestyle=s,
                    markersize=3.5,capsize=2)
    ax.set(xscale="log",yscale="log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=.22,which="major",linewidth=.4)


fig,axes=plt.subplots(1,2,figsize=(6.5,2.85),sharey=True)
rows=[r for r in ROWS if r["B"]==262144 and r["rho"]==.6]
plot_rows(axes[0],[r for r in rows if r["family"]=="equal"],"a1")
plot_rows(axes[1],[r for r in rows if r["family"]=="unequal"],"a1")
axes[0].set_title(r"(a) $a_1=a_2$",fontsize=10)
axes[1].set_title(r"(b) $a_2=0.8$",fontsize=10)
axes[0].set_xticks([1/64,1/16,1/4,1],[r"$1/64$",r"$1/16$",r"$1/4$","1"])
axes[1].set_xticks([1/32,1/8,1/2],[r"$1/32$",r"$1/8$",r"$1/2$"])
for ax in axes:
    ax.set_xlabel(r"Channel reliability, $a_1$")
    ax.set_ylim(2.3,600)
axes[0].set_ylabel("Scaled mean squared error")
fig.subplots_adjust(left=.10,right=.99,bottom=.33,top=.89,wspace=.20)
legend(fig)
finish(fig,"adaptive-risk")

fig,axes=plt.subplots(1,2,figsize=(6.5,2.85),sharey=True)
for ax,rho in zip(axes,[0.,.6]):
    rr=[r for r in ROWS if r["a1"]==r["a2"]==.125 and r["rho"]==rho]
    plot_rows(ax,rr,"B")
    ax.set_title(rf"$a_1=a_2=1/8,\ \rho={rho:g}$",fontsize=10)
    ax.set_xticks([2**14,2**18,2**22],[r"$2^{14}$",r"$2^{18}$",r"$2^{22}$"])
    ax.set_xlabel(r"Observation budget, $B$")
    ax.set_ylim(7,400)
axes[0].set_ylabel("Scaled mean squared error")
fig.subplots_adjust(left=.10,right=.99,bottom=.33,top=.89,wspace=.20)
legend(fig)
finish(fig,"adaptive-budget")

header = [
    r"\begin{longtable}{rrrccccc}",
    r"\caption{Equal-channel results. Entries are $Ba^2\operatorname{MSE}$; parentheses give the Monte Carlo standard error in the same units. Each entry uses 2000 trials.}\label{tab:equal}\\",
    r"\toprule $B$ & $a$ & $\rho$ & Two-stage & Multiscale & $r=2$ & $r=100$ & Known $a_i$\\",
    r"\midrule\endfirsthead",
    r"\multicolumn{8}{l}{Table~\thetable\ (continued): equal-channel scaled errors and MCSE.}\\",
    r"\toprule $B$ & $a$ & $\rho$ & Two-stage & Multiscale & $r=2$ & $r=100$ & Known $a_i$\\",
    r"\midrule\endhead",
    r"\bottomrule\endfoot",
]
lines=list(header)
for B in DATA["protocol"]["budgets"]:
    for rho in DATA["protocol"]["correlations"]:
        for a in DATA["protocol"]["equal_reliabilities"]:
            vals=[]
            for method in METHODS:
                rec=next(r for r in ROWS if (r["B"],r["a1"],r["a2"],r["rho"],r["method"])==(B,a,a,rho,method))
                vals.append(f'{rec["normalized_mse"]:.3g} ({B*a*a*rec["mcse"]:.2g})')
            lines.append(f'$2^{{{int(np.log2(B))}}}$ & {a:g} & {rho:g} & '+" & ".join(vals)+r"\\")
        lines.append(r"\midrule")
lines[-1]=r"\end{longtable}"
(ROOT/"output/adaptive-results-table.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")

lines=[r"\begin{table}[ht]",r"\centering",
       r"\caption{Unequal-channel results, $B=2^{18}$, $a_2=0.8$, $\rho=0.6$. Entries use the same scaled error and uncertainty units as Table~\ref{tab:equal}.}\label{tab:unequal}",
       r"\begin{tabular}{rccccc}",r"\toprule $a_1$ & Two-stage & Multiscale & $r=2$ & $r=100$ & Known $a_i$\\",r"\midrule"]
for a in [1/32,1/8,1/2]:
    vals=[]
    for method in METHODS:
        rec=next(r for r in ROWS if r["family"]=="unequal" and r["a1"]==a and r["method"]==method)
        vals.append(f'{rec["normalized_mse"]:.3g} ({rec["B"]*a*a*rec["mcse"]:.2g})')
    lines.append(f"{a:g} & "+" & ".join(vals)+r"\\")
lines.extend([r"\bottomrule\end{tabular}",r"\end{table}"])
(ROOT/"output/adaptive-unequal-table.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")

fields=["id","family","B","a1","a2","rho","method","mse","mcse","normalized_mse",
        "bias","sample_variance","used_bits_min","used_bits_max"]
with (ROOT/"experiments/adaptive_results/summary.csv").open("w",newline="",encoding="utf-8") as f:
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
    writer.writeheader();writer.writerows(ROWS)
print("Rendered two figures and complete result tables from the recorded study.")
