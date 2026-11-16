"""Writes tab_overall.tex from reports/week9_results.json (usage: python make_tables.py results.json outdir)."""
import json
import sys
from pathlib import Path

R = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
MODELS = [("majority", "Majority"), ("logreg", "LogReg"), ("logreg_balanced", "LogReg bal."), ("rf", "Random forest")]


def cell(metric, wt, model, st, tau=10, pct=False):
    v = R["overall"][wt][metric].get(f"{tau}|{model}|{st}")
    if not v or v[0] is None:
        return "--"
    mu, sd = v
    sd = sd or 0
    if pct:
        return f"{100 * mu:.1f}$\\pm${100 * sd:.1f}"
    return f"{mu:.3f}$\\pm${sd:.3f}"


lines = [r"\begin{table*}[t]\centering\footnotesize",
         r"\caption{Naive results, $\tau=10$, mean $\pm$ SD over 15 splits (5 grouped folds $\times$ 3 seeds). "
         r"F = held-out female types (within-sex), M = the same types in the male brain (cross-sex). "
         r"Cost = mean locked flow cost per neuron (0 = all correct); accuracy, macro-F1 and balanced accuracy in \%. "
         r"Top block type-weighted (headline), bottom block neuron-weighted.}",
         r"\label{tab:overall}",
         r"\begin{tabular}{@{}l cc cc cc cc@{}}\toprule",
         r" & \multicolumn{2}{c}{Expected cost $\downarrow$} & \multicolumn{2}{c}{Accuracy} & "
         r"\multicolumn{2}{c}{Macro-F1} & \multicolumn{2}{c}{Balanced acc.}\\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"Model & F & M & F & M & F & M & F & M\\\midrule"]
for wt in ("type", "neuron"):
    lines.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{{wt}-weighted}}}}\\")
    for key, name in MODELS:
        row = [name]
        for metric, pct in (("cost_flow125", False), ("accuracy", True), ("macro_f1", True),
                            ("balanced_accuracy", True)):
            row += [cell(metric, wt, key, "within", pct=pct), cell(metric, wt, key, "cross", pct=pct)]
        lines.append(" & ".join(row) + r"\\")
    if wt == "type":
        lines.append(r"\midrule")
lines += [r"\bottomrule\end{tabular}\end{table*}"]
(out / "tab_overall.tex").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
