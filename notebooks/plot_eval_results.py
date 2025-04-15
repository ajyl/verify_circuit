import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)


matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "text.latex.preamble": r"\usepackage{xcolor}",
    }
)
import matplotlib.pyplot as plt

#matplotlib.rcParams.update({
#    "text.latex.preamble": r"\usepackage{xcolor}"
#})

#interv_results = [0, 17.3, 47.3, 100]
# [null interv, MLP1, MLP_both, Prev_Token_attns, Verif_attns]
interv_results = [0, 16.3, 33, 93.33, 92.33]
mix = [0, 1, 10.67, 6, 7.33]
# this_percentage = [0, 0.826, ?, 0]
labels = [
    "Null\nInterv.",
    "Interv.\n" + r"$\textit{GLU}_{this}$",
    "Interv.\n" + r"$\textit{GLU}_{this}+$" + "\n" + r"$\textit{GLU}_{not}$",
    "Interv.\nAttn",
]


fig, (ax, ax2) = plt.subplots(
    1, 2, figsize=(6, 2.5), gridspec_kw={"width_ratios": [1, 1]}
)

ax.bar(range(len(interv_results)), interv_results)
ax.set_ylabel("Success Rate (\%)")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.tick_params(axis='x', labelsize=8)
ax.set_title("Intervention Success Rate")

text = (
    "Assistant: Let me solve this step by step.\n"
    r"$<$think$>$ "
    "We have the numbers 11, 5, and 68.\n"
    "We need to make an equation that equals 62.\n"
    "Let's try different combinations:\n"
    "68 - 11 - 5 = 52 - 5 = 47 (not 62)\n"
    r"68 - 11 + 5 = 57 + 5 = 62 $\color{red}{\textit{(not 62 - 11 + 5)}}$"
    "\n"
    r"68 - 11 + 5 = 57 + 5 = 62 $\color{red}{\textit{(not 62 + 11 - 5)}}$"
    "\n"
    "68 + 11 - 5 = 79 - 5 = 74 (not 62)\n"
    "68 + 11 + 5 = 79 + 5 = 84 (not 62)\n"
    "68 * 11 - 5 = 748 - 5 = 743 (not 62)\n"
    "68 * 11 / 5 = 748 / 5 = 149.6 (not 62)\n"
    "68 / 11 + 5 = 6.18 + 5 = 11.18 (not 62)\n"
    "68 / 11 + ..."
)

ax2.axis("off")
ax2.text(
    0,
    1,
    text,
    fontsize=8,
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.3", edgecolor="black", facecolor="none", linewidth=1
    ),
)
ax2.set_title("Intervened Output")

plt.tight_layout()
plt.savefig("interv_results.pdf")
