
import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = ["apply_viz_defaults"]

def apply_viz_defaults(use_tex=True):
    if use_tex:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["font.family"] = "serif"

    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
