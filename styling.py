import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Source: https://atchen.me/research/code/data-viz/2022/01/04/plotting-matplotlib-reference.html
# with alterations and additions.

offblack = '#373737'
sns.set_theme(style='ticks', font_scale=0.75, rc={
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'svg.fonttype': 'none',
    'text.usetex': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.labelpad': 2,
    'axes.linewidth': 0.5,
    'axes.titlepad': 4,
    'grid.linewidth': 0.4,
    'lines.linewidth': 0.5,
    'patch.linewidth': 0.5,
    'legend.fontsize': 8,
    'legend.title_fontsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.size': 2,
    'xtick.major.pad': 1,
    'xtick.major.width': 0.5,
    'ytick.major.size': 2,
    'ytick.major.pad': 1,
    'ytick.major.width': 0.5,
    'xtick.minor.size': 2,
    'xtick.minor.pad': 1,
    'xtick.minor.width': 0.5,
    'ytick.minor.size': 2,
    'ytick.minor.pad': 1,
    'ytick.minor.width': 0.5,

    'boxplot.flierprops.color':           offblack,
    'boxplot.flierprops.markeredgecolor': offblack,
    'boxplot.flierprops.markeredgewidth': 0.5,
    'boxplot.flierprops.markersize':      3,
    'boxplot.flierprops.linewidth':       0.5,
    'boxplot.boxprops.color':     offblack,
    'boxplot.boxprops.linewidth': 0.5,
    'boxplot.whiskerprops.color':     offblack,
    'boxplot.whiskerprops.linewidth': 0.5,
    'boxplot.capprops.color':     offblack,
    'boxplot.capprops.linewidth': 0.5,
    'boxplot.medianprops.linewidth': 0.5,

    # Avoid black unless necessary
    'text.color': offblack,
    'patch.edgecolor': offblack,
    'hatch.color': offblack,
    'axes.edgecolor': offblack,
    'axes.labelcolor': offblack,
    'xtick.color': offblack,
    'ytick.color': offblack,
})

label_kwargs = {'fontsize': 12, 'fontweight': 'bold'}
annotation_kwargs = {'fontsize': 9}
axline_kwargs = {'color': 'darkgrey', 'zorder': -1, 'lw': sns.mpl.rcParams['axes.linewidth']}

cmap_seq = 'viridis'
cmap_div = LinearSegmentedColormap.from_list(
# From http://www.ccctool.com/html_v_0_9_0_3/CCC_Tool/cccTool.html
    'ccc_divergent', list(zip([0, .16, .35, .5, .62, .8, 1],
        np.asarray([
            [.0862745098039216,0.00392156862745098,0.298039215686275, 1.],
            [.054902,0.317647,0.709804, 1.],
            [.0705882,0.854902,0.870588, 1.],
            [1, 1, 1, 1.],
            [.94902,0.823529,0.321569, 1.],
            [.811765,0.345098,0.113725, 1.],
            [.188235294117647,0,0.0705882352941176, 1.]
        ]))))

panel_labels = 'ABCDEFGHIJK'
# panel_labels = 'abcdefghijk'
fig_width = 8
