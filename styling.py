import seaborn as sns

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
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'axes.labelpad': 2,
    'axes.linewidth': 0.5,
    'axes.titlepad': 4,
    'grid.linewidth': 0.4,
    'lines.linewidth': 0.5,
    'patch.linewidth': 0.5,
    'legend.fontsize': 6,
    'legend.title_fontsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
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
annotation_kwargs = {'fontsize': 7}

panel_labels = 'ABCDEFGHIJK'
# panel_labels = 'abcdefghijk'
fig_width = 8
