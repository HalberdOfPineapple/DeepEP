"""Shared plotting helpers: matplotlib -> inline SVG, reference palette (dataviz skill), thin marks, direct labels."""
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, MUTED, GRID = '#17202A', '#5A6572', '#D5DBE2'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['IBM Plex Sans', 'DejaVu Sans'], 'font.size': 10,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'legend.frameon': False,
    'lines.linewidth': 2, 'lines.markersize': 5, 'svg.fonttype': 'none', 'figure.facecolor': 'none', 'axes.facecolor': 'none',
})


def fig(w=8.6, h=3.6, ncols=1, nrows=1, **kw):
    f, ax = plt.subplots(nrows, ncols, figsize=(w, h), **kw)
    return f, ax


def _wrap_titles(f):
    import textwrap
    W = f.get_size_inches()[0]
    for ax in f.axes:
        t = ax.get_title(loc='left')
        if not t: continue
        width_in = ax.get_position().width * W
        maxc = max(18, int(width_in * 8.0))  # ~8 bold 11pt chars per inch
        if len(t) > maxc: ax.set_title('\n'.join(textwrap.wrap(t, maxc)), loc='left')


def _repel_end_labels(f, dpi=100, min_px=13.0):
    """Spread the end labels registered by end_label() so they do not overlap, and make room on the right."""
    f.canvas.draw()
    for ax in f.axes:
        labs = getattr(ax, '_endlabels', [])
        if not labs: continue
        # desired pixel y of each label anchor
        items = []
        for ann in labs:
            px = ax.transData.transform(ann.xy)
            items.append([px[1], ann])
        items.sort(key=lambda t: t[0])
        ys = [t[0] for t in items]
        # forward pass: enforce min spacing from the bottom up, then shift the whole cluster down by half the overflow
        new = list(ys)
        for i in range(1, len(new)):
            if new[i] - new[i - 1] < min_px: new[i] = new[i - 1] + min_px
        shift = (new[-1] - ys[-1]) / 2 if len(new) > 1 else 0
        new = [y - shift for y in new]
        for (y0, ann), y1 in zip(items, new):
            dx = ann.xyann[0] if ann.xycoords != 'data' else 4
            ann.xyann = (dx, (y1 - y0) * 72.0 / dpi)
            ann.textcoords = 'offset points'
        # extend the x range so labels fit: longest label in characters
        longest = max(len(ann.get_text()) for ann in labs)
        frac = min(0.55, 0.05 + longest * 0.018)
        x0, x1 = ax.get_xlim()
        if ax.get_xscale() == 'log':
            import math
            ax.set_xlim(x0, x1 * (x1 / x0) ** frac)
        else:
            ax.set_xlim(x0, x1 + (x1 - x0) * frac)


def legend_below(ax, ncol=2):
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=ncol, frameon=False)


def to_svg(f, dpi=100):
    _wrap_titles(f)
    _repel_end_labels(f, dpi)
    buf = io.StringIO()
    f.savefig(buf, format='svg', bbox_inches='tight', dpi=dpi, transparent=True)
    plt.close(f)
    s = buf.getvalue()
    i = s.find('<svg'); s = s[i:]
    s = s.replace('<svg ', '<svg class="chart" ', 1)
    return s


def end_label(ax, x, y, text, color, dx=4, **kw):
    ann = ax.annotate(text, (x, y), xytext=(dx, 0), textcoords='offset points', va='center', ha='left', fontsize=9, color=INK, **kw)
    if not hasattr(ax, '_endlabels'): ax._endlabels = []
    ax._endlabels.append(ann)
    return ann


def pct(xs, p):
    xs = np.sort(np.asarray(xs)); return float(xs[int(round(p / 100 * (len(xs) - 1)))]) if len(xs) else float('nan')
