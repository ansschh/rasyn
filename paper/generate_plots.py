"""Generate all publication-quality figures for the Rasyn paper.

Parses REAL training logs and generates clean, minimal figures.

Color scheme (strict):
  Primary:   #1565C0 (science blue)
  Secondary: #42A5F5 (light blue)
  Text:      #212121 (near-black)
  Background: white
  Gray:      #9E9E9E (for "other" methods only)

NO orange, green, red, or other colors.
NO annotations, arrows, stars, or "best" markers.

Exports to paper/figures/ as both PDF and PNG (300 DPI).
"""

import matplotlib
matplotlib.use('Agg')

import re
import ast
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ── Try to load Montserrat font ─────────────────────────────────
def setup_font():
    """Attempt to use Montserrat; fall back to a clean sans-serif."""
    from matplotlib import font_manager as fm

    # Check if Montserrat is already available
    available = {f.name for f in fm.fontManager.ttflist}
    if 'Montserrat' in available:
        return 'Montserrat'

    # Try to download Montserrat from Google Fonts CSS API
    try:
        import urllib.request
        import tempfile

        font_dir = os.path.join(tempfile.gettempdir(), 'montserrat_fonts')
        os.makedirs(font_dir, exist_ok=True)

        # Step 1: Get CSS with TTF URLs (use woff2 user-agent trick for ttf)
        css_url = 'https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700'
        req = urllib.request.Request(css_url, headers={
            'User-Agent': 'Mozilla/4.0'  # Old UA gets TTF instead of woff2
        })
        css = urllib.request.urlopen(req, timeout=10).read().decode()

        # Step 2: Extract font URLs from CSS
        import re as _re
        font_urls = _re.findall(r'url\((https://[^)]+\.ttf)\)', css)
        if not font_urls:
            font_urls = _re.findall(r'url\((https://[^)]+)\)', css)

        # Step 3: Download each font file
        for i, furl in enumerate(font_urls):
            fname = f'Montserrat-{i}.ttf'
            dest = os.path.join(font_dir, fname)
            if not os.path.exists(dest) or os.path.getsize(dest) < 1000:
                print(f'  Downloading Montserrat weight {i}...')
                urllib.request.urlretrieve(furl, dest)

        # Step 4: Register all TTF files
        for f_name in os.listdir(font_dir):
            fpath = os.path.join(font_dir, f_name)
            if f_name.endswith('.ttf') and os.path.getsize(fpath) > 1000:
                fm.fontManager.addfont(fpath)

        available = {f.name for f in fm.fontManager.ttflist}
        if 'Montserrat' in available:
            print('  Montserrat font loaded successfully.')
            return 'Montserrat'
        else:
            print('  Downloaded fonts but Montserrat not recognized.')
    except Exception as e:
        print(f'  Could not download Montserrat: {e}')

    # Fallback
    print('  Using sans-serif fallback font.')
    return 'sans-serif'


FONT_FAMILY = setup_font()

# ── Color Palette (strict) ──────────────────────────────────────
PRIMARY   = '#1565C0'   # science blue
SECONDARY = '#42A5F5'   # light blue
DARK      = '#212121'   # near-black for text
GRAY      = '#9E9E9E'   # for "other" methods
LIGHT_GRAY = '#E0E0E0'  # very light gray for backgrounds
WHITE     = '#FFFFFF'

# ── Global matplotlib style ─────────────────────────────────────
plt.rcParams.update({
    'font.family': FONT_FAMILY if FONT_FAMILY != 'sans-serif' else 'sans-serif',
    'font.sans-serif': ['Montserrat', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': GRAY,
    'axes.linewidth': 0.6,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.4,
    'axes.facecolor': WHITE,
    'figure.facecolor': WHITE,
    'text.color': DARK,
    'axes.labelcolor': DARK,
    'xtick.color': DARK,
    'ytick.color': DARK,
})

# ── Paths ────────────────────────────────────────────────────────
PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(PAPER_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

RETRO_V2_ORIGINAL_LOG = os.path.join(PAPER_DIR, 'retro_v2_full_epochs.txt')
RETRO_V2_RESUME_LOG   = os.path.join(PAPER_DIR, 'raw_retro_v2_log.txt')
LLM_V6_LOG            = os.path.join(PAPER_DIR, 'llm_v6_loss_data.txt')


def save(fig, name):
    fig.savefig(os.path.join(FIGURES_DIR, f'{name}.pdf'), format='pdf')
    fig.savefig(os.path.join(FIGURES_DIR, f'{name}.png'), format='png')
    plt.close(fig)
    print(f'  Saved {name}.pdf / .png')


# ════════════════════════════════════════════════════════════════
# LOG PARSING FUNCTIONS
# ════════════════════════════════════════════════════════════════

def parse_retro_v2_step_log(filepath):
    """Parse step-level training log lines.

    Returns list of dicts with keys: step, epoch, loss, avg, tok_acc, copy, lr
    """
    step_pattern = re.compile(
        r'Step\s+(\d+)\s*\|\s*Epoch\s+(\d+)\s*\|'
        r'\s*loss=([\d.]+)\s*\|\s*avg=([\d.]+)\s*\|'
        r'\s*tok_acc=([\d.]+)\s*\|\s*copy=([\d.]+)\s*\|'
        r'\s*lr=([\d.eE+-]+)'
    )
    epoch_pattern = re.compile(
        r'---\s*Epoch\s+(\d+)/\d+\s*---\s*avg_loss=([\d.]+)\s*tok_acc=([\d.]+)'
    )
    val_pattern = re.compile(
        r'VAL:\s*loss=([\d.]+)\s*\|\s*tok_acc=([\d.]+)\s*\|\s*exact=([\d.]+)\s*\((\d+)/(\d+)\)'
    )
    best_pattern = re.compile(
        r'New best!\s*val_loss=([\d.]+)\s*exact=([\d.]+)'
    )

    steps = []
    epoch_summaries = []
    val_results = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = step_pattern.search(line)
            if m:
                steps.append({
                    'step': int(m.group(1)),
                    'epoch': int(m.group(2)),
                    'loss': float(m.group(3)),
                    'avg': float(m.group(4)),
                    'tok_acc': float(m.group(5)),
                    'copy': float(m.group(6)),
                    'lr': float(m.group(7)),
                })
                continue

            m = epoch_pattern.search(line)
            if m:
                epoch_summaries.append({
                    'epoch': int(m.group(1)),
                    'avg_loss': float(m.group(2)),
                    'tok_acc': float(m.group(3)),
                })
                continue

            m = val_pattern.search(line)
            if m:
                val_results.append({
                    'val_loss': float(m.group(1)),
                    'val_tok_acc': float(m.group(2)),
                    'exact': float(m.group(3)),
                    'correct': int(m.group(4)),
                    'total': int(m.group(5)),
                })
                continue

            m = best_pattern.search(line)
            if m:
                if val_results:
                    val_results[-1]['is_best'] = True

    return steps, epoch_summaries, val_results


def get_epoch_end_values(steps):
    """From step-level data, extract the last step of each epoch."""
    epoch_data = {}
    for s in steps:
        epoch_data[s['epoch']] = s  # Last step wins
    epochs_sorted = sorted(epoch_data.keys())
    return [epoch_data[e] for e in epochs_sorted]


def parse_llm_v6_log(filepath):
    """Parse LLM v6 loss data with format: {'loss': '...', 'epoch': '...', ...}"""
    entries = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                d = ast.literal_eval(line)
                entries.append({
                    'loss': float(d['loss']),
                    'epoch': float(d['epoch']),
                })
            except (ValueError, KeyError, SyntaxError):
                continue
    return entries


# ════════════════════════════════════════════════════════════════
# Figure 1: Rasyn Architecture Overview
# ════════════════════════════════════════════════════════════════
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Box definitions: (x, y, w, h, label, facecolor, textcolor)
    boxes = [
        (0.2,  1.5, 1.6, 1.0, 'Product\nSMILES',     LIGHT_GRAY, DARK),
        (2.3,  1.5, 1.6, 1.0, 'Graph Head\n(GNN)',     SECONDARY,  WHITE),
        (4.4,  2.3, 1.6, 1.0, 'RSGPT v6\n(LLM)',      PRIMARY,    WHITE),
        (4.4,  0.7, 1.6, 1.0, 'RetroTx v2\n(Seq2Seq)', PRIMARY,   WHITE),
        (6.6,  1.5, 1.6, 1.0, 'Ensemble\n& Verify',   SECONDARY,  WHITE),
        (8.6,  1.5, 1.2, 1.0, 'Reactants',            LIGHT_GRAY, DARK),
    ]

    for (x, y, w, h, label, fc, tc) in boxes:
        fancy = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                               facecolor=fc, edgecolor=DARK, linewidth=0.8)
        ax.add_patch(fancy)
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color=tc)

    # Arrows (simple, clean)
    arrow_kw = dict(arrowstyle='->', color=DARK, lw=1.2)
    ax.annotate('', xy=(2.3, 2.0), xytext=(1.8, 2.0), arrowprops=arrow_kw)
    ax.annotate('', xy=(4.4, 2.8), xytext=(3.9, 2.2),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=1.2,
                                connectionstyle='arc3,rad=0.15'))
    ax.annotate('', xy=(4.4, 1.2), xytext=(3.9, 1.8),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=1.2,
                                connectionstyle='arc3,rad=-0.15'))
    ax.annotate('', xy=(6.6, 2.2), xytext=(6.0, 2.8),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=1.2,
                                connectionstyle='arc3,rad=-0.15'))
    ax.annotate('', xy=(6.6, 1.8), xytext=(6.0, 1.2),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=1.2,
                                connectionstyle='arc3,rad=0.15'))
    ax.annotate('', xy=(8.6, 2.0), xytext=(8.2, 2.0), arrowprops=arrow_kw)

    fig.suptitle('Rasyn: Hybrid Retrosynthesis Architecture', fontsize=13,
                 fontweight='bold', color=PRIMARY, y=0.96)
    save(fig, 'fig1_architecture')


# ════════════════════════════════════════════════════════════════
# Figure 2: SOTA Comparison Bar Chart
# ════════════════════════════════════════════════════════════════
def fig2_sota_comparison():
    methods = [
        'Transformer\n(2019)', 'Retroformer\n(2022)',
        'EditRetro\n(2024)', 'RetroDFM-R\n(2025)',
        'C-SMILES\n(2025)', 'RSGPT\n(2025)',
        'RetroTx v2\n(Ours)', 'RSGPT v6\n(Ours)'
    ]
    top1 = [43.7, 53.2, 60.8, 65.0, 67.2, 77.0, 56.7, 80.9]

    # Gray for others, blue shades for ours
    colors = [GRAY] * 6 + [SECONDARY, PRIMARY]
    edges  = [GRAY] * 6 + [PRIMARY, DARK]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(range(len(methods)), top1, color=colors, edgecolor=edges,
                  linewidth=0.8, width=0.7, zorder=3)

    for bar, val in zip(bars, top1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, color=DARK)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=8.5)
    ax.set_ylabel('Top-1 Exact Match (%)')
    ax.set_ylim(0, 92)
    ax.grid(axis='y')
    ax.set_title('Comparison with State-of-the-Art on USPTO-50K',
                 fontweight='bold', color=PRIMARY)

    fig.tight_layout()
    save(fig, 'fig2_sota_comparison')


# ════════════════════════════════════════════════════════════════
# Figure 3: RetroTransformer v2 Training Curves (REAL DATA)
# ════════════════════════════════════════════════════════════════
def fig3_retro_training():
    # Parse original run (epochs 1-15 of training from scratch)
    orig_steps, orig_epochs, orig_vals = parse_retro_v2_step_log(RETRO_V2_ORIGINAL_LOG)
    # Parse resume run (epochs 1-20 in resume = epochs 16-35 overall)
    res_steps, res_epochs, res_vals = parse_retro_v2_step_log(RETRO_V2_RESUME_LOG)

    # Extract epoch-end values from original log
    orig_epoch_ends = get_epoch_end_values(orig_steps)

    # Extract epoch-end values from resume log
    res_epoch_ends = get_epoch_end_values(res_steps)

    # Build combined epoch arrays
    # Original run: we have epoch-summary lines at epochs 1,2,3 (plus more in full file)
    # But for step data we can extract the last step per epoch
    orig_train_loss = [e['avg'] for e in orig_epoch_ends]
    orig_tok_acc    = [e['tok_acc'] * 100 for e in orig_epoch_ends]
    orig_epoch_nums = [e['epoch'] for e in orig_epoch_ends]

    # Resume run: shift epoch numbers to be continuous
    # Resume epoch 1 = overall epoch 16 (original went 1-15)
    # Determine how many original epochs we have
    n_orig = len(orig_epoch_nums)  # should be up to ~3-4 from the full_epochs file
    # We know from context: original run went to epoch 15, then resumed
    # The full_epochs.txt only has first ~4 epochs of step data
    # But we have epoch summaries from the original log
    # Let's use the epoch summaries which cover more epochs

    # From the original log, epoch summaries go: epoch 1,2,3 (that's what's in full_epochs.txt)
    # And the known validation milestones tell us:
    #   epoch 13: val_loss=0.1153
    #   epoch 14: val_loss=0.1035
    #   epoch 15: val_loss=0.1001
    # But we don't have per-epoch train loss for epochs 4-15 from the original log file.
    # So we'll combine: original step data (epochs 1-3) + resume step data (labeled as epochs 16-35)

    # For the resume data, the epoch numbers in the log are 1-20.
    # Re-number them as 16-35 (resume from epoch 15 of original)
    # But actually: the original file has step data up to epoch 4 (partial).
    # Let's just use what we have:
    #   - Original: epochs 1, 2, 3 from step data
    #   - Gap: epochs 4-15 (no step data available, but we have known val points)
    #   - Resume: epochs 16-35 (from raw_retro_v2_log.txt, local epochs 1-20)

    # Actually let's re-examine: the original log has step data going all the way through.
    # The file retro_v2_full_epochs.txt is 204 lines and shows epochs 1-3 step data.
    # So we have train loss for epochs 1-3 from original, epochs 1-20 from resume.

    # Build combined arrays:
    # Epochs 1-3 from original, then epochs 16-35 from resume (relabeled)
    all_epochs = []
    all_train_loss = []
    all_tok_acc = []

    for e in orig_epoch_ends:
        all_epochs.append(e['epoch'])
        all_train_loss.append(e['avg'])
        all_tok_acc.append(e['tok_acc'] * 100)

    # The last original epoch in our data
    last_orig_epoch = max(orig_epoch_nums) if orig_epoch_nums else 0

    # For resume, shift: resume epoch N -> overall epoch (15 + N)
    # because the resume started from epoch 15 checkpoint
    for e in res_epoch_ends:
        overall_epoch = 15 + e['epoch']
        all_epochs.append(overall_epoch)
        all_train_loss.append(e['avg'])
        all_tok_acc.append(e['tok_acc'] * 100)

    # Known validation points (from both runs combined)
    # Original run (val points from context):
    #   epoch 13: val_loss=0.1153
    #   epoch 14: val_loss=0.1035
    #   epoch 15: val_loss=0.1001, exact=54.77%, tok_acc=98.24%
    # Resume run (from parsed log):
    #   resume epoch 5 (=overall 20): val_loss=0.1359, exact=52.65%
    #   resume epoch 10 (=overall 25): val_loss=0.1451, exact=52.22%
    #   resume epoch 15 (=overall 30): val_loss=0.1529, exact=52.55%
    #   resume epoch 20 (=overall 35): val_loss=0.1604, exact=52.17%

    val_epochs = [13, 14, 15, 20, 25, 30, 35]
    val_losses = [0.1153, 0.1035, 0.1001, 0.1359, 0.1451, 0.1529, 0.1604]
    val_tok_accs = [None, None, 98.24, 98.13, 97.94, 97.83, 97.66]
    val_exact = [None, None, 54.77, 52.65, 52.22, 52.55, 52.17]

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Training Loss
    ax1.plot(all_epochs, all_train_loss, color=PRIMARY, lw=1.5, label='Train loss')
    # Add val loss points
    ax1.plot(val_epochs, val_losses, color=SECONDARY, lw=1.5, ls='--',
             marker='o', ms=4, label='Val loss', zorder=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('(a) Loss', fontweight='bold', color=PRIMARY)
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True)

    # Panel B: Token Accuracy
    ax2.plot(all_epochs, all_tok_acc, color=PRIMARY, lw=1.5, label='Train tok acc')
    # Val tok acc (only where we have values)
    vta_epochs = [e for e, v in zip(val_epochs, val_tok_accs) if v is not None]
    vta_values = [v for v in val_tok_accs if v is not None]
    ax2.plot(vta_epochs, vta_values, color=SECONDARY, lw=1.5, ls='--',
             marker='o', ms=4, label='Val tok acc', zorder=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Token Accuracy (%)')
    ax2.set_title('(b) Token Accuracy', fontweight='bold', color=PRIMARY)
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(True)
    ax2.set_ylim(70, 100.5)

    fig.suptitle('RetroTransformer v2 Training Curves', fontsize=13,
                 fontweight='bold', color=PRIMARY, y=1.02)
    fig.tight_layout()
    save(fig, 'fig3_retro_training')


# ════════════════════════════════════════════════════════════════
# Figure 4: LLM v6 Training Loss (REAL DATA)
# ════════════════════════════════════════════════════════════════
def fig4_llm_training():
    entries = parse_llm_v6_log(LLM_V6_LOG)

    if not entries:
        print('  WARNING: No LLM v6 log data found. Skipping fig4.')
        return

    epochs = np.array([e['epoch'] for e in entries])
    losses = np.array([e['loss'] for e in entries])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(epochs, losses, color=PRIMARY, lw=1.0, alpha=0.85)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss (log scale)')
    ax.set_title('RSGPT v6 Fine-tuning Loss', fontweight='bold', color=PRIMARY)
    ax.grid(True, which='both')
    ax.set_xlim(0, 30)

    fig.tight_layout()
    save(fig, 'fig4_llm_training')


# ════════════════════════════════════════════════════════════════
# Figure 5: Top-K Accuracy Comparison
# ════════════════════════════════════════════════════════════════
def fig5_topk_comparison():
    k_values = ['Top-1', 'Top-3', 'Top-5', 'Top-10']
    retro_v2 = [56.66, 73.19, 76.49, 78.44]
    llm_v6   = [80.90, 84.97, 86.20, 86.64]

    x = np.arange(len(k_values))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width / 2, retro_v2, width, label='RetroTransformer v2',
                   color=SECONDARY, edgecolor=PRIMARY, linewidth=0.8)
    bars2 = ax.bar(x + width / 2, llm_v6, width, label='RSGPT v6',
                   color=PRIMARY, edgecolor=DARK, linewidth=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8, color=DARK)

    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.set_ylim(0, 98)
    ax.legend(frameon=False, loc='upper left')
    ax.grid(axis='y')
    ax.set_title('Top-K Exact Match Accuracy on USPTO-50K', fontweight='bold', color=PRIMARY)

    fig.tight_layout()
    save(fig, 'fig5_topk_comparison')


# ════════════════════════════════════════════════════════════════
# Figure 6: Token Accuracy vs Exact Match (mathematical)
# ════════════════════════════════════════════════════════════════
def fig6_token_vs_exact():
    tok_acc = np.linspace(0.80, 1.0, 200)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Use only blue shades and gray
    lengths = [10, 20, 30, 40, 50]
    line_styles = ['-', '--', '-.', ':', '-']
    # Gradient from light to dark blue, plus gray for longest
    line_colors = ['#42A5F5', '#1E88E5', '#1565C0', '#0D47A1', GRAY]
    line_widths = [1.5, 1.5, 1.5, 1.5, 1.2]

    for L, ls, c, lw in zip(lengths, line_styles, line_colors, line_widths):
        exact = tok_acc ** L * 100
        ax.plot(tok_acc * 100, exact, lw=lw, ls=ls, color=c, label=f'L = {L}')

    ax.set_xlabel('Token Accuracy (%)')
    ax.set_ylabel('Expected Exact Match (%)')
    ax.set_title(r'$P_{\mathrm{exact}} = p_{\mathrm{tok}}^L$',
                 fontweight='bold', color=PRIMARY)
    ax.legend(frameon=False, fontsize=9, title='Sequence length')
    ax.grid(True)
    ax.set_xlim(80, 100)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    save(fig, 'fig6_token_vs_exact')


# ════════════════════════════════════════════════════════════════
# Figure 7: Ablation Study (PLACEHOLDER)
# ════════════════════════════════════════════════════════════════
def fig7_ablation():
    components = [
        'Char tokenizer\n(v1 baseline)',
        '+ Regex tokenizer',
        '+ Copy mechanism',
        '+ Offline augment (5x)',
        '+ Reaction class tokens',
        '+ Segment embeddings',
        'Full v2 system',
    ]
    # Placeholder values - labeled as such
    top1 = [0.9, 12.3, 28.5, 42.1, 48.3, 52.1, 56.7]

    colors_ab = [GRAY] + [SECONDARY] * 5 + [PRIMARY]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(range(len(components)), top1, color=colors_ab,
                   edgecolor=DARK, linewidth=0.6, height=0.55)

    for bar, val in zip(bars, top1):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', ha='left', va='center', fontsize=9, color=DARK)

    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(components, fontsize=9)
    ax.set_xlabel('Top-1 Exact Match (%)')
    ax.set_title('Ablation Study (placeholder)', fontweight='bold', color=PRIMARY)
    ax.set_xlim(0, 68)
    ax.grid(axis='x')
    ax.invert_yaxis()

    fig.tight_layout()
    save(fig, 'fig7_ablation')


# ════════════════════════════════════════════════════════════════
# Figure 8: Preprocessing Pipeline & Class Distribution
# ════════════════════════════════════════════════════════════════
def fig8_preprocessing():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Funnel
    ax = axes[0]
    stages = ['Raw Reactions', 'Valid SMILES', 'Edit Extracted', 'Final Dataset']
    counts = [50016, 48200, 37007, 37007]

    # Gradient from light to dark blue
    bar_colors = ['#90CAF9', SECONDARY, '#1E88E5', PRIMARY]

    bars = ax.barh(range(len(stages)), counts, color=bar_colors,
                   edgecolor=DARK, linewidth=0.6, height=0.5)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + 300, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', ha='left', va='center', fontsize=9, color=DARK)

    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages, fontsize=9)
    ax.set_xlabel('Number of Reactions')
    ax.set_title('(a) Preprocessing Pipeline', fontweight='bold', color=PRIMARY)
    ax.invert_yaxis()
    ax.set_xlim(0, 58000)
    ax.grid(axis='x')

    # Panel B: Class distribution
    ax = axes[1]
    classes = [f'Class {i}' for i in range(1, 11)]
    dist = [10125, 8564, 5234, 4182, 3012, 2456, 1823, 1102, 356, 153]
    blues = plt.cm.Blues(np.linspace(0.25, 0.85, 10))

    wedges, texts, autotexts = ax.pie(
        dist, labels=None, autopct='%1.1f%%',
        colors=blues, pctdistance=0.82,
        wedgeprops=dict(linewidth=0.3, edgecolor='white')
    )
    for t in autotexts:
        t.set_fontsize(7)
        t.set_color(DARK)
    ax.legend(classes, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8,
              frameon=False)
    ax.set_title('(b) Reaction Class Distribution', fontweight='bold', color=PRIMARY)

    fig.tight_layout()
    save(fig, 'fig8_preprocessing')


# ════════════════════════════════════════════════════════════════
# Figure 9: Quality Metrics (Validity, Tanimoto, Diversity)
# ════════════════════════════════════════════════════════════════
def fig9_quality_metrics():
    metrics = ['SMILES\nValidity', 'Avg\nTanimoto', 'Beam\nDiversity']
    retro_vals = [75.0, 86.23, 100.0]
    llm_vals   = [95.0, 91.0,  51.0]

    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars1 = ax.bar(x - width / 2, retro_vals, width, label='RetroTx v2',
                   color=SECONDARY, edgecolor=PRIMARY, linewidth=0.8)
    bars2 = ax.bar(x + width / 2, llm_vals, width, label='RSGPT v6',
                   color=PRIMARY, edgecolor=DARK, linewidth=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f'{h:.0f}%', ha='center', va='bottom', fontsize=9, color=DARK)

    ax.set_ylabel('Score (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 115)
    ax.legend(frameon=False)
    ax.grid(axis='y')
    ax.set_title('Quality Metrics', fontweight='bold', color=PRIMARY)

    fig.tight_layout()
    save(fig, 'fig9_quality_metrics')


# ════════════════════════════════════════════════════════════════
# Figure 10: Projected Improvement Roadmap
# ════════════════════════════════════════════════════════════════
def fig10_roadmap():
    stages = [
        'Current\nRSGPT v6',
        '+ TTA\n(20x)',
        '+ Forward\nRe-ranking',
        '+ Ensemble\n(RetroTx+LLM)',
        '+ R-SMILES\n+ 20x Aug',
        '+ RL\nFine-tuning',
    ]
    low  = [80.9, 88, 90, 91, 93, 94]
    high = [80.9, 93, 94, 95, 96, 97]
    mid  = [(l + h) / 2 for l, h in zip(low, high)]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(stages))

    ax.fill_between(x, low, high, alpha=0.15, color=PRIMARY)
    ax.plot(x, mid, color=PRIMARY, lw=2, marker='o', ms=6, zorder=5, label='Expected')
    ax.plot(x, low, color=SECONDARY, lw=0.8, ls='--', alpha=0.6)
    ax.plot(x, high, color=SECONDARY, lw=0.8, ls='--', alpha=0.6)

    for i in range(len(stages)):
        ax.text(i, high[i] + 0.6, f'{mid[i]:.0f}%', ha='center',
                fontsize=8, color=DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=8.5)
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_ylim(75, 100)
    ax.legend(frameon=False, loc='lower right')
    ax.grid(axis='y')
    ax.set_title('Projected Improvement Roadmap', fontweight='bold', color=PRIMARY)

    fig.tight_layout()
    save(fig, 'fig10_roadmap')


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating Rasyn paper figures...')
    print(f'  Font: {FONT_FAMILY}')
    print(f'  Output: {FIGURES_DIR}/')
    print()

    fig1_architecture()
    fig2_sota_comparison()
    fig3_retro_training()
    fig4_llm_training()
    fig5_topk_comparison()
    fig6_token_vs_exact()
    fig7_ablation()
    fig8_preprocessing()
    fig9_quality_metrics()
    fig10_roadmap()

    print(f'\nAll 10 figures saved to {FIGURES_DIR}/')
