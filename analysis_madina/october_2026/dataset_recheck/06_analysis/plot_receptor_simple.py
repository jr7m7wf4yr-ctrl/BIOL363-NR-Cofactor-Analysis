"""Simple slide version of the receptor distribution: PDB entries per receptor, split into
coactivator (LXXLL) and corepressor (CoRNR) complexes, with the number of distinct cofactors.
Small receptors are pooled into 'Other'. Uses the same pairs as cofactor_counts.py.
Colours follow the BIOL 363 progress-update slides.
Output: figures/fig9_receptor_simple.png (slides) and fig9_receptor_simple_report.png (white, report)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cofactor_counts import HERE, load_pairs

TOP = 10
COACT, COREP, INK, MUTED, BG = '#466b2f', '#c9a227', '#263b1f', '#6d726b', '#f9f9f3'


def main():
    pairs = load_pairs()
    # one row per PDB and receptor; a PDB counts as corepressor if any of its peptides is CoRNR
    pdb = (pairs.groupby(['receptor_display', 'pdb_id']).motif
           .agg(lambda s: 'CoRNR' if (s == 'CoRNR').any() else 'LXXLL').reset_index())
    counts = pdb.groupby(['receptor_display', 'motif']).size().unstack(fill_value=0)
    counts['total'] = counts.sum(axis=1)
    named = pairs[pairs.cofactor_v2 != 'unresolved']
    counts['cofactors'] = named.groupby('receptor_display').cofactor_v2.nunique()
    counts = counts.fillna(0).sort_values('total', ascending=False)

    top, rest = counts.iloc[:TOP], counts.iloc[TOP:]
    labels = list(top.index) + [f'Other ({len(rest)} receptors)']
    co = list(top.LXXLL) + [rest.LXXLL.sum()]
    cr = list(top.CoRNR) + [rest.CoRNR.sum()]
    ncof = list(top.cofactors.astype(int)) + [None]
    labels, co, cr, ncof = labels[::-1], co[::-1], cr[::-1], ncof[::-1]

    plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans']})
    fig, ax = plt.subplots(figsize=(11.5, 6.1), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    y = range(len(labels))
    ax.barh(y, co, color=COACT, height=0.7, label='Coactivator (LXXLL)')
    ax.barh(y, cr, left=co, color=COREP, height=0.7, label='Corepressor (CoRNR)')
    for i, (a, b, n) in enumerate(zip(co, cr, ncof)):
        note = f'{a + b}' + (f'   ·  {n} cofactor{"s" if n != 1 else ""}' if n is not None else '')
        ax.text(a + b + 3, i, note, va='center', fontsize=15, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=16, color=INK)
    ax.set_xlim(0, max(a + b for a, b in zip(co, cr)) * 1.35)
    ax.set_xlabel('PDB structures', fontsize=15, color=MUTED)
    ax.tick_params(axis='x', labelsize=13, colors=MUTED)
    ax.tick_params(axis='y', length=0)
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#c9cdc2')
    ax.grid(axis='x', color='#e3e6dc', linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', fontsize=15, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / 'figures/fig9_receptor_simple.png', facecolor=BG)
    # white copy for the written report
    ax.set_facecolor('white')
    fig.savefig(HERE / 'figures/fig9_receptor_simple_report.png', facecolor='white')
    print(counts.to_string())
    print('receptors', len(counts), 'PDB', pdb.pdb_id.nunique(), 'coact PDB', counts.LXXLL.sum(), 'corep PDB', counts.CoRNR.sum())


if __name__ == '__main__':
    main()
