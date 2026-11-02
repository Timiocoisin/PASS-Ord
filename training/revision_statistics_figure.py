"""Render the revised stage-aware inference figure; supersedes frame-iid CIs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def main():
    summary=json.loads((ROOT/'outputs/baselines/revision_audit/summary.json').read_text())
    rows=summary['intervals']
    table=[]
    for r in rows:
        table.append({**r,'delta':r['estimate'],'delta_ci_lo':r['ci95_lo'],
                      'delta_ci_hi':r['ci95_hi'],'resampling_unit':'whole stage',
                      'progress_level':'pass' if 'MAE' in r['contrast'] else '',
                      'bootstrap_p_one_sided':np.nan,
                      'note':'Old frame-iid p-values withdrawn; stage and block CIs are primary.'})
    tab=ROOT/'outputs/baselines/paper_tables'
    pd.DataFrame(table).to_csv(tab/'T_significance.csv',index=False)
    # Strict JSON: unavailable p-values are null, never fabricated as zero.
    for r in table:r['bootstrap_p_one_sided']=None
    (tab/'T_significance.json').write_text(json.dumps({'rows':table,
        'frame_ours_mae':summary['frame_ours_mae'],'scope':summary['scope']},indent=2),encoding='utf-8')
    canonical=ROOT/'outputs/baselines/ours_nested/paper_nested_numbers.json'
    numbers=json.loads(canonical.read_text(encoding='utf-8'))
    numbers['sig']=table
    numbers['stage_aware_reanalysis']={'frame_ours_mae':summary['frame_ours_mae'],
                                      'scope':summary['scope']}
    canonical.write_text(json.dumps(numbers,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9,2.8),gridspec_kw={'width_ratios':[1.2,1]})
    for ax,indices,title,xlabel in [
        (axes[0],[0,1],'(a) Classification differences',r'$\Delta$Acc (PASS-Ord $-$ baseline)'),
        (axes[1],[2],'(b) Matched pass-level progress',r'$\Delta$MAE (B3 $-$ PASS-Ord)')]:
        for j,i in enumerate(indices):
            r=rows[i]; y=len(indices)-j-1
            for offset,lo,hi,color,marker in [(.10,r['ci95_lo'],r['ci95_hi'],'#0072B2','o'),
                (-.10,r['block_ci95_lo'],r['block_ci95_hi'],'#D55E00','s')]:
                ax.plot([lo,hi],[y+offset]*2,color=color,lw=2)
                ax.plot(r['estimate'],y+offset,marker,color=color,ms=4)
        ax.axvline(0,color='#999999',lw=.8,ls='--')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(['Acc$_5$ vs B5','Acc$_3$ vs B3'] if len(indices)==2 else ['Pass MAE'])
        ax.set_ylim(-.6,len(indices)-.4)
        ax.set_title(title,fontweight='bold');ax.set_xlabel(xlabel)
        ax.spines['left'].set_visible(False)
    axes[0].set_xlim(-.016,.049);axes[1].set_xlim(-.0005,.0065)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],color='#0072B2',marker='o',label='Whole-stage bootstrap'),
                        Line2D([0],[0],color='#D55E00',marker='s',label='Five-test-stage block bootstrap')],
               loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.01),fontsize=8)
    fig.suptitle('Paired 95% intervals on one specimen trajectory (85 test stages)',fontsize=10)
    fig.subplots_adjust(left=.13,right=.98,bottom=.27,top=.80,wspace=.55)
    fig.savefig(ROOT/'template/figures/Fig09_significance.pdf',bbox_inches='tight')
    fig.savefig(ROOT/'outputs/baselines/revision_audit/stage_intervals_preview.png',dpi=160,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
