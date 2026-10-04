# Reproduction entry; reads the frozen inputs through qvdfe.paths and writes to reproduce/output.
"""Editable Figure 12: full-function and power-approximation route assignment."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent / 'lib'))  # reproduce/lib
from _paths import ROOT as _ROOT, FROZEN as _FROZEN, OUT as _OUT  # noqa: E402
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figure_style import style, panel, save_figure, QA, BLUE, ORANGE, TEAL, PURPLE, INK, MUTED, GRAY, LIGHT, WHITE

HERE = Path(__file__).resolve().parent
ROOT = _ROOT
DEFAULT_DATA = _FROZEN / 'calibration/pigou'
OUT = _OUT / 'figs'
POLS = ['CO2', 'NOx', 'CO', 'HC']
CASES = [
    ('time_UE', 'Time UE', PURPLE, 'o'),
    ('time_SO', 'Time SO', ORANGE, 's'),
    ('generalized_SO_eta1', r'Generalized SO ($\eta=1$)', TEAL, 'D'),
    ('generalized_UE_eta1', r'Generalized UE ($\eta=1$)', BLUE, '^'),
]


def network(ax, context):
    panel(ax, 'a', 'Two-route example')
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    # Arrows are layout objects; no geographic network is implied.
    for ys, ye, color in [(.70, .70, BLUE), (.25, .25, GRAY)]:
        ax.plot([.07, .23, .76, .94], [.48, ys, ye, .48], color=color, lw=1.4)
        ax.add_patch(FancyArrowPatch((.55, ys), (.75, ye), arrowstyle='-|>',
                                    mutation_scale=10, color=color, lw=1.4))
    ax.scatter([.07, .94], [.48, .48], s=25, facecolor=WHITE, edgecolor=INK, zorder=4)
    ax.text(.035, .48, 'O', ha='right', va='center', fontsize=8.5)
    ax.text(.975, .48, 'D', ha='left', va='center', fontsize=8.5)
    ax.text(.47, .84, r'Route 1: $B+f$', ha='center', fontsize=8.7, color=BLUE)
    ax.text(.47, .59, r'QVDFE time and CO$_2$', ha='center', fontsize=8.2)
    ax.text(.48, .35, r'Route 2: $N-f$', ha='center', fontsize=8.7, color=MUTED)
    ax.text(.48, .15, r'Constant time and CO$_2$', ha='center', fontsize=8.2)
    ax.add_patch(FancyArrowPatch((.23, .97), (.23, .72), arrowstyle='-|>',
                                mutation_scale=10, color=INK, lw=.9))
    ax.text(.25, .95, r'Background $B$', va='center', fontsize=8)
    ax.text(.48, .025,
            rf'$B={context["background_veh_lane"]:.1f}$; $N={context["flexible_veh_lane"]:.1f}$ veh/lane',
            ha='center', va='center', fontsize=8)


def build(data):
    style()
    context = json.loads((data / 'pigou_selection.json').read_text())
    results = {}
    for model in ['full', 'surrogate']:
        folder = data / model
        results[model] = {'curves':pd.read_csv(folder/'network_curves.csv'),
                          'outcomes':pd.read_csv(folder/'assignment_results.csv').set_index('assignment'),
                          'sweep':pd.read_csv(folder/'weight_sensitivity.csv')}
    fig, axs = plt.subplots(2, 2, figsize=(7.07, 6.35))
    fig.subplots_adjust(left=.095, right=.98, top=.94, bottom=.205, wspace=.34, hspace=.50)
    network(axs[0, 0], context)

    ax = axs[0, 1]
    panel(ax, 'b', 'Time-emission tradeoff')
    for model, line_style in [('full','-'),('surrogate',(0,(4,2)))]:
        curves = results[model]['curves']
        ax.plot(curves.network_time, curves.network_CO2/1000, color=GRAY,
                lw=1.6, ls=line_style, zorder=1)
        for key, _, color, marker in CASES:
            row = results[model]['outcomes'].loc[key]
            ax.scatter(row.travel_time_veh_h, row.CO2_g/1000, marker=marker,
                       s=30 if model=='full' else 48,
                       facecolors=color if model=='full' else 'none',
                       edgecolors=WHITE if model=='full' else color,
                       linewidths=.45 if model=='full' else .8, zorder=4)
    ax.set(xlabel='Total travel time (veh h)', ylabel=r'Total CO$_2$ (kg)')
    ax.margins(.08)
    ax.locator_params(axis='both',nbins=4)
    ax.grid(False)

    ax = axs[1, 0]
    panel(ax, 'c', 'Response to the CO$_2$ weight')
    for model,line_style in [('full','-'),('surrogate',(0,(4,2)))]:
        sweep=results[model]['sweep']
        ax.plot(sweep.eta,sweep.UE_share,color=INK,lw=1.6,ls=line_style)
        ax.plot(sweep.eta,sweep.SO_share,color=GRAY,lw=1.6,ls=line_style)
    ax.axvline(1, color=LIGHT, lw=.8, zorder=0)
    for model in ['full','surrogate']:
        for key, _, color, marker in CASES:
            row = results[model]['outcomes'].loc[key]
            eta = 0 if key.startswith('time_') else 1
            ax.scatter(eta,row.flexible_route1_share,marker=marker,s=29 if model=='full' else 47,
                       facecolors=color if model=='full' else 'none',
                       edgecolors=WHITE if model=='full' else color,
                       linewidths=.45 if model=='full' else .8,zorder=4,clip_on=False)
    max_eta=max(float(v['sweep'].eta.max()) for v in results.values())
    ax.set(xlabel=r'Normalized CO$_2$ weight $\eta$', ylabel=r'Flexible route-1 share $f/N$',
           xlim=(-.05,max_eta+.05),ylim=(0,1.08),yticks=[0,.25,.5,.75,1])
    ax.legend(handles=[Line2D([],[],color=INK,lw=1.4,label='UE'),
                       Line2D([],[],color=GRAY,lw=1.4,label='SO')],loc='lower right',frameon=False,fontsize=7.5,ncol=2)
    ax.grid(False)

    ax = axs[1, 1]
    panel(ax, 'd', 'Pollutant changes from time UE')
    x = np.arange(len(POLS))
    bar_cases = [CASES[1], CASES[2], CASES[3]]
    changes = {}
    offsets=np.arange(6)*.13-.325
    for j,(key, _, color, _) in enumerate(bar_cases):
        for k,model in enumerate(['full','surrogate']):
            outcomes=results[model]['outcomes']
            vals=100*(outcomes.loc[key,[p+'_g' for p in POLS]].to_numpy(float)/
                      outcomes.loc['time_UE',[p+'_g' for p in POLS]].to_numpy(float)-1)
            ax.bar(x+offsets[2*j+k],vals,width=.12,
                   facecolor=color if model=='full' else WHITE,
                   edgecolor=WHITE if model=='full' else color,
                   linewidth=.35 if model=='full' else .7,hatch=None if model=='full' else '///',zorder=3)
            changes[f'{model}:{key}']=dict(zip(POLS,vals.tolist()))
    ax.axhline(0, color=INK, lw=.7)
    ax.set(xticks=x, xticklabels=[r'CO$_2$', r'NO$_x$', 'CO', 'HC'],
           ylabel='Change in total emissions (%)')
    ax.margins(y=.15)
    ax.grid(False)
    # Small positive HC changes retain their true heights and get direct labels.
    # Do not inflate a bar to an arbitrary minimum height.
    for model,position,y in [('full',offsets[4],-10),('surrogate',offsets[5],-22)]:
        value=changes[f'{model}:generalized_UE_eta1']['HC']
        ax.annotate(f'{value:+.3f}%',xy=(3+position,value),xytext=(2.0,y),
            textcoords='data',ha='left',va='center',fontsize=7.5,color=BLUE,
            arrowprops={'arrowstyle':'-','color':BLUE,'lw':.65,'shrinkA':2,'shrinkB':2,
                        'connectionstyle':'arc3'})

    handles = [Line2D([], [], ls='', marker=mark, markersize=5.5, color=color, label=label)
               for _, label, color, mark in CASES]
    handles.extend([Line2D([], [], color=MUTED, lw=1.4, label='Full function'),
                    Line2D([], [], color=MUTED, lw=1.4, ls=(0, (4, 2)), label='Power approximation')])
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53,.015),
               ncol=2, columnspacing=1.8, handlelength=1.6, fontsize=7.5, labelspacing=.55)
    paths = save_figure(fig,'fig12_pigou_network',OUT)
    audit = {
        'status': 'PASS', 'cases': [row[0] for row in CASES],
        'models':['full','surrogate'], 'assumed_coincident_optima':False,
        'bar_change_pct_relative_to_time_UE': changes,
        'curves_rows':{m:len(v['curves']) for m,v in results.items()},
        'weight_sweep_rows':{m:len(v['sweep']) for m,v in results.items()},
        'results_recomputed': False, 'files': paths,
        'input_sha256':{f'{model}/{name}':hashlib.sha256((data/model/name).read_bytes()).hexdigest()
                         for model in results for name in ['network_curves.csv','assignment_results.csv','weight_sensitivity.csv']},
        'caption':'Illustrative two-route assignment comparing the full function with its power approximation. '
                  '(a) Fixed background B uses route 1 and flexible demand N chooses between routes. '
                  '(b) Aggregate travel time and CO2 across feasible splits. '
                  '(c) UE and SO shares as the CO2 weight varies. '
                  '(d) Pollutant changes for time SO, generalized SO and generalized UE at eta=1, each relative '
                  'to its own model time UE. Filled markers/bars denote the full function; hollow markers and '
                  'hatched bars denote the approximation. No coincident optima are assumed.'
    }
    (QA/'fig12_pigou_network_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({'files': paths, 'changes': changes}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    build(parser.parse_args().data)
