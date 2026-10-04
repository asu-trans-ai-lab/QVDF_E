"""Figure export and layout checks; polish_style.py holds the style."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.text import Text
import polish_style as P

BLUE, ORANGE, TEAL, PURPLE = P.ACCENT, P.INK, P.MID, P.DARK
INK, GRAY, MUTED, GRID, LIGHT, WHITE = P.INK, P.MID, P.MID, P.LIGHT, P.LIGHT, P.WHITE
from _paths import ROOT as _ROOT, OUT as _OUT
QA = _OUT / 'qa'

def style():
    P.apply()

def panel(ax, letter, title):
    ax.set_title(f'({letter})  {title}', loc='left', pad=6)

def clean(ax):
    ax.set_facecolor(WHITE)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(False)

def identity(ax, obs, pred):
    import numpy as np
    values=np.r_[np.asarray(obs,float),np.asarray(pred,float)]
    values=values[np.isfinite(values)]
    high=max(1,float(values.max())*1.045) if len(values) else 1
    ax.plot([0,high],[0,high],color=MUTED,lw=.8,ls='--',zorder=1)
    ax.set(xlim=(0,high),ylim=(0,high))
    ax.set_aspect('equal',adjustable='box')

def save_figure(fig, stem, output):
    output, qa = Path(output), QA
    output.mkdir(parents=True,exist_ok=True); qa.mkdir(parents=True,exist_ok=True)
    # Do not tight-crop: the PDF must retain the 7.07-inch printed width.
    fig.set_size_inches(P.TEXT_WIDTH_IN, fig.get_figheight(), forward=True)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    outside_ticks=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            low,high=sorted(axis.get_view_interval())
            for tick in list(axis.get_major_ticks())+list(axis.get_minor_ticks()):
                if not low-1e-10 <= tick.get_loc() <= high+1e-10:
                    outside_ticks.update([id(tick.label1),id(tick.label2)])
    texts=[]; seen=set()
    for item in fig.findobj(match=Text):
        if not item.get_visible() or not item.get_text().strip() or id(item) in seen or id(item) in outside_ticks:
            continue
        seen.add(id(item)); box=Text.get_window_extent(item,renderer=renderer)
        if box.width<=0 or box.height<=0: continue
        texts.append({'text':item.get_text(),'font_pt':item.get_fontsize(),'bbox_px':list(box.extents)})
    hits=[]
    for i,a in enumerate(texts):
        x0,y0,x1,y1=a['bbox_px']
        for b in texts[i+1:]:
            u0,v0,u1,v1=b['bbox_px']; dx=min(x1,u1)-max(x0,u0);dy=min(y1,v1)-max(y0,v0)
            if dx>.5 and dy>.5: hits.append({'a':a['text'],'b':b['text'],'intersection_px':[dx,dy]})
    width,height=fig.bbox.width,fig.bbox.height
    clipped=[x for x in texts if x['bbox_px'][0]<0 or x['bbox_px'][1]<0 or x['bbox_px'][2]>width or x['bbox_px'][3]>height]
    paths=[]
    for ext in ('pdf','svg','png'):
        target=output/f'{stem}.{ext}'
        fig.savefig(target,dpi=600,facecolor=WHITE)
        paths.append(str(target.relative_to(_ROOT)))
    fig.savefig(qa/f'{stem}_preview.png',dpi=160,facecolor=WHITE)
    (qa/f'{stem}_layout.json').write_text(json.dumps({'figure':stem,'width_in':fig.get_figwidth(),
        'height_in':fig.get_figheight(),'text_intersection_candidates':hits,'clipped_text':clipped,
        'texts':texts,'note':'Conservative text-box screen; inspect curves, markers and legends separately.'},indent=2),encoding='utf-8')
    plt.close(fig)
    return paths
