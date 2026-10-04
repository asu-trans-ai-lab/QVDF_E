# Reproduction entry; reads the frozen inputs through qvdfe.paths and writes to reproduce/output.
"""Editable, exactly constructed mechanics figures for QVDFE.

Run this file to regenerate Figure 3 as vector PDF/SVG and 600-dpi PNG.
All coordinates come from the equations below. No images, AI-generated geometry,
or empirical fits are used. These are illustrative constructions, not estimates.
The Newell point queue is distinct from the piecewise-constant spatial queue.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent / 'lib'))  # reproduce/lib
from _paths import ROOT as _ROOT, FROZEN as _FROZEN, OUT as _OUT  # noqa: E402
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from figure_style import (
    style, panel, save_figure, BLUE, ORANGE, TEAL, PURPLE, INK, GRAY,
    MUTED, LIGHT, WHITE,
)
DEFAULT_OUTPUT = _OUT / 'figs'


@dataclass(frozen=True)
class SpatialQueue:
    vf: float = 65.0      # mi/h
    capacity: float = 1800.0  # veh/h/lane
    kj: float = 220.0     # veh/mi/lane
    mu: float = 1350.0    # veh/h/lane
    qa: float = 1700.0    # upstream high-flow state
    qr: float = 900.0     # upstream low-flow recovery state
    length: float = 2.0   # mi
    tb: float = .15      # hour: time the low-flow characteristic reaches tail

    @property
    def kc(self): return self.capacity / self.vf
    @property
    def omega(self): return self.capacity / (self.kj-self.kc)
    @property
    def kq(self): return self.kj-self.mu/self.omega
    @property
    def vq(self): return self.mu/self.kq
    @property
    def ka(self): return self.qa/self.vf
    @property
    def kr(self): return self.qr/self.vf
    @property
    def sb(self): return (self.mu-self.qa)/(self.kq-self.ka)
    @property
    def sr(self): return (self.mu-self.qr)/(self.kq-self.kr)
    @property
    def xb(self): return self.sb*self.tb
    @property
    def t3(self): return self.tb-self.xb/self.sr

    def tail(self, t):
        t = np.asarray(t)
        return np.where(t <= self.tb, self.sb*t, self.xb+self.sr*(t-self.tb))

    def trajectory(self, entry):
        """Intersect vf trajectory with the exact moving tail, then move at vq."""
        free_exit = entry+self.length/self.vf
        if free_exit <= 0 or free_exit >= self.t3:
            return np.array([entry, free_exit]), np.array([-self.length, 0.]), None
        hit = (self.length+self.vf*entry)/(self.vf-self.sb)
        if hit > self.tb:
            hit = (self.length+self.vf*entry+self.xb-self.sr*self.tb)/(self.vf-self.sr)
        location = float(self.tail(hit))
        actual_exit = hit-location/self.vq
        return np.array([entry, hit, actual_exit]), np.array([-self.length, location, 0.]), {
            "entry_h": float(entry), "hit_h": float(hit), "hit_mi": location,
            "free_exit_h": float(free_exit), "actual_exit_h": float(actual_exit),
            "delay_h": float(actual_exit-free_exit),
        }


def make_shockwave(output):
    s = SpatialQueue()
    fig, (fd, tx) = plt.subplots(1, 2, figsize=(7.07, 3.34),
                                 gridspec_kw={"width_ratios": [1, 1.42]})
    fig.subplots_adjust(left=.075, right=.99, top=.85, bottom=.28, wspace=.29)
    panel(fd, "a", "Flow, density and wave speed")
    fd.plot([0,s.kc,s.kj], [0,s.capacity,0], color=INK, lw=1.5)
    fd.plot([s.ka,s.kq],[s.qa,s.mu], color=BLUE, lw=1.6)
    fd.plot([s.kr,s.kq],[s.qr,s.mu], color=ORANGE, lw=1.6)
    fd.plot([0,s.kq],[0,s.mu], color=PURPLE, lw=1.1)
    for k,q,c in [(s.ka,s.qa,BLUE),(s.kr,s.qr,ORANGE),(s.kq,s.mu,INK)]:
        fd.scatter(k,q,s=22,c=c,edgecolors="white",linewidths=.55,zorder=5)
    fd.scatter(s.kc,s.capacity,s=12,c=INK,zorder=5)
    fd.annotate(r"$H$", (s.ka,s.qa), xytext=(58,1740), fontsize=10,
                arrowprops={"arrowstyle":"-", "color":GRAY,"lw":.65}, color=BLUE)
    fd.text(s.kr+13,s.qr-170,r"$L$",fontsize=10,color=ORANGE)
    fd.text(s.kq+8,s.mu+45,r"$(k_q,\mu)$",fontsize=9)
    fd.text(s.kc+5,s.capacity+165,r"$C$",fontsize=9)
    fd.text(62,285,r"$v_q=\mu/k_q=17.8$ mph",color=PURPLE,fontsize=8)
    fd.text(.97,.93,r"$c_b=-7.1$ mph"+"\n"+r"$c_r=+7.3$ mph",
            transform=fd.transAxes,ha="right",va="top",fontsize=8,linespacing=1.6)
    fd.set(xlim=(0,235),ylim=(0,2150),xlabel=r"Density $k$ (veh/mi/lane)",
           ylabel=r"Flow $q$ (veh/h/lane)")
    fd.set_xticks([0,50,100,150,220]); fd.set_yticks([0,600,1200,1800])
    panel(tx,"b","Spatial queue and vehicle traversal")
    polygon = Polygon([[0,0],[s.tb*60,s.xb],[s.t3*60,0]],facecolor=ORANGE,alpha=.08,edgecolor="none")
    tx.add_patch(polygon)
    checks=[]
    # A single upstream flow change reaches the moving tail at t_b.
    # Equal-size representative cohorts make their spacing consistent with H/L.
    switch_entry=s.tb-(s.xb+s.length)/s.vf
    first_entry=-.033; final_entry=s.t3-.009
    high_count=(switch_entry-first_entry)*s.qa
    whole_count=high_count+(final_entry-switch_entry)*s.qr
    cohorts=np.arange(0,whole_count,30.)
    entry_times=np.where(cohorts<=high_count,first_entry+cohorts/s.qa,
                         switch_entry+(cohorts-high_count)/s.qr)
    for te in entry_times:
        tt,xx,check=s.trajectory(float(te))
        tx.plot(tt*60,xx,color=GRAY,lw=.65,alpha=.53,zorder=1)
        if check: checks.append(check)
    tt,xx,selected=s.trajectory(.13)
    tx.plot(tt*60,xx,color=BLUE,lw=2.05,zorder=5)
    tx.scatter(tt[1:]*60,xx[1:],c=BLUE,s=13,zorder=6)
    tx.plot([0,s.tb*60],[0,s.xb],color=BLUE,lw=1.6,zorder=4)
    tx.plot([s.tb*60,s.t3*60],[s.xb,0],color=ORANGE,lw=1.6,zorder=4)
    tx.plot([selected["entry_h"]*60,selected["free_exit_h"]*60],[-s.length,0],
            color=INK,lw=1,ls=(0,(2,2)),zorder=3)
    tx.axhline(0,color=INK,lw=1)
    tx.annotate("",xy=(selected["free_exit_h"]*60,.11),
                xytext=(selected["actual_exit_h"]*60,.11),
                arrowprops={"arrowstyle":"<->","lw":.8,"color":INK,"shrinkA":0,"shrinkB":0})
    tx.text((selected["actual_exit_h"]+selected["free_exit_h"])*30,.14,
            r"$w$",ha="center",va="bottom",fontsize=10)
    tx.text(19.9,.02,"Bottleneck",ha="right",va="bottom",fontsize=8)
    tx.annotate(r"Buildup $c_b$",xy=(2.9,float(s.tail(2.9/60))),xytext=(.1,-.83),
                color=BLUE,fontsize=8,bbox={"boxstyle":"square,pad=0.12","fc":"white","ec":"none"},arrowprops={"arrowstyle":"-","lw":.6,"color":BLUE})
    tx.annotate(r"Recovery $c_r$",xy=(14.5,float(s.tail(14.5/60))),xytext=(14.2,-1.03),
                color=ORANGE,fontsize=8,bbox={"boxstyle":"square,pad=0.12","fc":"white","ec":"none"},arrowprops={"arrowstyle":"-","lw":.6,"color":ORANGE})
    tx.text(4.55,-1.66,r"$v_f$",color=BLUE,fontsize=10,bbox={"boxstyle":"square,pad=0.12","fc":"white","ec":"none"})
    tx.text(10.95,-.70,r"$v_q$",color=BLUE,fontsize=10,bbox={"boxstyle":"square,pad=0.12","fc":"white","ec":"none"})
    tx.set(xlim=(-2.2,20.3),ylim=(-2.05,.37),
           xlabel=r"Time from onset $t-t_0$ (min)",ylabel="Distance from bottleneck (mi)")
    tx.set_xticks([0,s.tb*60,s.t3*60]);tx.set_xticklabels([r"$t_0=0$",r"$t_b=9.0$",r"$t_3=17.7$"])
    tx.set_yticks([-2,-1,0])
    fig.legend(handles=[Line2D([0],[0],color=BLUE,lw=2,label="Vehicle trajectory"),
                        Line2D([0],[0],color=INK,lw=1,ls=(0,(2,2)),label="Same vehicle at free-flow speed")],
               loc="lower center",bbox_to_anchor=(.71,-.003),ncol=1,labelspacing=.3,handlelength=2.3)
    trajectory_residuals=[]
    for tr in checks:
        free_duration=tr["hit_h"]-tr["entry_h"]
        queue_duration=tr["actual_exit_h"]-tr["hit_h"]
        trajectory_residuals.append(abs(s.vf*free_duration+s.vq*queue_duration-s.length))
        assert free_duration>=0 and queue_duration>=0 and tr["delay_h"]>=-1e-12
        test_t=np.linspace(tr["hit_h"],tr["actual_exit_h"],100)
        vehicle_x=tr["hit_mi"]+s.vq*(test_t-tr["hit_h"])
        assert np.all(vehicle_x>=s.tail(test_t)-1e-10), "Vehicle exits through tail incorrectly"
    assert s.sb<0<s.sr<s.vq<s.vf and -s.xb<s.length
    report={"parameters":asdict(s),"kc":s.kc,"omega":s.omega,"kq":s.kq,"vq":s.vq,
            "buildup_wave_mph":s.sb,"recovery_wave_mph":s.sr,"t3_h":s.t3,
            "max_queue_extent_mi":-s.xb,"selected_vehicle":selected,
            "upstream_low_flow_switch_entry_h":switch_entry,
            "representative_vehicle_cohort_veh_lane":30.,
            "max_trajectory_distance_residual_mi":max(trajectory_residuals),
            "recovery_state":"upstream low-flow L versus queued Q; not downstream discharge",
            "spatial_maximum_time":"t_b; not identified with the point-queue t_2",
            "files":save_figure(fig,"fig2_shockwave_bottleneck",output)}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild Fig. 3 (shock waves at the bottleneck) on the triangular FD.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    style()
    report = make_shockwave(args.output)
    audit = _OUT / 'qa' / 'figure3_construction.json'
    audit.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
