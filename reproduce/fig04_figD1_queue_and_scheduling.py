# Reproduction entry; reads the frozen inputs through qvdfe.paths and writes to reproduce/output.
"""Editable, exactly constructed mechanics figures for QVDFE.

Run this file to regenerate Figures 2--4 as vector PDF/SVG and 600-dpi PNG.
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

from figure_style_mechanics import (
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
    fig, (fd, tx) = plt.subplots(1, 2, figsize=(7.1, 3.34),
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
    fd.annotate(r"$A$", (s.ka,s.qa), xytext=(58,1740), fontsize=10,
                arrowprops={"arrowstyle":"-", "color":GRAY,"lw":.65}, color=BLUE)
    fd.text(s.kr+13,s.qr-170,r"$R$",fontsize=10,color=ORANGE)
    fd.text(s.kq+8,s.mu+45,r"$(k_q,\mu)$",fontsize=9)
    fd.text(s.kc+5,s.capacity+165,r"$C$",fontsize=9)
    fd.text(84,285,r"$v_q=\mu/k_q=17.8$ mph",color=PURPLE,fontsize=8)
    fd.text(.97,.93,r"$s_b=-7.1$ mph"+"\n"+r"$s_r=+7.3$ mph",
            transform=fd.transAxes,ha="right",va="top",fontsize=8,linespacing=1.6)
    fd.set(xlim=(0,235),ylim=(0,2150),xlabel=r"Density $k$ (veh/mi/lane)",
           ylabel=r"Flow $q$ (veh/h/lane)")
    fd.set_xticks([0,50,100,150,220]); fd.set_yticks([0,600,1200,1800])
    panel(tx,"b","Spatial queue and vehicle traversal")
    polygon = Polygon([[0,0],[s.tb*60,s.xb],[s.t3*60,0]],facecolor=ORANGE,alpha=.08,edgecolor="none")
    tx.add_patch(polygon)
    checks=[]
    # A single upstream flow change reaches the moving tail at t_b.
    # Equal-size representative cohorts make their spacing consistent with A/R.
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
    tx.annotate(r"Buildup $s_b$",xy=(2.9,float(s.tail(2.9/60))),xytext=(.1,-.83),
                color=BLUE,fontsize=8,arrowprops={"arrowstyle":"-","lw":.6,"color":BLUE})
    tx.annotate(r"Recovery $s_r$",xy=(14.5,float(s.tail(14.5/60))),xytext=(14.2,-1.03),
                color=ORANGE,fontsize=8,arrowprops={"arrowstyle":"-","lw":.6,"color":ORANGE})
    tx.text(4.55,-1.66,r"$v_f$",color=BLUE,fontsize=10)
    tx.text(9.8,-.76,r"$v_q$",color=BLUE,fontsize=10)
    tx.set(xlim=(-2.2,20.3),ylim=(-2.05,.37),
           xlabel=r"Time from onset $t-t_0$ (min)",ylabel="Distance from bottleneck (mi)")
    tx.set_xticks([0,s.tb*60,s.t3*60]);tx.set_xticklabels([r"$t_0=0$",r"$t_b=9.0$",r"$t_3=17.7$"])
    tx.set_yticks([-2,-1,0])
    fig.legend(handles=[Line2D([0],[0],color=BLUE,lw=2,label="Vehicle trajectory"),
                        Line2D([0],[0],color=INK,lw=1,ls=(0,(2,2)),label="Same vehicle at free-flow speed")],
               loc="lower center",bbox_to_anchor=(.71,-.003),ncol=1,labelspacing=.3,handlelength=2.3)
    fig.text(.075,.045,"Illustrative triangular FD.\nA: high inflow; R: low inflow.",fontsize=8,color=INK,linespacing=1.5)
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
            "recovery_state":"upstream low-flow R versus queued Q; not downstream discharge",
            "spatial_maximum_time":"t_b; not identified with the point-queue t_2",
            "files":save_figure(fig,"fig2_shockwave_bottleneck",output)}
    return report


@dataclass(frozen=True)
class NewellQueue:
    mu: float = 1800.0   # veh/h/lane
    P: float = 2.0       # hours
    b: float = 675.0     # veh/h^3/lane
    Tf: float = 1/65.0   # h (a separate illustrative free-flow trip)

    def Q(self,t):
        t=np.asarray(t)
        return self.b/3*t*t*(self.P-t)
    def lam(self,t):
        t=np.asarray(t)
        return self.mu+self.b/3*(2*self.P*t-3*t*t)
    def w(self,t): return self.Q(t)/self.mu
    def wprime(self,t): return self.lam(t)/self.mu-1
    def nin(self,t): return self.mu*np.asarray(t)+self.Q(t)
    def nout(self,t): return self.mu*np.asarray(t)
    def arrival(self,t): return np.asarray(t)+self.Tf+self.w(t)
    @property
    def t2(self): return 2*self.P/3
    @property
    def qmax(self): return 4*self.b*self.P**3/81
    @property
    def wmax(self): return self.qmax/self.mu
    @property
    def mean_w(self): return self.b*self.P**3/(36*self.mu)
    @property
    def D(self): return self.mu*self.P
    @property
    def W(self): return self.b*self.P**4/36


def newell_checks(n):
    # High-order Gaussian quadrature independently checks both averaging measures.
    zz,ww=np.polynomial.legendre.leggauss(64)
    t=(zz+1)*n.P/2; wt=ww*n.P/2
    D=float(wt@n.lam(t)); area=float(wt@n.Q(t))
    vehicle_delay=float(wt@(n.lam(t)*n.w(t)))
    mean_time=float(wt@n.w(t)/n.P)
    samples=np.linspace(0,n.P,2001)
    horizontal=np.abs(n.nout(samples+n.w(samples))-n.nin(samples))
    assert abs(D-n.D)<1e-8
    assert abs(area-n.W)<1e-8 and abs(vehicle_delay-n.W)<1e-8
    assert np.min(n.lam(samples))>=0
    assert abs(n.w(n.t2)-n.wmax)<1e-12
    assert abs(mean_time/n.wmax-9/16)<1e-12
    assert float(np.max(horizontal))<1e-9
    return {"parameters":asdict(n),"t2_h":n.t2,"qmax_veh_lane":n.qmax,
            "wmax_min":n.wmax*60,"mean_delay_min":n.mean_w*60,
            "theta":n.mean_w/n.wmax,"D_veh_lane":n.D,"W_veh_h_lane":n.W,
            "lambda_t0":float(n.lam(0)),"lambda_t2":float(n.lam(n.t2)),
            "lambda_t3":float(n.lam(n.P)),"min_lambda":float(np.min(n.lam(samples))),
            "demand_integral_residual_veh":D-n.D,
            "queue_area_residual_veh_h":area-n.W,
            "arrival_weighted_delay_residual_veh_h":vehicle_delay-n.W,
            "horizontal_separation_max_residual_veh":float(np.max(horizontal)),
            "cumulative_clock":"Free-flow time aligned; same cumulative vehicle cohort"}


def episode_ticks(ax,n,labels=True):
    ax.set_xlim(0,n.P*60)
    ax.set_xticks([0,n.t2*60,n.P*60])
    if labels:
        ax.set_xticklabels([r"$t_0=0$",r"$t_2=80$",r"$t_3=120$"])
    else:
        ax.tick_params(labelbottom=False)


def make_fluid_queue(output):
    n=NewellQueue(); t=np.linspace(0,n.P,1201); tm=t*60
    # Keep the full episode and its original parameters. A separate, explicitly
    # bounded detail panel makes the small cumulative separation legible.
    fig,axs=plt.subplots(2,2,figsize=(7.1,5.40))
    fig.subplots_adjust(left=.085,right=.987,top=.93,bottom=.18,wspace=.30,hspace=.49)
    a,b,c,d=axs.flat
    panel(a,"a","Arrival and service")
    a.plot(tm,n.lam(t),color=BLUE,lw=1.5)
    a.axhline(n.mu,color=ORANGE,lw=1.15)
    a.fill_between(tm,n.mu,n.lam(t),where=t<=n.t2,color=BLUE,alpha=.10)
    a.fill_between(tm,n.mu,n.lam(t),where=t>=n.t2,color=ORANGE,alpha=.10)
    a.scatter([0,n.t2*60,n.P*60],[n.mu,n.mu,n.lam(n.P)],s=14,color=BLUE,zorder=4)
    a.text(24,2190,r"$\lambda(t)$",color=BLUE,fontsize=10)
    a.text(108,1840,r"$\mu$",fontsize=10,color=ORANGE)
    a.text(33,1872,"buildup",fontsize=8,color=BLUE)
    a.text(74,900,"recovery",fontsize=8,color=ORANGE)
    a.set(ylabel="Rate (veh/h/lane)",ylim=(750,2310),yticks=[900,1350,1800,2250])
    panel(b,"b","Cumulative counts: full episode")
    b.fill_between(tm,n.nout(t),n.nin(t),color=BLUE,alpha=.10)
    b.plot(tm,n.nin(t),color=BLUE,lw=1.5,label=r"Arrivals $N_{\rm in}$")
    b.plot(tm,n.nout(t),color=ORANGE,lw=1.4,label=r"Departures $N_{\rm out}$")
    b.legend(loc="upper left",labelspacing=.3,handlelength=2.2)
    # This is a real rectangle in the original axes, not a displaced curve.
    zoom_xlim=(56.,78.);zoom_ylim=(1600.,2520.)
    b.add_patch(Rectangle((zoom_xlim[0],zoom_ylim[0]),zoom_xlim[1]-zoom_xlim[0],
                         zoom_ylim[1]-zoom_ylim[0],facecolor="none",edgecolor=MUTED,
                         lw=.85,zorder=5))
    b.annotate("Detail in (d)",xy=(zoom_xlim[1],zoom_ylim[0]),xytext=(85,760),
               fontsize=8,ha="center",va="center",
               arrowprops={"arrowstyle":"-","color":MUTED,"lw":.7})
    b.annotate(r"Shaded area $=W_P$",xy=(31,float(n.nout(31/60)+n.Q(31/60)/2)),
               xytext=(9,1720),fontsize=8,
               arrowprops={"arrowstyle":"-","color":MUTED,"lw":.7})
    b.set(ylabel="Cumulative count (veh/lane)",ylim=(0,3950),yticks=[0,1200,2400,3600])

    panel(c,"c","Queue and excess delay")
    normalized=n.w(t)/n.wmax
    c.fill_between(tm,0,normalized,color=BLUE,alpha=.08)
    c.plot(tm,normalized,color=BLUE,lw=1.5)
    c.axhline(9/16,color=ORANGE,lw=1,ls=(0,(3,2)))
    c.scatter([n.t2*60],[1],s=18,color=BLUE,zorder=4)
    c.plot([n.t2*60,n.t2*60],[0,1],lw=.65,color=GRAY,ls=(0,(1.5,2)))
    c.text(7,.62,r"$\bar w_P/w_{t_2}=9/16$",color=ORANGE,fontsize=9)
    c.set(ylabel=r"$Q/Q_{\max}=w/w_{t_2}$",ylim=(0,1.2),yticks=[0,.5,1])

    panel(d,"d","Same-cohort queue and delay: detail")
    d.fill_between(tm,n.nout(t),n.nin(t),color=BLUE,alpha=.10)
    d.plot(tm,n.nin(t),color=BLUE,lw=1.6)
    d.plot(tm,n.nout(t),color=ORANGE,lw=1.5)
    # Both arrows use the same entering cohort at t=63 min. Departure time is
    # solved from N_out(t+w)=N_in(t), not approximated by a shifted sketch.
    ti=1.05; ni=float(n.nin(ti)); no=float(n.nout(ti)); wi=float(n.w(ti))
    d.annotate("",xy=(ti*60,ni),xytext=(ti*60,no),
               arrowprops={"arrowstyle":"<->","lw":1,"color":INK,"shrinkA":0,"shrinkB":0,"mutation_scale":9})
    d.annotate("",xy=((ti+wi)*60,ni),xytext=(ti*60,ni),
               arrowprops={"arrowstyle":"<->","lw":1,"color":INK,"shrinkA":0,"shrinkB":0,"mutation_scale":9})
    d.scatter([ti*60,ti*60,(ti+wi)*60],[ni,no,ni],color=INK,s=13,zorder=6)
    d.text(ti*60-.7,(ni+no)/2,r"$Q(t)$",fontsize=10,ha="right",va="center")
    d.text((ti+wi/2)*60,ni+30,r"$w(t)$",fontsize=10,ha="center",va="bottom")
    d.set(xlim=zoom_xlim,ylim=zoom_ylim,ylabel="Cumulative count (veh/lane)",
          xlabel="Time from onset (min)",xticks=[56,63,71,78],yticks=[1600,1900,2200,2500])
    for ax in [a,b,c]:
        episode_ticks(ax,n)
        ax.set_xlabel(r"Time from onset (min)")
    fig.text(.085,.026,r"Same Newell episode in all panels: $P=2$ h, $\mu=1800$ veh/h/lane; free-flow travel times aligned."+"\n"
             +r"Detail (d): arrival at 63.0 min, departure at 70.86 min; $Q=235.66$ veh/lane and $w=7.86$ min.",
             fontsize=8,linespacing=1.6)
    report=newell_checks(n)
    assert zoom_xlim[0]<ti*60<(ti+wi)*60<zoom_xlim[1]
    assert zoom_ylim[0]<no<ni<zoom_ylim[1]
    assert abs(float(n.nout(ti+wi))-ni)<1e-9
    report["layout"]={"a":"arrival and service","b":"full cumulative episode",
                       "c":"normalized queue and delay","d":"same-cohort magnified cumulative detail"}
    report["magnification"]={"x_minute_limits":zoom_xlim,"count_limits_veh_lane":zoom_ylim,
                             "full_context_preserved":True,"curve_offset_applied":False,
                             "model_parameters_changed_for_visual_separation":False,
                             "arrival_time_min":ti*60,"departure_time_min":(ti+wi)*60,
                             "vertical_queue_veh_lane":ni-no,"horizontal_delay_min":wi*60}
    report["files"]=save_figure(fig,"fig3_fluid_queue",output)
    return report


def make_scheduling(output):
    n=NewellQueue(); t=np.linspace(0,n.P,1501); tm=t*60
    preferred=float(n.arrival(n.t2))
    cw,ce,cl=18.,9.,24.
    waiting=cw*n.w(t)
    early=ce*np.maximum(preferred-n.arrival(t),0)
    late=cl*np.maximum(n.arrival(t)-preferred,0)
    total=waiting+early+late
    fig,axs=plt.subplots(3,1,figsize=(5.25,5.05),sharex=True,
                        gridspec_kw={"height_ratios":[.88,.95,1.38]})
    fig.subplots_adjust(left=.14,right=.98,top=.94,bottom=.17,hspace=.54)
    a,b,c=axs
    for ax in axs:
        ax.axvspan(0,n.t2*60,facecolor=BLUE,alpha=.025)
        ax.axvspan(n.t2*60,n.P*60,facecolor=ORANGE,alpha=.025)
        # Stop the cost-panel guide below its legend so no symbol is crossed.
        ax.axvline(n.t2*60,color=GRAY,lw=.75,ls=(0,(2,2)),zorder=1,
                   ymin=.30 if ax is a else 0., ymax=.60 if ax is c else 1.)
        episode_ticks(ax,n,ax is c)
    panel(a,"a","Arrivals and service")
    a.plot(tm,n.lam(t),color=BLUE,lw=1.5,label=r"Arrival $\lambda(t)$")
    a.axhline(n.mu,color=ORANGE,lw=1.15,label=r"Service $\mu$")
    a.set(ylabel="Rate\n(veh/h/lane)",ylim=(750,2350),yticks=[900,1800])
    a.text(7,960,r"$\lambda>\mu$: queue grows",fontsize=8,color=BLUE)
    a.text(85,2110,r"$\lambda<\mu$: queue clears",fontsize=8,color=ORANGE)
    a.legend(loc="lower center",bbox_to_anchor=(.59,.015),ncol=2,handlelength=1.7,columnspacing=1.0)
    panel(b,"b","Delay and preferred arrival time")
    b.plot(tm,n.w(t)*60,color=PURPLE,lw=1.6)
    b.scatter([n.t2*60],[n.wmax*60],s=19,color=PURPLE,zorder=3)
    b.set(ylabel=r"Delay $w(t)$"+"\n(min)",ylim=(0,11.2),yticks=[0,5,10])
    b.text(7,8.7,r"$a(t)=t+T_f+w(t)$",fontsize=10)
    b.annotate(r"$t^*=a(t_2)=89.8$ min"+"\n"+r"Departure $t_2=80$ min; delay $8.89$ min",
               xy=(n.t2*60,n.wmax*60),xytext=(50,2.6),ha="center",va="center",fontsize=8,
               arrowprops={"arrowstyle":"-","color":GRAY,"lw":.7})
    panel(c,"c","Components of generalized cost")
    c.plot(tm,early,color=BLUE,lw=1.5,label=r"Early: $c_E[t^*-a(t)]_+$")
    c.plot(tm,waiting,color=PURPLE,lw=1.5,label=r"Waiting: $c_w w(t)$")
    c.plot(tm,late,color=ORANGE,lw=1.5,label=r"Late: $c_L[a(t)-t^*]_+$")
    c.plot(tm,total,color=INK,lw=1.35,ls=(0,(5,2)),label=r"Total: $G(t)$")
    c.set(ylabel="Cost per vehicle ($)",ylim=(-.35,16.7),yticks=[0,5,10,15],
           xlabel=r"Departure time from onset $t-t_0$ (min)")
    c.legend(loc="upper center",bbox_to_anchor=(.50,1.00),ncol=2,
             columnspacing=1.25,labelspacing=.45,handlelength=2.1,fontsize=7.5)
    fig.text(.14,.026,r"Illustrative costs: $c_w=18$, $c_E=9$, $c_L=24$ dollars/h; $T_f=0.923$ min."+"\n"
             +r"The arrival target is specified; these costs are not an equilibrium fit.",fontsize=7.5,linespacing=1.5)
    deriv=1+n.wprime(t)
    assert np.min(deriv)>0 and abs(float(n.arrival(n.t2))-preferred)<1e-12
    assert np.all(early[t>=n.t2]==0) and np.all(late[t<=n.t2]==0)
    assert np.max(np.abs(total-early-waiting-late))<1e-10
    report={"parameters":asdict(n),"costs_per_hour":{"waiting":cw,"early":ce,"late":cl},
            "preferred_arrival_min":preferred*60,"peak_departure_min":n.t2*60,
            "arrival_map_min_derivative":float(np.min(deriv)),
            "max_total_component_residual":float(np.max(np.abs(total-early-waiting-late))),
            "equal_cost_equilibrium_claim":False,"structural_preference_estimation":False,
            "files":save_figure(fig,"fig4_scheduling",output)}
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--only",choices=['all','scheduling'],default='all')
    args=parser.parse_args();style()
    if args.only=='scheduling':
        report=make_scheduling(args.output)
        audit=args.output/'fig4_scheduling_geometry_checks.json'
        audit.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({'geometry_checks':'PASS','audit':str(audit)}))
        return
    report={"figure3":make_fluid_queue(args.output),
            "figure4":make_scheduling(args.output),"geometry_checks":"PASS"}
    args.output.mkdir(parents=True,exist_ok=True)
    audit=args.output/"mechanics_geometry_checks.json"
    audit.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(args.output),"geometry_checks":"PASS","audit":str(audit)},indent=2))


if __name__=="__main__":
    main()
