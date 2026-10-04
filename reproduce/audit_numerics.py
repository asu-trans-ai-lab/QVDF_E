# Reproduction entry; reads the frozen inputs through qvdfe.paths and writes to reproduce/output.
"""Numerical audit from the frozen predictions: reference state, exclusion ledger, emission metrics, recorded-speed
recalculation with the archived operating-mode rates, and the later-date diagnostic with a null benchmark.

This does not fit a QVDF model or replace rates. Run from anywhere. Dependencies: numpy and pandas.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent / 'lib'))  # reproduce/lib
from _paths import ROOT as _ROOT, FROZEN as _FROZEN, OUT as _OUT  # noqa: E402
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

PACKAGE=_ROOT
DATA=_FROZEN / 'calibration'
OUT=_OUT/'audit'
pass  # emissions.py is in reproduce/lib
from emissions import load_moves_rates,compute_vsp,get_op_mode_bin

POLS=['CO2','NOx','CO','HC']
LABELS={'CO2':r'CO$_2$','NOx':r'NO$_x$','CO':'CO','HC':'HC'}

def scores(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    assert np.isfinite(y).all() and np.isfinite(p).all()
    err=p-y
    return dict(N=len(y),R2=float(1-np.sum(err**2)/np.sum((y-y.mean())**2)),
                RMSE=float(np.sqrt(np.mean(err**2))),MAE=float(np.mean(abs(err))),
                bias_pct=float(100*err.sum()/y.sum()),
                WAPE_pct=float(100*np.abs(err).sum()/np.abs(y).sum()))

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def table3():
    selection=json.loads((DATA/'pigou/pigou_selection.json').read_text())
    c=selection['context']; x=selection['x_reference_h']
    P=max(c['fd']*x**c['n'],x); C=c['C_vphpl']; vf=c['vf_mph']; L=c['L_mi']; Tf=L/vf
    mu=C*x/P;omega=C/(c['kj']-C/vf);kq=c['kj']-mu/omega;vq=mu/kq
    peak=Tf*c['fp']*P**c['s'];wbar=c['theta']*peak
    rates=pd.read_csv(DATA/'cubic_rate_coefficients.csv',float_precision='round_trip')
    before={'CO2':(294.557980720482,31.5541186575651,326.112099378047),
            'NOx':(.413780,.34784806181390254-.413780,.34784806181390254),
            'CO':(1.952736,2.5662767929588752-1.952736,2.5662767929588752),
            'HC':(.081926,.10820955711178307-.081926,.10820955711178307)}
    rows=[]
    for row in rates.to_dict('records'):
        p=row['pollutant']; c0,c1,c2,c3=[row[k] for k in ['c0','c1','c2','c3']]
        Gamma=c0-c2*vf*vq-c3*vf*vq*(vf+vq)
        e0=Tf*(c0+vf*(c1+vf*(c2+vf*c3)))
        inc=wbar*Gamma
        rows.append(dict(pollutant=p,e0_g_vehicle=e0,Gamma_g_vehicle_h=Gamma,
                         increment_g_vehicle=inc,total_g_vehicle=e0+inc,
                         rounded_input_total_g_vehicle=before[p][2],
                         difference_from_rounded_input_g_vehicle=e0+inc-before[p][2]))
    pd.DataFrame(rows).to_csv(OUT/'table3_full_precision_reference.csv',index=False)
    state=dict(reference_sensor_uid=selection['reference_sensor_uid'],period=selection['period'],
               calibration_scope=c['calibration_scope'],N_train=c['N_train'],N_source=c['N_source'],
               reference_interpretation='Conditional median reference; not an observed month or episode.',
               x_h=x,P_h=P,C_vphpl=C,vf_mph=vf,L_mi=L,Tf_h=Tf,mu_vphpl=mu,
               omega_mph=omega,kq_veh_mi_lane=kq,vq_mph=vq,peak_delay_h=peak,
               mean_delay_h=wbar,kj_veh_mi_lane=c['kj'],
               traffic_parameters={k:c[k] for k in ['fd','n','fp','s','theta']},
               cubic_rate_sha256=hashlib.sha256((DATA/'cubic_rate_coefficients.csv').read_bytes()).hexdigest(),
               rates=[{k:r[k] for k in ['pollutant','c0','c1','c2','c3','speed_min_mph','speed_max_mph']} for r in rates.to_dict('records')])
    dv_mps=(vf-vq)*.44704
    down_s=1.5*dv_mps/1.5; up_s=1.5*dv_mps/1.0
    half_h=(down_s+up_s)/7200
    lower_h=(vf-vq)/vf*half_h
    upper_h=L*(1/vq-1/vf)-(vf-vq)/vq*half_h
    transitions=dict(peak_deceleration_mps2=1.5,peak_acceleration_mps2=1.0,
                     T_down_s=down_s,T_up_s=up_s,
                     distance_down_mi=(vf+vq)/2*down_s/3600,
                     distance_up_mi=(vf+vq)/2*up_s/3600,
                     admissible_delay_lower_min=lower_h*60,
                     admissible_delay_upper_min=upper_h*60,
                     mean_delay_min=wbar*60,peak_delay_min=peak*60,
                     peak_free_flow_portion_s=(Tf-vq*peak/(vf-vq))*3600,
                     required_half_transitions_s=(down_s+up_s)/2,
                     mean_admitted=bool(lower_h<=wbar<=upper_h),
                     peak_admitted=bool(lower_h<=peak<=upper_h))
    state['transition_geometry']=transitions
    dump('table3_reference_state.json',state)
    return state,rows

def emission_accounting():
    ep=pd.read_csv(DATA/'average_weekday/episode_results_v2.csv')
    fitted=ep.calibration_eligible & np.isfinite(ep.P_hat_h)
    remaining=fitted.copy();reason=pd.Series('not_calibration_eligible',index=ep.index)
    reason[ep.calibration_eligible&~fitted]='unsupported_fitted_model'
    stages=[]
    for col,name in [('physical_predicted_pass','finite_link_admission'),
                     ('profile_closed_pass','profile_reconstruction'),
                     ('cubic_domain_pass','common_rate_domain'),
                     ('matched_ladder_pass','common_positive_finite_emissions')]:
        flag=ep[col].fillna(False).astype(bool)
        failed=remaining&~flag
        reason[failed]=name
        remaining &= flag
        stages.append(dict(stage=name,excluded=int(failed.sum()),remaining=int(remaining.sum())))
    reason[remaining]='retained_common_cohort'
    assert int(fitted.sum())==286 and int(remaining.sum())==137
    assert remaining.equals(ep.matched_ladder_pass)
    ledger=ep[['episode_id','corridor','period','sensor_uid','date','calibration_eligible',
               'P_hat_h','physical_predicted_pass','profile_closed_pass','cubic_domain_pass',
               'matched_ladder_pass','physical_failure_reason','x_h','P_h','vt2_obs_mph']].copy()
    ledger['within_fitted_denominator']=fitted
    ledger['sequential_reason']=reason
    ledger.to_csv(OUT/'episode_exclusion_ledger.csv',index=False)
    pd.DataFrame(stages).to_csv(OUT/'sequential_exclusion_counts.csv',index=False)
    medians=[]
    for name,mask in [('retained',remaining),('excluded',fitted&~remaining)]:
        medians.append(dict(population=name,N=int(mask.sum()),median_x_h=float(ep.loc[mask,'x_h'].median()),
                            median_P_h=float(ep.loc[mask,'P_h'].median()),
                            median_vt2_mph=float(ep.loc[mask,'vt2_obs_mph'].median())))
    pd.DataFrame(medians).to_csv(OUT/'retained_excluded_medians.csv',index=False)
    cohort=ep.loc[remaining]; metrics=[]
    saved=pd.read_csv(DATA/'average_weekday/emission_metrics.csv')
    for p in POLS:
        for m in ['M','S']:
            result=dict(pollutant=p,method=m,**scores(cohort['O_'+p],cohort[m+'_'+p]))
            old=saved[(saved['sample']=='All average-weekday profiles')&(saved.pollutant==p)&(saved.method==m)].iloc[0]
            for k in ['N','R2','RMSE','MAE','bias_pct','WAPE_pct']:
                assert np.isclose(result[k],old[k],rtol=1e-12,atol=1e-12),(p,m,k)
            metrics.append(result)
    pd.DataFrame(metrics).to_csv(OUT/'emission_metrics_reproduced.csv',index=False)
    # Independently reconstruct O from the saved five-minute observations using
    # the archived exact operating-mode implementation, not cubic approximation.
    panel=pd.read_csv(DATA/'average_weekday/traffic_panel.csv.gz',parse_dates=['datetime'],low_memory=False)
    panel['date']=panel.datetime.dt.strftime('%Y-%m-%d')
    groups={k:g for k,g in panel.groupby(['sensor_uid','date'])}
    moves=load_moves_rates(DATA/'emission_input')
    checks=[]
    for r in cohort.to_dict('records'):
        g=groups[(r['sensor_uid'],r['date'])]
        g=g[(g.datetime>=pd.Timestamp(r['t0']))&(g.datetime<pd.Timestamp(r['t3']))]
        v=g.speed_mph.to_numpy();q=g.flow_vph.to_numpy()
        vsp=compute_vsp(v,moves.vehicle_params)
        modes=[get_op_mode_bin(a,b) for a,b in zip(v,vsp)]
        for p in POLS:
            rate=moves.rate_for_modes(modes,'NOX' if p=='NOx' else p)
            value=float(np.sum((q/12)*(r['L_mi']/v)*rate))
            recorded=float(r['O_'+p]);difference=value-recorded
            assert np.isclose(value,recorded,rtol=1e-12,atol=1e-9),(r['episode_id'],p,value,recorded)
            checks.append(dict(episode_id=r['episode_id'],pollutant=p,recorded_g=recorded,
                               recalculated_g=value,difference_g=difference))
    pd.DataFrame(checks).to_csv(OUT/'recorded_speed_O_recalculation.csv',index=False)
    summary=dict(detected=len(ep),eligible=int(ep.calibration_eligible.sum()),fitted=int(fitted.sum()),
                 stages=stages,retained=int(remaining.sum()),excluded=int((fitted&~remaining).sum()),
                 excluded_fraction=float((fitted&~remaining).sum()/fitted.sum()),medians=medians,
                 physical_failure_reasons=ep.loc[fitted&~remaining,'physical_failure_reason'].value_counts().to_dict(),
                 O_max_abs_error_g=max(abs(r['difference_g']) for r in checks),
                 O_accounting='sum(q_vphpl/12 * L_mi/v_mph * rate_g_vehicle_hour) within [t0,t3)',
                 WAPE='100 sum(abs(predicted_episode_total - O_episode_total)) / sum(O_episode_total)')
    dump('emission_audit.json',summary)
    return summary,metrics

def testing_diagnostic():
    ep=pd.read_csv(DATA/'daily_validation/episode_results_v2.csv')
    cards=pd.read_csv(DATA/'daily_validation/parameter_card_full.csv')
    # Availability and validity checks only. No residual, observed delay/ratio,
    # or geometric-admission criterion enters this alternative evaluation mask.
    measurement=(~ep.incomplete_episode & ~ep.invalid_observed_speed_or_flow &
                 ~ep.day_boundary_censored & ~ep.metadata_capacity_flag &
                 np.isfinite(ep.x_h)&ep.x_h.gt(0)&np.isfinite(ep.C_vphpl)&ep.C_vphpl.gt(0))
    train=ep[~ep.holdout & ep.calibration_eligible]
    rows=[];lookup={}
    for card in cards.to_dict('records'):
        source=train[train.corridor.eq(card['corridor'])]
        if card['calibration_scope']=='corridor_period':source=source[source.period.eq(card['period'])]
        assert len(source)==card['N_source']
        ratios=source.x_h/source.P_h
        assert np.isfinite(ratios).all() and (ratios>0).all()
        theta_mu=float(ratios.median())
        lookup[(card['corridor'],card['period'])]=theta_mu
        rows.append(dict(corridor=card['corridor'],period=card['period'],
                         calibration_scope=card['calibration_scope'],N_source=len(source),
                         theta_mu_bar=theta_mu,estimator='median(x_h/P_h), original eligible training cohort'))
    pd.DataFrame(rows).to_csv(OUT/'null_training_discharge_ratios.csv',index=False)
    t=ep[ep.holdout].copy(); t['measurement_qc_pass']=measurement[ep.holdout]
    t['theta_mu_bar_train']=[lookup[(c,p)] for c,p in zip(t.corridor,t.period)]
    t['P_null_hat_h']=t.x_h/t.theta_mu_bar_train
    # All testing predictions were stored before this diagnostic; verify against
    # the archived coefficients without optimizing or tuning them.
    assert np.allclose(t.P_hat_h,np.maximum(t.fd*t.x_h**t.n,t.x_h),equal_nan=True)
    results=[]
    masks=[('Published eligible sample',t.calibration_eligible),('Measurement QC only',t.measurement_qc_pass)]
    for sample,mask in masks:
        for target,pred in [('P_h','P_hat_h'),('vt2_obs_mph','vt2_hat_mph'),('wt2_h','wt2_hat_h')]:
            finite=mask&np.isfinite(t[target])&np.isfinite(t[pred])
            if target=='P_h':finite &= np.isfinite(t.P_null_hat_h)
            z=t[finite]
            metric=scores(z[target],z[pred])
            if target=='P_h':
                null=scores(z[target],z.P_null_hat_h)
                results.append(dict(sample=sample,target=target,model='Null median discharge ratio',**null))
                delta={f'delta_{k}':metric[k]-null[k] for k in ['R2','RMSE','MAE']}
            else:delta={}
            results.append(dict(sample=sample,target=target,model='Frozen QVDF',**metric,**delta))
    pd.DataFrame(results).to_csv(OUT/'testing_robustness_metrics.csv',index=False)
    cols=['episode_id','date','corridor','period','measurement_qc_pass','calibration_eligible',
          'qc_hard_fail','qc_outlier_fail','incomplete_episode','invalid_observed_speed_or_flow',
          'day_boundary_censored','metadata_capacity_flag','x_h','P_h','P_hat_h',
          'theta_mu_bar_train','P_null_hat_h','vt2_obs_mph','vt2_hat_mph','wt2_h','wt2_hat_h']
    t[cols].to_csv(OUT/'testing_diagnostic_episode_ledger.csv',index=False)
    raw_exclusions=[];remaining=pd.Series(True,index=t.index)
    for col in ['incomplete_episode','invalid_observed_speed_or_flow','day_boundary_censored','metadata_capacity_flag']:
        failure=remaining&t[col]
        raw_exclusions.append(dict(reason=col,excluded=int(failure.sum())))
        remaining &= ~t[col]
    failure=remaining&~(np.isfinite(t.x_h)&t.x_h.gt(0)&np.isfinite(t.C_vphpl)&t.C_vphpl.gt(0))
    raw_exclusions.append(dict(reason='nonfinite_or_nonpositive_demand_capacity',excluded=int(failure.sum())))
    summary=dict(detected_testing=len(t),measurement_qc_pass=int(t.measurement_qc_pass.sum()),
                 original_hard_qc_pass=int((~t.qc_hard_fail).sum()),published_eligible=int(t.calibration_eligible.sum()),
                 restored_residual_exclusions=int((t.measurement_qc_pass&~t.calibration_eligible).sum()),
                 excluded_by_measurement_qc=raw_exclusions,
                 training_last_date=str(train.date.max()),testing_first_date=str(t.date.min()),
                 target_finite_pair_counts={r['target']:r['N'] for r in results if r['sample']=='Measurement QC only'},
                 null_estimator='median of x/P on original eligible training rows in each original corridor-period source; original all-period fallback if used',
                 mask_definition='complete speed/flow rows; raw speeds in (0,90] mph and nonnegative flows; uncensored day boundary; unflagged capacity metadata; finite positive x and capacity. No delay positivity, mean/peak ratio, target magnitude, residual or finite-link screen.',
                 interpretation='Fit-residual-independent evaluation conditional on speed-based detection and measurement QC; not independent of observation availability.',
                 metrics=results)
    dump('testing_robustness_specification.json',summary)
    return summary

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    state,reference=table3()
    emission,metrics=emission_accounting()
    testing=testing_diagnostic()
    dump('numerical_audit_summary.json',dict(reference_state=state,reference_components=reference,
                                               emissions=emission,emission_metrics=metrics,testing=testing))
    print(json.dumps(dict(reference_CO2_g_vehicle=reference[0]['total_g_vehicle'],
                         fitted=emission['fitted'],retained=emission['retained'],
                         O_max_error_g=emission['O_max_abs_error_g'],
                         testing_QC_N=testing['measurement_qc_pass'],checks='passed'),indent=2))

if __name__=='__main__':main()
