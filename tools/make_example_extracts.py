"""Write the small data extracts used by examples/02 (detector 78, November 2016).

Reads only frozen files and writes data/examples/. Re-running reproduces the same files byte for byte.
"""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from qvdfe.paths import frozen_dir, data_dir  # noqa: E402

F = frozen_dir()
OUT = data_dir() / 'examples'
OUT.mkdir(parents=True, exist_ok=True)
audit = json.loads((F / 'applicability' / 'fig7_paired_episode_audit.json').read_text(encoding='utf-8'))
SENSOR, MONTH, DATES = audit['sensor'], audit['month'], audit['dates']
src = {'daily': F / 'calibration/daily_validation/traffic_panel.csv.gz', 'monthly': F / 'calibration/average_weekday/traffic_panel.csv.gz',
       'daily_episodes': F / 'calibration/daily_validation/episode_results_v2.csv'}

d = pd.read_csv(src['daily'], usecols=['sensor_uid', 'date', 'datetime', 'flow_vph', 'speed_mph'], low_memory=False)
d = d[(d.sensor_uid == SENSOR) & d.date.astype(str).isin(DATES)].copy()
ts = pd.to_datetime(d.datetime)
d['minute'] = (ts.dt.hour * 60 + ts.dt.minute).astype(int)          # minute of day (bin start)
d = d.sort_values(['date', 'minute'])
assert (d.groupby('date').minute.nunique() == 288).all()
d[['date', 'minute', 'flow_vph', 'speed_mph']].to_csv(OUT / 'detector78_2016-11_days.csv', index=False, float_format='%.10g')

m = pd.read_csv(src['monthly'], usecols=['sensor_uid', 'month', 'minute', 'flow_vph', 'speed_mph'], low_memory=False)
m = m[(m.sensor_uid == SENSOR) & (m.month == MONTH)].sort_values('minute')
m[['minute', 'flow_vph', 'speed_mph']].to_csv(OUT / 'detector78_2016-11_average.csv', index=False, float_format='%.10g')

e = pd.read_csv(src['daily_episodes'], low_memory=False)
e = e[(e.sensor_uid == SENSOR) & e.date.astype(str).isin(DATES)]
cols = ['date', 'period', 't0', 't3', 'P_h', 'D_veh', 'x_h', 'wt2_h', 'wbar_h', 'min_speed_mph', 'vq_mph']
e[cols].sort_values(['date', 't0']).to_csv(OUT / 'detector78_2016-11_daily_episodes.csv', index=False, float_format='%.10g')

meta = {'sensor': SENSOR, 'month': MONTH, 'dates': DATES, 'representative_date': audit['representative_date'],
        'average_profile_episode': audit['average_profile_episode'], 'vf_monthly_mph': audit['vf_monthly_mph'],
        'flow_convention': 'veh/h/lane', 'averaging': 'arithmetic mean flow; flow-weighted harmonic mean speed',
        'provenance': 'legacy ADOT loop-detector data, cleaned, synchronized and post-processed for Zhou et al. (2022); research use',
        'sources': {str(p.relative_to(F)).replace(chr(92), '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in src.values()}}
(OUT / 'detector78_2016-11_meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
print({p.name: p.stat().st_size for p in OUT.glob('detector78_*')})
