"""MOVES operating-mode emission benchmark for reconstructed QVDFE profiles.

The equations and input-file interpretation are intentionally aligned with the
previous QVDFE paper package.  Acceleration is held at zero because the source
traffic products contain five-minute mean speeds rather than trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("qvdfe_cbi")
MPH_TO_MPS = 0.44704
MIN_SPEED_MPH = 1.0
POLLUTANTS = {
    "CO2": "meanBaseRate_CO2_(g/hr)",
    "NOX": "meanBaseRate_NOX_(g/hr)",
    "CO": "meanBaseRate_CO_(g/hr)",
    "HC": "meanBaseRate_HC_(g/hr)",
}


def compute_vsp(speed_mph, vehicle_params: dict, acceleration_mps2: float = 0.0) -> np.ndarray:
    v = np.asarray(speed_mph, dtype=float) * MPH_TO_MPS
    mass = max(float(vehicle_params["source_mass"]), 1e-12)
    return (
        float(vehicle_params["rolling_term_a"]) * v
        + float(vehicle_params["rotating_term_b"]) * v**2
        + float(vehicle_params["drag_term_c"]) * v**3
        + mass * v * float(acceleration_mps2)
    ) / mass


def get_op_mode_bin(speed_mph: float, vsp: float) -> int | None:
    if not np.isfinite(speed_mph) or not np.isfinite(vsp):
        return None
    if speed_mph <= 0:
        return 1
    if speed_mph < 25:
        for threshold, mode in [(0, 11), (3, 12), (6, 13), (9, 14), (12, 15)]:
            if vsp < threshold:
                return mode
        return 16
    if speed_mph < 50:
        for threshold, mode in [(0, 21), (3, 22), (6, 23), (9, 24), (12, 25), (18, 27), (24, 28), (30, 29)]:
            if vsp < threshold:
                return mode
        return 30
    for threshold, mode in [(6, 33), (12, 35), (18, 37), (24, 38), (30, 39)]:
        if vsp < threshold:
            return mode
    return 40


@dataclass
class MovesRates:
    vehicle_params: dict
    rates: pd.DataFrame

    def rate_for_modes(self, modes, pollutant: str) -> np.ndarray:
        lookup = self.rates.set_index("OpModeID")[f"ER_{pollutant}"].to_dict()
        return np.asarray([lookup.get(int(x), np.nan) if pd.notna(x) else np.nan for x in modes], dtype=float)


def load_moves_rates(input_dir: Path, vehicle_type: int = 1) -> MovesRates:
    input_dir = Path(input_dir)
    vehicles = pd.read_csv(input_dir / "input_vehicle_type.csv")
    raw = pd.read_csv(input_dir / "input_vehicle_emission_rate.csv")
    selected = vehicles.loc[vehicles["vehicle_type"].eq(vehicle_type)]
    if selected.empty:
        raise ValueError(f"MOVES vehicle_type={vehicle_type} is absent")
    params = selected.iloc[0].to_dict()
    weights = pd.DataFrame(
        [
            {"age": int(c.rsplit("_", 1)[-1]), "age_weight": float(params[c])}
            for c in vehicles.columns
            if c.startswith("percentage_of_age_")
        ]
    )
    weights["age_weight"] /= weights["age_weight"].sum()
    merged = raw.loc[raw["vehicle_type"].eq(vehicle_type)].merge(weights, on="age", how="inner")
    rows: list[dict] = []
    for mode, group in merged.groupby("OpModeID"):
        row = {"OpModeID": int(mode)}
        for pollutant, column in POLLUTANTS.items():
            row[f"ER_{pollutant}"] = float((group[column] * group["age_weight"]).sum())
        rows.append(row)
    return MovesRates(params, pd.DataFrame(rows).sort_values("OpModeID").reset_index(drop=True))


def fit_cubic_surrogates(moves: MovesRates) -> pd.DataFrame:
    speeds = np.linspace(5.0, 75.0, 701)
    vsp = compute_vsp(speeds, moves.vehicle_params)
    modes = [get_op_mode_bin(v, p) for v, p in zip(speeds, vsp)]
    rows = []
    for pollutant in POLLUTANTS:
        rates = moves.rate_for_modes(modes, pollutant)
        good = np.isfinite(rates)
        descending = np.polyfit(speeds[good], rates[good], 3)
        pred = np.polyval(descending, speeds[good])
        ss_tot = np.sum((rates[good] - rates[good].mean()) ** 2)
        rows.append(
            {
                "pollutant": pollutant,
                "speed_min_mph": 5.0,
                "speed_max_mph": 75.0,
                "c0": descending[3],
                "c1": descending[2],
                "c2": descending[1],
                "c3": descending[0],
                "rate_r2": 1.0 - np.sum((rates[good] - pred) ** 2) / ss_tot if ss_tot else np.nan,
                "rate_rmse_gph": float(np.sqrt(np.mean((rates[good] - pred) ** 2))),
                "negative_grid_share": float(np.mean(pred < 0)),
            }
        )
    return pd.DataFrame(rows)


def build_rate_speed_curve(moves: MovesRates) -> pd.DataFrame:
    """Return the exact zero-acceleration MOVES lookup over a dense speed grid."""
    speed = np.linspace(1.0, 80.0, 791)
    vsp = compute_vsp(speed, moves.vehicle_params)
    modes = pd.array([get_op_mode_bin(v, p) for v, p in zip(speed, vsp)], dtype="Int64")
    out = pd.DataFrame({"speed_mph": speed, "VSP": vsp, "OpModeID": modes})
    for pollutant in POLLUTANTS:
        out[f"ER_{pollutant}_gph"] = moves.rate_for_modes(modes, pollutant)
    return out


def _load_profile_traffic(output_root: Path, profiles: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["episode_id", "dataset", "length_mi", "episode_demand", "flow_synthetic"]
    meta = episodes[meta_cols].drop_duplicates("episode_id")
    frame = profiles.merge(meta, on="episode_id", how="left")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["flow_vph"] = np.nan
    frame["length_qc_mi"] = np.nan
    for dataset, needed in frame.groupby("dataset"):
        if pd.isna(dataset):
            continue
        source = Path(output_root) / "datasets" / str(dataset) / "stage1" / "qc_repaired.csv.gz"
        sensors = set(needed["sensor_uid"].astype(str))
        keys = needed[["sensor_uid", "datetime"]].drop_duplicates()
        pieces = []
        for chunk in pd.read_csv(
            source,
            usecols=["sensor_uid", "datetime", "flow_vph", "length_mi"],
            chunksize=250_000,
        ):
            chunk["sensor_uid"] = chunk["sensor_uid"].astype(str)
            chunk = chunk.loc[chunk["sensor_uid"].isin(sensors)]
            if not chunk.empty:
                chunk["datetime"] = pd.to_datetime(chunk["datetime"])
                pieces.append(chunk.merge(keys, on=["sensor_uid", "datetime"], how="inner"))
        if pieces:
            traffic = pd.concat(pieces, ignore_index=True).groupby(
                ["sensor_uid", "datetime"], as_index=False
            )[["flow_vph", "length_mi"]].median()
            idx = frame["dataset"].eq(dataset)
            joined = frame.loc[idx, ["sensor_uid", "datetime"]].merge(
                traffic.rename(columns={"length_mi": "length_qc_mi"}),
                on=["sensor_uid", "datetime"],
                how="left",
            )
            frame.loc[idx, "flow_vph"] = joined["flow_vph"].to_numpy()
            frame.loc[idx, "length_qc_mi"] = joined["length_qc_mi"].to_numpy()
    frame["flow_source"] = np.where(frame["flow_vph"].notna(), "interval_qc", "episode_mean_fallback")
    frame["flow_vph"] = frame["flow_vph"].fillna(frame["episode_demand"])
    frame["length_mi"] = frame["length_qc_mi"].fillna(frame["length_mi"])
    return frame.drop(columns=["length_qc_mi"])


def _add_scenario(frame: pd.DataFrame, speed_col: str, prefix: str, moves: MovesRates) -> pd.DataFrame:
    speed = pd.to_numeric(frame[speed_col], errors="coerce")
    speed_for_travel_time = speed.clip(lower=MIN_SPEED_MPH)
    vsp = compute_vsp(speed, moves.vehicle_params)
    modes = pd.array([get_op_mode_bin(v, p) for v, p in zip(speed, vsp)], dtype="Int64")
    frame[f"{prefix}_speed_for_travel_time_mph"] = speed_for_travel_time
    frame[f"{prefix}_VSP"] = vsp
    frame[f"{prefix}_OpModeID"] = modes
    travel_time = frame["length_mi"] / speed_for_travel_time
    volume = frame["flow_vph"] * (5.0 / 60.0)
    for pollutant in POLLUTANTS:
        rate = moves.rate_for_modes(modes, pollutant)
        frame[f"{prefix}_rate_{pollutant}_gph"] = rate
        frame[f"{prefix}_emission_{pollutant}_g_per_lane"] = rate * travel_time * volume
    return frame


def run_emission_benchmark(repo_root: Path, output_root: Path) -> dict[str, pd.DataFrame]:
    """Calculate interval and episode MOVES emissions for Stage 5 profiles."""
    output_root = Path(output_root)
    out = output_root / "stage6_emissions"
    out.mkdir(parents=True, exist_ok=True)
    profiles = pd.read_csv(
        output_root / "stage5_speed_reconstruction" / "speed_profiles.csv.gz",
        dtype={"sensor_uid": str},
    )
    episodes = pd.read_csv(
        output_root / "stage2b_outliers" / "clean_valid_episodes.csv",
        dtype={"sensor_uid": str},
    )
    moves = load_moves_rates(Path(repo_root) / "qvdfe-experiments" / "emission_input")
    frame = _load_profile_traffic(output_root, profiles, episodes)
    frame = _add_scenario(frame, "observed_speed_mph", "observed", moves)
    frame = _add_scenario(frame, "predicted_speed_mph", "reconstructed", moves)

    group_cols = ["episode_id", "detector", "corridor", "sensor_uid"]
    agg = {}
    for pollutant in POLLUTANTS:
        agg[f"observed_emission_{pollutant}_g_per_lane"] = (
            f"observed_emission_{pollutant}_g_per_lane",
            "sum",
        )
        agg[f"reconstructed_emission_{pollutant}_g_per_lane"] = (
            f"reconstructed_emission_{pollutant}_g_per_lane",
            "sum",
        )
    episode = frame.groupby(group_cols, as_index=False).agg(**agg)
    for pollutant in POLLUTANTS:
        obs = episode[f"observed_emission_{pollutant}_g_per_lane"]
        pred = episode[f"reconstructed_emission_{pollutant}_g_per_lane"]
        episode[f"relative_error_{pollutant}_pct"] = 100.0 * (pred - obs) / obs.replace(0, np.nan)

    metric_rows = []
    for keys, group in episode.groupby(["detector", "corridor"]):
        for pollutant in POLLUTANTS:
            obs = group[f"observed_emission_{pollutant}_g_per_lane"].to_numpy(float)
            pred = group[f"reconstructed_emission_{pollutant}_g_per_lane"].to_numpy(float)
            good = np.isfinite(obs) & np.isfinite(pred) & (obs > 0)
            obs, pred = obs[good], pred[good]
            ss_tot = np.sum((obs - obs.mean()) ** 2) if len(obs) else np.nan
            metric_rows.append(
                {
                    "detector": keys[0],
                    "corridor": keys[1],
                    "pollutant": pollutant,
                    "n_episodes": len(obs),
                    "observed_total_g_per_lane": float(obs.sum()),
                    "reconstructed_total_g_per_lane": float(pred.sum()),
                    "total_bias_pct": float(100 * (pred.sum() - obs.sum()) / obs.sum()) if obs.sum() else np.nan,
                    "episode_MAPE_pct": float(np.mean(np.abs(pred - obs) / obs) * 100) if len(obs) else np.nan,
                    "episode_RMSE_g_per_lane": float(np.sqrt(np.mean((pred - obs) ** 2))) if len(obs) else np.nan,
                    "episode_R2": float(1 - np.sum((pred - obs) ** 2) / ss_tot) if np.isfinite(ss_tot) and ss_tot > 0 else np.nan,
                }
            )
    metrics = pd.DataFrame(metric_rows)
    opmodes = []
    for scenario in ("observed", "reconstructed"):
        counts = frame.groupby(["detector", "corridor", f"{scenario}_OpModeID"]).size().rename("n_intervals").reset_index()
        counts["scenario"] = scenario
        counts = counts.rename(columns={f"{scenario}_OpModeID": "OpModeID"})
        counts["share"] = counts["n_intervals"] / counts.groupby(["detector", "corridor"])["n_intervals"].transform("sum")
        opmodes.append(counts)
    opmodes = pd.concat(opmodes, ignore_index=True)
    surrogates = fit_cubic_surrogates(moves)
    rate_curve = build_rate_speed_curve(moves)

    frame.to_csv(out / "interval_emissions.csv.gz", index=False, compression="gzip")
    episode.to_csv(out / "episode_emissions.csv", index=False)
    metrics.to_csv(out / "emission_metrics.csv", index=False)
    opmodes.to_csv(out / "opmode_distribution.csv", index=False)
    moves.rates.to_csv(out / "moves_age_weighted_rates.csv", index=False)
    surrogates.to_csv(out / "moves_cubic_surrogates.csv", index=False)
    rate_curve.to_csv(out / "moves_rate_speed_curve.csv", index=False)
    LOGGER.info(
        "Stage 6 emissions: %s intervals, %s episodes, %.2f%% interval-flow fallback",
        len(frame),
        len(episode),
        100 * frame["flow_source"].ne("interval_qc").mean(),
    )
    return {
        "interval": frame,
        "episode": episode,
        "metrics": metrics,
        "opmodes": opmodes,
        "rates": moves.rates,
        "surrogates": surrogates,
        "rate_curve": rate_curve,
    }
