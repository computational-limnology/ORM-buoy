from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## function to calculate density from temperature
def calc_dens(wtemp):
    dens = (
        999.842594
        + (6.793952 * 1e-2 * wtemp)
        - (9.095290 * 1e-3 * wtemp**2)
        + (1.001685 * 1e-4 * wtemp**3)
        - (1.120083 * 1e-6 * wtemp**4)
        + (6.536336 * 1e-9 * wtemp**5)
    )
    return dens


def thermo_depth(temp, depth):
    temp = np.asarray(temp, dtype=float)
    depth = np.asarray(depth, dtype=float)
    if depth.size < 2:
        return np.nan
    dtdz = np.gradient(temp, depth)
    return float(depth[np.nanargmax(np.abs(dtdz))])


def buoyancy_frequency(temp, depth, g=9.81):
    temp = np.asarray(temp, dtype=float)
    depth = np.asarray(depth, dtype=float)
    rho = calc_dens(temp)
    drho_dz = np.gradient(rho, depth)
    with np.errstate(divide='ignore', invalid='ignore'):
        n2 = g * drho_dz / rho
    return n2


def center_buoyancy(temp, depth, g=9.81):
    depth = np.asarray(depth, dtype=float)
    n2 = buoyancy_frequency(temp, depth, g=g)
    n2_positive = np.where(n2 > 0, n2, 0.0)
    denom = np.trapz(n2_positive, depth)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float(np.trapz(depth * n2_positive, depth) / denom)


def core_metrics(in_area, in_depth_area, in_temp, in_depth_temp, dz=0.1, g=9.81):
    in_area = np.asarray(in_area, dtype=float)
    in_depth_area = np.asarray(in_depth_area, dtype=float)
    in_temp = np.asarray(in_temp, dtype=float)
    in_depth_temp = np.asarray(in_depth_temp, dtype=float)

    max_depth = max(np.nanmax(in_depth_area), np.nanmax(in_depth_temp))
    depth = np.arange(0.0, max_depth + dz / 2.0, dz)

    area = np.interp(
        depth,
        in_depth_area,
        in_area,
        left=in_area[0],
        right=in_area[-1],
    )
    temp = np.interp(
        depth,
        in_depth_temp,
        in_temp,
        left=in_temp[0],
        right=in_temp[-1],
    )

    V = np.trapz(area, depth)
    z_v = np.trapz(depth * area, depth) / V

    rho = calc_dens(temp)
    mass = np.trapz(rho * area, depth)
    z_g = np.trapz(depth * rho * area, depth) / mass
    rho_mean = mass / V
    z_mean = V / np.max(area)
    Ws = g * z_mean * rho_mean * (z_g - z_v)

    w = area * dz
    w_sum = np.sum(w)
    if w_sum == 0:
        raise ValueError('Area weights sum to zero')
    w = w / w_sum

    rho_p = rho - rho_mean
    z_p = depth - z_v
    M1 = np.sum(w * rho_p * z_p)
    M2 = np.sum(w * rho_p * z_p**2)
    sigma_rho = np.sqrt(np.sum(w * rho_p**2))
    sigma_z = np.sqrt(np.sum(w * z_p**2))
    eta = np.nan
    if sigma_rho != 0 and sigma_z != 0:
        eta = M1 / (sigma_rho * sigma_z)
    m_ratio = np.nan
    if M1 != 0:
        m_ratio = M2 / M1

    z_therm = thermo_depth(temp, depth)
    z_n2 = center_buoyancy(temp, depth, g=g)

    n2 = buoyancy_frequency(temp, depth, g=g)
    n2_max = np.nanmax(n2)
    n2_therm = np.nan
    if np.isfinite(z_therm):
        n2_therm = float(np.interp(z_therm, depth, n2))

    b_z = -g * (rho - rho_mean) / rho_mean
    b_var = np.sum(w * b_z**2)

    return {
        'z_fluc': float(z_g - z_v),
        'Ws': float(Ws),
        'M1': float(M1),
        'M2': float(M2),
        'eta': float(eta),
        'm_ratio': float(m_ratio),
        'z_therm': float(z_therm),
        'z_n2': float(z_n2),
        'n2_max': float(n2_max),
        'b_var': float(b_var),
        'n2_therm': float(n2_therm),
    }


def process_stability_metrics(df, hypsography, exclude_depths=None):
    if 'variable' in df.columns:
        temp_df = df[df['variable'] == 'temperature'].copy()
    else:
        temp_df = df.copy()

    if 'datetime' in temp_df.columns:
        temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])

    metrics_rows = []
    temp_groups = temp_df.groupby(['site_id', 'datetime'], sort=True)
    hypsography = hypsography.sort_values('Depth_meter')
    depth_area = hypsography['Depth_meter'].values
    area = hypsography['Area_meterSquared'].values

    for (site_id, datetime_value), group in temp_groups:
        group = group.sort_values('depth')
        
        # Filter out excluded depths
        if exclude_depths is not None:
            group = group[~group['depth'].isin(exclude_depths)].copy()
        
        if group['depth'].isna().any() or group['observation'].isna().any():
            continue

        profile_metrics = core_metrics(
            in_area=area,
            in_depth_area=depth_area,
            in_temp=group['observation'].values,
            in_depth_temp=group['depth'].values,
        )
        profile_metrics['site_id'] = site_id
        profile_metrics['datetime'] = datetime_value
        metrics_rows.append(profile_metrics)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(['site_id', 'datetime']).reset_index(drop=True)
    return metrics_df




script_dir = Path(__file__).resolve()
root_dir = script_dir.parent.parent

# Read the parquet file from output/
parquet_path = root_dir / "output" / "orm_long.parquet"
df = pd.read_parquet(parquet_path)

# Read the hypsography
hypsography_path = root_dir / "lake" / "lake_bathymetry.csv"
hypsography = pd.read_csv(hypsography_path)

# Process all metrics for the available temperature profiles
stability_metrics_df = process_stability_metrics(df, hypsography, exclude_depths=[0.5])
print(f"Processed stability metrics for {len(stability_metrics_df)} profiles")

# Save reduced data plus metrics in long format
metrics_long = (
    stability_metrics_df
    .melt(
        id_vars=["site_id", "datetime"],
        var_name="variable",
        value_name="observation",
    )
    .dropna(subset=["observation"])
)
metrics_long["depth"] = 0.0
metrics_long = metrics_long[["datetime", "site_id", "depth", "observation", "variable"]]

reduced_final = df[["datetime", "site_id", "depth", "observation", "variable"]].copy()
combined_final = pd.concat([reduced_final, metrics_long], ignore_index=True, sort=False)
combined_final = combined_final.sort_values(["datetime", "site_id", "depth", "variable"]).reset_index(drop=True)

merged_output_parquet = root_dir / "output" / "orm_long_processed.parquet"

combined_final.to_parquet(merged_output_parquet, index=False)

print(f"Saved reduced data with metrics long format to: {merged_output_parquet}")
