import xarray as xr
import pandas as pd
import numpy as np
import os
import argparse
from netCDF4 import Dataset
from io import StringIO


def load_and_preprocess(ds, lat_range, lon_range, coarsen_factor):
    """
    Load raw dataset, select spatial region, and coarsen the spatial resolution.
    Returns a resampled xarray.Dataset.
    """
    lat_name = 'lat' if 'lat' in ds.dims else 'latitude'
    lon_name = 'lon' if 'lon' in ds.dims else 'longitude'
    ds_region = ds.sel({lat_name: slice(*lat_range), lon_name: slice(*lon_range)})
    ds_resampled = ds_region.coarsen({lat_name: coarsen_factor, lon_name: coarsen_factor}, boundary='trim').mean()
    return ds_resampled

def stack_flux(ds_resampled, var_name='flux_interpolated', lat_name='lat', lon_name='lon'):
    """
    Stack spatial dimensions (lat, lon) into a single 'node' dimension and transpose to (time, node).
    Returns an xarray.DataArray.
    """    
    flux = ds_resampled[var_name]
    flux_stacked = flux.stack(node=(lat_name, lon_name)).transpose("time", "node")
    return flux_stacked

def to_dataframe(flux_stacked, time_index, suffix=""):
    """
    Convert stacked flux data into a pandas DataFrame with time index and node columns.
    Adds an optional suffix to column names.
    Returns a pandas.DataFrame.
    """
    df = pd.DataFrame(flux_stacked.values, index=time_index)
    df.columns = [f"node_{i}{suffix}" for i in range(df.shape[1])]
    return df

def read_groops_mask(file_path, col_name):
    """
    Read a Groops mask file, ignore comments, and parse the mask values.
    Returns a DataFrame with rounded lat/lon and the specified mask column.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data_lines = [line for line in lines if not line.startswith('#') and not line.strip().startswith('groops')]
    df = pd.read_csv(StringIO("".join(data_lines)), sep=r'\s+', header=None,
                    names=["lon", "lat", "height", "area", col_name])
    df['lat_rounded'] = df['lat'].round().astype(int)
    df['lon_rounded'] = df['lon'].round().astype(int)
    return df[['lat_rounded', 'lon_rounded', col_name]]

def preprocess(args):
    """
    Load raw data and generate preprocessed datasets.
    Saves output as compressed .npz file.
    """

    # === Load ERA5 ===
    ds_P = xr.open_dataset(args.era5_P)
    ds_E = xr.open_dataset(args.era5_E)
    ds_R = xr.open_dataset(args.era5_R)

    lat_range = tuple(args.lat_range)
    lon_range = tuple(args.lon_range)

    ds_P_resampled = load_and_preprocess(ds_P, lat_range, lon_range, args.coarsen)
    ds_E_resampled = load_and_preprocess(ds_E, lat_range, lon_range, args.coarsen)
    ds_R_resampled = load_and_preprocess(ds_R, lat_range, lon_range, args.coarsen)

    time_era5 = pd.to_datetime(ds_P_resampled['time'].values)

    flux_P = stack_flux(ds_P_resampled, var_name='P')
    flux_E = stack_flux(ds_E_resampled, var_name='E')
    flux_R = stack_flux(ds_R_resampled, var_name='R')

    df_P = to_dataframe(flux_P, time_era5, "_P")
    df_E = to_dataframe(flux_E, time_era5, "_E")
    df_R = to_dataframe(flux_R, time_era5, "_R")

    era5_df = pd.concat([df_P, df_E, df_R], axis=1)
    feature_order = {'P': 0, 'E': 1, 'R': 2}
    era5_df = era5_df.reindex(
        sorted(
            era5_df.columns,
            key=lambda x: (int(x.split('_')[1]), feature_order.get(x.split('_')[2], 0))
        ),
        axis=1
    )

    # === Load GRACE ===
    ds_grace = xr.open_dataset(args.grace)
    ds_resampled_grace = load_and_preprocess(ds_grace, lat_range, lon_range, args.coarsen)
    time_grace = pd.to_datetime(ds_resampled_grace['time'].values)
    flux_grace = stack_flux(ds_resampled_grace)

    grace_df = to_dataframe(flux_grace, time_grace)

    # === lat/lon mapping ===
    lat_lon = list(zip(
        flux_grace['lat'].values,
        flux_grace['lon'].values
    ))
    lat_lon = np.array(lat_lon)

    # === Leakage reduction ===
    leakagemask_df = read_groops_mask(args.leakage_mask, col_name="leakage")

    node_coords = pd.DataFrame({
        'lat': flux_grace['lat'].values,
        'lon': flux_grace['lon'].values
    })
    node_coords['lat_rounded'] = node_coords['lat'].round().astype(int)
    node_coords['lon_rounded'] = node_coords['lon'].round().astype(int)

    merged = node_coords.merge(leakagemask_df, on=['lat_rounded', 'lon_rounded'], how='left')
    merged = merged[merged['leakage'] == 1]
    merged = merged[merged['leakage'].notna()]
    valid_indices = merged.index.tolist()

    grace_df_land = grace_df[[grace_df.columns[i] for i in valid_indices]]

    era5_land_columns = []
    for i in valid_indices:
        era5_land_columns.extend([
            f"node_{i}_P",
            f"node_{i}_E",
            f"node_{i}_R"
        ])
    era5_df_land = era5_df[era5_land_columns]

    # === Save
    np.savez_compressed(
        args.output_file,
        era5=era5_df_land.values,
        grace=grace_df_land.values,
        daily_dates=era5_df.index.values,
        monthly_dates=grace_df.index.values,
        lat_lon=lat_lon[valid_indices].astype(np.float32)
    )
    print(f"Saved preprocessed data to {args.output_file}")


def main(args):
    preprocess(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ERA5 & GRACE NetCDF to .npz format")
    parser.add_argument("--era5_P", type=str, default="data/era5_P_daily_1940-01-01_2023-12-31.nc", help="ERA5 P dataset (daily)")
    parser.add_argument("--era5_E", type=str, default="data/era5_E_daily_1940-01-01_2023-12-31.nc", help="ERA5 E dataset (daily)")
    parser.add_argument("--era5_R", type=str, default="data/era5_R_daily_1940-01-01_2023-12-31.nc", help="ERA5 R dataset (daily)")
    parser.add_argument("--grace", type=str, default="data/ITSG-Grace2018_monthly_n96_replaced_C20_deg1_2002-04-01_2023-12-31_noGIA_DDK3_nomean_interpolatedCubic.nc", help="GRACE dataset (monthly)")
    parser.add_argument("--leakage_mask", type=str, default="data/leakage_mask.txt", help="Leakage mask file")
    parser.add_argument("--output_file", type=str, default="data/data_2deg_global_land_1940.npz", help="Output file (.npz)")
    parser.add_argument("--coarsen", type=int, default=1, help="Coarsening factor for grid resampling")
    parser.add_argument("--lat_range", type=float, nargs=2, default=[15, -60], help="Latitude range (min max)")  
    parser.add_argument("--lon_range", type=float, nargs=2, default=[-85, -30], help="Longitude range (min max)")  
    args = parser.parse_args()
    main(args)

    # python preprocess_data.py --era5_P="data/era5_P_daily_1940-01-01_2023-12-31.nc" --era5_E="data/era5_E_daily_1940-01-01_2023-12-31.nc" --era5_R="data/era5_R_daily_1940-01-01_2023-12-31.nc" --grace="data/ITSG-Grace2018_monthly_n96_replaced_C20_deg1_2002-04-01_2023-12-31_noGIA_DDK3_nomean_interpolatedCubic.nc" --leakage_mask="data/leakage_mask.txt" --output_file="data/data_2deg_global_land_1940.npz" --coarsen=2 --lat_range -90 90 --lon_range -180 180
