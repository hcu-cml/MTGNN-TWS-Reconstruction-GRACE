from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(x_df, y_df, x_offsets, y_offsets, num_features, num_nodes):
    """
    Generate input-output sequences for graph-based seq2seq modeling.
    x_df: ERA5 data (daily)
    y_df: GRACE data (monthly)
    x_offsets: time offsets for input (e.g., [-29, ..., 0])
    y_offsets: time offsets for target (e.g., [0])
    """
    x_list, y_list = [], []
    target_timestamps = []

    x_df = x_df.copy()
    x_df['date'] = x_df.index

    # === Create a sorted list of all unique (year, month) periods ===
    x_df['year_month'] = x_df['date'].dt.to_period('M')
    unique_year_months = np.sort(x_df['year_month'].unique())

    for target_ym in unique_year_months:
        
        last_day_of_month = x_df[x_df['year_month'] == target_ym]['date'].max()
        if pd.isnull(last_day_of_month):
            continue

        mask = (x_df['date'] <= last_day_of_month) & \
               (x_df['date'] >= last_day_of_month - pd.Timedelta(days=np.abs(x_offsets[0])))
        input_data = x_df.loc[mask].drop(columns=['year_month', 'date'])

        if len(input_data) < np.abs(x_offsets[0]):
            print(f"Skipping {target_ym}: only {len(input_data)} days available (need {np.abs(x_offsets[0])})")
            continue

        grace_vals = []
        for offset in y_offsets:
            offset_month = (target_ym.to_timestamp() + pd.DateOffset(months=offset)).to_period('M')
            try:
                grace_val = y_df.loc[f"{offset_month.year}-{offset_month.month:02d}"].values
                grace_vals.append(grace_val)
            except KeyError:
                grace_vals = None
                break 

        if grace_vals is None or len(grace_vals) != len(y_offsets):
            print(f"Skipping {target_ym}: Not enough GRACE data for offsets {y_offsets}")
            continue
        
        target_timestamps.append(last_day_of_month)
        x_list.append(input_data.values)
        y_list.append(np.stack(grace_vals))  

    if not x_list:
        raise ValueError("No valid samples found! Please check the data.")

    # === Final stacking and reshaping ===
    x_array = np.stack(x_list)  
    y_array = np.stack(y_list)  

    x_array = x_array.reshape(x_array.shape[0], x_array.shape[1], num_nodes, num_features)
    y_array = y_array.reshape(y_array.shape[0], y_array.shape[1], num_nodes, 1)

    print("x shape:", x_array.shape, ", y shape:", y_array.shape)
    return x_array, y_array, np.array(target_timestamps)


def generate_train_val_test(args):
    """
    Load raw data and generate training, validation, and test sets.
    Saves the output in compressed .npz format.
    """
    data = np.load(args.df_filename)
    era5_values = data["era5"]
    grace_values = data["grace"]
    daily_dates = pd.to_datetime(data["daily_dates"])
    monthly_dates = pd.to_datetime(data["monthly_dates"])

    era5_df = pd.DataFrame(era5_values, index=daily_dates)
    grace_df = pd.DataFrame(grace_values, index=monthly_dates)

    x_offsets = np.arange(-29, 1, 1)  
    y_offsets = np.array([0])        

    num_features = int(era5_df.shape[1]/grace_df.shape[1])
    num_nodes = int(grace_df.shape[1])

    # === Generate sequences ===
    x, y, target_timestamps = generate_graph_seq2seq_io_data(era5_df, grace_df, x_offsets, y_offsets, num_features, num_nodes)

    # === Define custom time splits ===
    train_start, train_end = pd.Timestamp("2002-04-01"), pd.Timestamp("2017-06-30")
    val_start, val_end = pd.Timestamp("2018-06-01"), pd.Timestamp("2021-03-31")
    test_start, test_end = pd.Timestamp("2021-04-01"), pd.Timestamp("2022-12-31")

    target_timestamps = pd.to_datetime(target_timestamps)

    train_mask = (target_timestamps >= train_start) & (target_timestamps <= train_end)
    val_mask = (target_timestamps >= val_start) & (target_timestamps <= val_end)
    test_mask = (target_timestamps >= test_start) & (target_timestamps <= test_end)

    # === Apply masks ===
    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    # === Save to disk ===
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def generate_inference_data(x_df, x_offsets, num_features, num_nodes):
    """
    Generate input sequences for inference (no y/target).
    """
    x_list = []

    x_df = x_df.copy()
    x_df['date'] = x_df.index
    x_df['year_month'] = x_df['date'].dt.to_period('M')
    unique_year_months = np.sort(x_df['year_month'].unique())

    for target_ym in unique_year_months:
        last_day_of_month = x_df[x_df['year_month'] == target_ym]['date'].max()

        if pd.isnull(last_day_of_month):
            continue

        mask = (x_df['date'] <= last_day_of_month) & \
               (x_df['date'] >= last_day_of_month - pd.Timedelta(days=np.abs(x_offsets[0])))
        input_data = x_df.loc[mask].drop(columns=['year_month', 'date'])

        if len(input_data) < np.abs(x_offsets[0]):
            print(f"Skipping {target_ym}: only {len(input_data)} days available (need {np.abs(x_offsets[0])})")
            continue

        x_list.append(input_data.values)

    if not x_list:
        raise ValueError("No valid inference samples found!")

    x_array = np.stack(x_list) 
    x_array = x_array.reshape(x_array.shape[0], x_array.shape[1], num_nodes, num_features)

    print("Inference x shape:", x_array.shape)
    return x_array


def generate_inference_dataset(args):
    """
    Wrapper for generating and saving inference input.
    """
    data = np.load(args.df_filename)
    era5_values = data["era5"]
    grace_values = data["grace"]

    daily_dates = pd.to_datetime(data["daily_dates"])
    monthly_dates = pd.to_datetime(data["monthly_dates"])

    era5_df = pd.DataFrame(era5_values, index=daily_dates)
    grace_df = pd.DataFrame(grace_values, index=monthly_dates)

    x_offsets = np.arange(-29, 1, 1) 

    num_features = int(era5_df.shape[1]/grace_df.shape[1])
    num_nodes = int(grace_df.shape[1])

    # === Generate sequences ===
    x_inference = generate_inference_data(era5_df, x_offsets, num_features, num_nodes)

    # === Save to disk ===
    np.savez_compressed(
        os.path.join(args.output_dir, "inference_input.npz"),
        x=x_inference,
        x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1])
    )



def main(args):
    """
    Main entry point. Selects between training and inference mode.
    """
    if args.mode == "train":
        print("Generating training data")
        generate_train_val_test(args)
    elif args.mode == "inference":
        print("Generating inference data")
        generate_inference_dataset(args)
    else:
        raise ValueError(f"Unknown mode {args.mode}. Use 'train' or 'inference'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/", help="Output directory.")
    parser.add_argument("--df_filename", type=str, default="data/data_1deg_southamerica_land_1940.npz", help="GRACE, ERA5, global (2°), 1940-2023")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="inference",
                        help="Mode: 'train' to generate train/val/test splits, 'inference' to generate inference inputs.")
    args = parser.parse_args()
    main(args)

    # python generate_training_data.py --output_dir=data --df_filename=C:\Users\dmz-user\Desktop\Lara\MTGNN_newGRACE\data\data_1deg_dailyera5unfiltered_monthlygrace_unfiltered_southamerica_land_seperate_features_1940.npz --mode=inference

    