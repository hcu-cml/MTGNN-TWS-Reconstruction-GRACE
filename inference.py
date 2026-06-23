import argparse
import json
import torch
import numpy as np
import pandas as pd
import xarray as xr
import pickle
from net.net import gtnet 
import sys
import net.util
sys.modules['util'] = net.util 

def load_adj(adj_data_path):
    with open(adj_data_path, 'rb') as f:
        adj_mx = pickle.load(f)
    return adj_mx[2]

def load_model(args, device, run_num):

    predefined_A = load_adj(args.adj_data)
    predefined_A = torch.tensor(predefined_A)-torch.eye(args.num_nodes)
    predefined_A = predefined_A.to(device)

    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                device, predefined_A=predefined_A, 
                dropout=args.dropout, subgraph_size=args.subgraph_size,
                node_dim=args.node_dim,
                dilation_exponential=args.dilation_exponential,
                conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                skip_channels=args.skip_channels, end_channels=args.end_channels,
                seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True
    ) 

    model_path = f"{args.save}/exp{args.expid}/run{run_num}/exp{args.expid}_{run_num}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def predict(model, input_data, device, scaler=None, target_scaler=None, batch_size=32):
    if scaler:
        for i, s in enumerate(scaler):
            input_data[..., i] = s.transform(input_data[..., i])

    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    input_tensor = input_tensor.transpose(1, 3)  

    all_outputs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, input_tensor.shape[0], batch_size):
            batch = input_tensor[i:i+batch_size]
            output = model(batch)
            all_outputs.append(output.cpu())
    
    output = torch.cat(all_outputs, dim=0).numpy()
    if target_scaler is not None:
        output = target_scaler[0].inverse_transform(output)
    return output

def main(save_path, run_num):
    with open(f"{save_path}/args.json", 'r') as f:
        args_dict = json.load(f)

    parser = argparse.ArgumentParser()

    for key, value in args_dict.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)

    args = parser.parse_args([])
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # === Load model ===
    model = load_model(args, device, run_num)

    # === Load input data ===
    input_data_file = f"{args.data}/inference_input.npz"
    data = np.load(input_data_file,  allow_pickle=True)
    input_data = data['x']
    print(f"Loaded input data shape: {input_data.shape}")

    # === Optional: load scaler ===
    scaler_file = f"{args.data}/scaler.pkl"
    target_scaler_file = f"{args.data}/target_scaler.pkl"
    try:
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        print("Input scaler loaded.")
        with open(target_scaler_file, 'rb') as f:
            target_scaler = pickle.load(f)
        print("Target scaler loaded.")
    except FileNotFoundError:
        scaler = None
        print("No scaler file found!")

    # === Predict and save ===
    predictions = predict(model, input_data, device, scaler, target_scaler, batch_size=args.batch_size)
    predictions_flat = predictions.squeeze(axis=(1, 3))
    print(f"Predictions shape: {predictions_flat.shape}")

    orig_data = np.load(args.orig_data_file, allow_pickle=True)

    start_date = pd.to_datetime(orig_data['daily_dates'][0])
    lat_lon = orig_data['lat_lon']
    latitudes = lat_lon[:, 0]
    longitudes = lat_lon[:, 1]
    n_time = predictions_flat.shape[0]
    time = pd.date_range(start=start_date, periods=n_time, freq="MS")

    lat_grid = np.sort(np.unique(np.round(latitudes, 4)))[::-1] 
    lon_grid = np.sort(np.unique(np.round(longitudes, 4)))       
    twsa_grid = np.full((n_time, len(lat_grid), len(lon_grid)), np.nan)

    lat_to_idx = {lat: i for i, lat in enumerate(lat_grid)}
    lon_to_idx = {lon: i for i, lon in enumerate(lon_grid)}

    for node in range(lat_lon.shape[0]):
        lat = round(latitudes[node], 4)
        lon = round(longitudes[node], 4)

        if lat in lat_to_idx and lon in lon_to_idx:
            i = lat_to_idx[lat]
            j = lon_to_idx[lon]
            twsa_grid[:, i, j] = predictions_flat[:, node]

    ds = xr.Dataset(
        data_vars={
            "twsa_pred": (("time", "lat", "lon"), twsa_grid)
        },
        coords={
            "time": time,
            "lat": lat_grid,
            "lon": lon_grid
        },
        attrs={"description": "GRACE TWSA Prediction"}
    )

    nc_output_file = f"results/inference_output_model_exp{args.expid}_{run_num}.nc"
    ds.to_netcdf(nc_output_file)
    print(f"Predictions saved as: {nc_output_file}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path', type=str, default= "save/exp101/run0/", help='Model path')
    parser.add_argument('--run_num', type=int, default=0, help='Run ID')

    args = parser.parse_args()

    main(args.save_path, args.run_num)

    # python inference.py --save_path save/exp98/run0/ --run_num 0


