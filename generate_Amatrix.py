import argparse
import pickle
import numpy as np
from tqdm import tqdm
from geopy.distance import geodesic


def lagged_correlation(x, y, max_lag=30):
    best = -1
    for lag in range(0, max_lag + 1):
        if lag > 0:
            x_lagged = x[:-lag]
            y_lagged = y[lag:]
        else:
            x_lagged = x
            y_lagged = y
        if len(x_lagged) < 2:
            continue
        r = np.corrcoef(x_lagged, y_lagged)[0, 1]
        if np.isnan(r):
            r = 0
        best = max(best, abs(r))
    return best


def build_hybrid_adjacency(lat_lon_list, timeseries, alpha=0.5, spatial_sigma=300, max_lag=30, feature_weights=None):
    T, N, D = timeseries.shape
    print(f"Input shape: T={T}, N={N}, D={D}")

    if feature_weights is None:
        feature_weights = np.ones(D) / D

    climate_sim = np.zeros((N, N))
    print("Compute climate similarity ...")
    for i in tqdm(range(N), desc="climate-similarity"):
        for j in range(i, N):
            sim_total = 0
            for d in range(D):
                ts_i = timeseries[:, i, d]
                ts_j = timeseries[:, j, d]
                sim = lagged_correlation(ts_i, ts_j, max_lag=max_lag)
                sim_scaled = (sim + 1) / 2
                sim_total += feature_weights[d] * sim_scaled
            climate_sim[i, j] = sim_total
            climate_sim[j, i] = sim_total

    spatial_sim = np.zeros((N, N))
    print("Compute distance ...")
    for i in tqdm(range(N), desc="dictance"):
        for j in range(i, N):
            if i == j:
                spatial_sim[i, j] = 1.0
            else:
                dist_km = geodesic(lat_lon_list[i], lat_lon_list[j]).km
                sim = np.exp(-dist_km / spatial_sigma)
                spatial_sim[i, j] = sim
                spatial_sim[j, i] = sim

    A = alpha * climate_sim + (1 - alpha) * spatial_sim
    return A


def main(args):
    data = np.load(args.input_data)
    raw = data['era5'].T 
    raw = raw.reshape(args.num_nodes, args.num_features, -1)
    timeseries = np.transpose(raw, (2, 0, 1))

    lat_lon = data['lat_lon']
    node_ids = [f"node_{i}" for i in range(len(lat_lon))]
    node_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}

    adj = build_hybrid_adjacency(
        lat_lon_list=lat_lon,
        timeseries=timeseries,
        alpha=args.alpha,
        spatial_sigma=args.spatial_sigma,
        max_lag=args.max_lag
    )

    output = [node_ids, node_to_index, adj]
    with open(args.output_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"Adjacency matrix save as: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data', type=str, default='data\data_1deg_southamerica_land_1940.npz', help='Path to .npz file')
    parser.add_argument('--output_path', type=str, default='data\A(60-40-3500)_data_1deg_southamerica_land_1940.pkl', help='output file name')
    parser.add_argument('--alpha', type=float, default=0.6, help='Weight between climate and distance similartity (alpha * climate)')
    parser.add_argument('--spatial_sigma', type=float, default=3500, help='Scalar for distance')
    parser.add_argument('--max_lag', type=int, default=30, help='Maximum lag foo climate korrelation')
    parser.add_argument('--num_nodes', type=int, default=1120, help='Number of nodes')
    parser.add_argument('--num_features', type=int, default=4, help='Number of features')

    args = parser.parse_args()
    main(args)

    #python generate_Amatrix.py --input_data PATH/TO/INPUT.npz --output_path PATH/TO/OUTPUT.pkl
