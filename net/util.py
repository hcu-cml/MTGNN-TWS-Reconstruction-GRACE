import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))


class MonthlySupervisedLoader:
    def __init__(self, daily_file, monthly_file, device, normalize=True, train_ratio=0.6, valid_ratio=0.2):
        # Load daily and monthly data
        df_x = pd.read_csv(daily_file, sep=';', header=0)
        df_y = pd.read_csv(monthly_file, sep=';', header=0)

        # Convert dates
        df_x['date'] = pd.to_datetime(df_x['date'])
        df_y['date'] = pd.to_datetime(df_y['date'])

        # Group daily data by month
        df_x['month'] = df_x['date'].dt.strftime('%Y-%m')  
        grouped = df_x.groupby('month')

        # Initialize the lists for X and Y
        self.X_list = []
        self.Y_list = []

        for i, row in df_y.iterrows():
            month = row[0].to_period('M')
            if month in [pd.Period(k, freq='M') for k in grouped.groups.keys()]:
                daily_vals = grouped.get_group(str(month)).iloc[:, 1:-1].values.astype(np.float32)
                daily_vals = daily_vals[:28] 
                self.X_list.append(torch.tensor(daily_vals, dtype=torch.float32)) 
                self.Y_list.append(torch.tensor(row[1:].values.astype(np.float32), dtype=torch.float32)) 

        # Define number of samples 
        self.n = len(self.X_list)
        self.m = self.X_list[0].shape[1]  

        if normalize:
            self._normalize()

        # Normalize and split the data
        self._split(int(train_ratio * self.n), int((train_ratio + valid_ratio) * self.n), self.n)

        # Calculate scaling
        self.scale = np.ones(self.m)
        self.scale = torch.from_numpy(self.scale).float()

        # Apply scaling to the test set (example)
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        # Move to device (CPU/GPU)
        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        # Calculate error metrics (RSE, RAE)
        self.rse = normal_std(tmp)  #
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    def _normalize(self):
   
        all_x = torch.cat(self.X_list, dim=0)
        global_max = all_x.abs().max()
        print("normalize", global_max)  

        self.global_max = global_max

        for i in range(self.n):
            self.X_list[i] = self.X_list[i] / global_max

    def _split(self, train_idx, valid_idx, test_idx):
        self.train = [
            torch.stack(self.X_list[:train_idx]),
            torch.stack(self.Y_list[:train_idx])
        ]
        self.valid = [
            torch.stack(self.X_list[train_idx:valid_idx]),
            torch.stack(self.Y_list[train_idx:valid_idx])
        ]
        self.test = [
            torch.stack(self.X_list[valid_idx:test_idx]),
            torch.stack(self.Y_list[valid_idx:test_idx])
        ]


    def get_batches(self, inputs, targets, batch_size, shuffle=True):
            length = len(inputs)
            if shuffle:
                index = torch.randperm(length)
            else:
                index = torch.LongTensor(range(length))
            start_idx = 0
            while (start_idx < length):
                end_idx = min(length, start_idx + batch_size)
                excerpt = index[start_idx:end_idx]
                X = inputs[excerpt]
                Y = targets[excerpt]
                X = X.to(self.device)
                Y = Y.to(self.device)
                yield Variable(X), Variable(Y)
                start_idx += batch_size

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def fit(self, data):
        """
        Berechnet den Mittelwert und die Standardabweichung für die gegebenen Daten.
        :param data: Die Trainingsdaten, die standardisiert werden sollen.
        """
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    def __repr__(self):
        return f"StandardScaler(mean={self.mean:.4f}, std={self.std:.4f})"


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj



def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y'] 
        
    num_features = data['x_train'].shape[-1]
    means = np.mean(data['x_train'], axis=(0, 1, 2)) 
    stds = np.std(data['x_train'], axis=(0, 1, 2))
    scalers = [StandardScaler(mean=means[i], std=stds[i]) for i in range(num_features)]

    for category in ['train', 'val', 'test']:
        for feature_idx in range(num_features):
            original = data['x_' + category][:, :, :, feature_idx]
            flat = original.reshape(-1)
            scaled = scalers[feature_idx].transform(flat)
            data['x_' + category][:, :, :, feature_idx] = scaled.reshape(original.shape)

    target_num_features = data['y_train'].shape[-1]
    target_means = np.mean(data['y_train'], axis=(0,1,2))
    target_stds = np.std(data['y_train'], axis=(0,1,2))
    target_scalers = [StandardScaler(mean=target_means[i], std=target_stds[i]) for i in range(target_num_features)]

    for category in ['train', 'val', 'test']:
        for feature_idx in range(target_num_features):
            original = data['y_' + category][:, :, :, feature_idx]
            flat = original.reshape(-1)
            scaled = target_scalers[feature_idx].transform(flat)
            data['y_' + category][:, :, :, feature_idx] = scaled.reshape(original.shape)

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scalers  
    data['target_scaler'] = target_scalers  
    return data



def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_smape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    denominator = (torch.abs(preds) + torch.abs(labels)) / 2 + 1e-5
    loss = torch.abs(preds - labels) / denominator
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    mse = masked_mse(pred,real,0.0).item()
    return mse, rmse, mae


def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    z = torch.tensor((x-mean)/std,dtype=torch.float)
    return z


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))



            