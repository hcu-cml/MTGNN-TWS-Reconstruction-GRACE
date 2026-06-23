import argparse
import os
import time
import json
import pickle
import numpy as np
import torch
import random
from net.trainer import Trainer
from net.util import *
from net.net import gtnet
from train_logger import TrainLogger
from evaluation import evaluate_model


def main(runid, args, save_path):

    seed = args.seed if hasattr(args, 'seed') else 101
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)
    dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    target_scaler = dataloader['target_scaler']
    with open(os.path.join(args.data, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(args.data, 'target_scaler.pkl'), 'wb') as f:
        pickle.dump(target_scaler, f)

    predefined_A = load_adj(args.adj_data)
    predefined_A = torch.tensor(predefined_A) - torch.eye(args.num_nodes)
    predefined_A = predefined_A.to(device)


    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, predefined_A=predefined_A, dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels=args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha,
                  layer_norm_affline=True)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1,
                     args.seq_out_len, scaler, device, args.cl)

    logger = TrainLogger(args.expid, run_id, base_dir=args.save)
    print("Training gestartet")

    valid_loss_save = []
    train_loss_per_epoch, train_rmse_save, train_mae_save = [], [], []
    val_time, train_time = [], []
    minl = 1e5

    for i in range(1, args.epochs + 1):
        train_loss, train_smape, train_rmse, train_mae = [], [], [], []
        t1 = time.time()

        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device).permute(0, 3, 2, 1)
            trainy = torch.Tensor(y).to(device).permute(0, 3, 2, 1)
            id = torch.arange(args.num_nodes).to(device)
            metrics = engine.train(trainx, trainy, id)
            train_loss.append(metrics[0])
            train_rmse.append(metrics[1])
            train_mae.append(metrics[2])

        t2 = time.time()
        train_time.append(t2 - t1)

        valid_loss, valid_rmse, valid_mae = [], [], []
        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device).permute(0, 3, 2, 1)
            testy = torch.Tensor(y).to(device).permute(0, 3, 2, 1)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_rmse.append(metrics[1])
            valid_mae.append(metrics[2])

        s2 = time.time()
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_mae = np.mean(train_mae)
        mvalid_loss = np.mean(valid_loss)


        train_loss_per_epoch.append(mtrain_loss)
        train_rmse_save.append(mtrain_rmse)
        train_mae_save.append(mtrain_mae)
        valid_loss_save.append(mvalid_loss)

        logger.log_epoch(i, mtrain_loss, mvalid_loss)

        if mvalid_loss < minl:
            torch.save(engine.model.state_dict(), save_path + '/' + f"exp{args.expid}_{runid}.pth")
            minl = mvalid_loss

    print("Training abgeschlossen")

    bestid = np.argmin(valid_loss_save)
    engine.model.load_state_dict(torch.load(save_path + '/' + f"exp{args.expid}_{runid}.pth"))

    # EVALUATE VALID + TEST
    # val_metrics = evaluate_model(engine, dataloader['val_loader'], device)
    # test_metrics = evaluate_model(engine, dataloader['test_loader'], device) 
    val_metrics = evaluate_model(engine, dataloader['val_loader'], device, target_scaler=dataloader['target_scaler'])
    test_metrics = evaluate_model(engine, dataloader['test_loader'], device, target_scaler=dataloader['target_scaler'])

    # Log final results
    logger.log_metrics(val_metrics, test_metrics)
    logger.save_training_summary(train_loss_per_epoch, valid_loss_save, train_time, val_time, args)
    logger.append_to_global_summary(args, val_metrics, test_metrics, runid)


if __name__ == "__main__":
   


    def str_to_bool(value):
        if isinstance(value, bool):
            return value
        if value.lower() in {'false', 'f', '0', 'no', 'n'}:
            return False
        elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
            return True
        raise ValueError(f'{value} is not a valid boolean value')
        
    parser = argparse.ArgumentParser()

    parser.add_argument('--device',type=str,default='cuda:0',help='')
    parser.add_argument('--orig_data_file', type=str, default='data/data_1deg_southamerica_land_1940.npz')
    parser.add_argument('--data',type=str,default='data/',help='data path')

    parser.add_argument('--adj_data', type=str,default='data/A(60-40-3500)_data_1deg_southamerica_land_1940.pkl',help='adj data path') 
    parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
    parser.add_argument('--buildA_true', type=str_to_bool, default=False ,help='whether to construct adaptive adjacency matrix') 
    parser.add_argument('--load_static_feature', type=str_to_bool, default=False,help='whether to load static feature')
    parser.add_argument('--cl', type=str_to_bool, default=False,help='whether to do curriculum learning') 

    parser.add_argument('--gcn_depth',type=int,default=4,help='graph convolution depth') 
    parser.add_argument('--num_nodes',type=int,default=1120,help='number of nodes/variables') 
    parser.add_argument('--dropout',type=float,default=0.4,help='dropout rate') 
    parser.add_argument('--subgraph_size',type=int,default=500,help='k') 
    parser.add_argument('--node_dim',type=int,default=64,help='dim of nodes')
    parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

    parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels') 
    parser.add_argument('--residual_channels',type=int,default=64,help='residual channels') 
    parser.add_argument('--skip_channels',type=int,default=64,help='skip channels') 
    parser.add_argument('--end_channels',type=int,default=128,help='end channels') 


    parser.add_argument('--in_dim',type=int,default=3,help='inputs dimension') 
    parser.add_argument('--seq_in_len',type=int,default=30,help='input sequence length') 
    parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length') 

    parser.add_argument('--layers',type=int,default=4,help='number of layers')
    parser.add_argument('--batch_size',type=int,default=64,help='batch size') 
    parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
    parser.add_argument('--weight_decay',type=float,default=0.005,help='weight decay rate') 
    parser.add_argument('--clip',type=int,default=5,help='clip') 
    parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
    parser.add_argument('--step_size2',type=int,default=100,help='step_size')


    parser.add_argument('--epochs',type=int,default=40,help='') 
    parser.add_argument('--print_every',type=int,default=50,help='')
    parser.add_argument('--seed',type=int,default=101,help='random seed')
    parser.add_argument('--save',type=str,default='save/',help='save path')
    parser.add_argument('--expid',type=int,default=301,help='experiment id')

    parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha') 
    parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')

    parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')

    parser.add_argument('--runs',type=int,default=1,help='number of runs') 

    args = parser.parse_args()
    torch.set_num_threads(3)

for run_id in range(args.runs):
    save_path = os.path.join(args.save, f"exp{args.expid}", f"run{run_id}")
    main(run_id, args, save_path)

""" python train.py \
--device cuda:0 \
--data data/ \
--adj_data data/A(60-40-3500)_data_1deg_southamerica_land_1940.pkl \
--gcn_true true \
--buildA_true false \
--load_static_feature false \
--cl false \
--gcn_depth 4 \
--num_nodes 1120 \
--dropout 0.2 \
--subgraph_size 500 \
--node_dim 64 \
--dilation_exponential 1 \
--conv_channels 32 \
--residual_channels 64 \
--skip_channels 64 \
--end_channels 128 \
--in_dim 3 \
--seq_in_len 30 \
--seq_out_len 1 \
--layers 4 \
--batch_size 64 \
--learning_rate 0.0005 \
--weight_decay 0.0001 \
--clip 5 \
--step_size1 2500 \
--step_size2 100 \
--epochs 50 \
--print_every 50 \
--seed 101 \
--save save/ \
--expid 98 \
--propalpha 0.05 \
--tanhalpha 3 \
--num_split 1 \
--runs 1"""