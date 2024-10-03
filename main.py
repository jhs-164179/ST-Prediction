import warnings
warnings.filterwarnings('ignore')

import sys
import time
import torch
import argparse
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler
from dataset import MovingMNIST, TaxibjDataset
from model import InvVP_Model
from metrics import *
from utils import *


if __name__ == '__main__':
    seed_everything()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='taxibj')
    parser.add_argument('--in_shape', type=parse_tuple, default=(4, 2, 32, 32), help='Plz provide as "(T,C,H,W)"')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--loss', type=str, default='l1l2')
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--save_path', type=str, default='base_result')

    args = parser.parse_args()
    
    # load model params
    config = load_configs(f'./configs/{args.data}.json')
    config['in_shape'] = args.in_shape
    model = InvVP_Model(
        **config
    )
    # analysis model params and flops
    flops = FlopCountAnalysis(model, torch.randn(1, *args.in_shape))
    gflops = flops.total() / 1e9
    flop_table = flop_count_table(flops)

    # setting environment(device, optimizer, scheduler and loss function)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    if args.scheduler == 'cosine':
        scheduler = CosineLRScheduler(
            optimizer=optimizer, t_initial=args.epochs, lr_min=1e-06, warmup_lr_init=1e-05, warmup_t=args.warmup_epoch
        )
    elif args.scheduler == 'onecycle':
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=10
        )
    else:
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='min', factor=.5, verbose=True
        )
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()

    # load dataset and dataloader
    if args.data == 'mmnist':
        train_set = MovingMNIST(root=os.path.join(args.data_path, 'moving_mnist'), is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[2])
        test_set = MovingMNIST(root=os.path.join(args.data_path, 'moving_mnist'), is_train=False, n_frames_input=10, n_frames_output=10, num_objects=[2])
    elif args.data == 'taxibj':
        dataset = np.load(os.path.join(args.data_path, 'taxibj/dataset.npz'))
        X_train, Y_train, X_test, Y_test = dataset['X_train'], dataset['Y_train'], dataset['X_test'], dataset['Y_test']
        train_set = TaxibjDataset(X=X_train, Y=Y_train)
        test_set = TaxibjDataset(X=X_test, Y=Y_test)
    else:
        print('Additional Datasets are not supported yet')
        sys.exit()

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)

    best_loss = np.inf
    best_epoch = 0
    save_path = f'./logs/{args.save_path}'
    check_path(save_path)
    with open(os.path.join(save_path, f'{args.data}_{args.epochs}epochs.txt'), 'w') as log_file:
        print_f(model, file=log_file)
        print_f(flop_table, file=log_file)
        print_f(f'model params: {count_parameters(model)}', file=log_file)
        print_f(f'model gflops: {gflops}', file=log_file)
        for epoch in range(args.epochs):
            # train
            model.train()
            t0 = time.time()
            loss_train = 0.0
            for X, y in tqdm(train_loader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                preds = model(X)
                if args.loss == 'l1l2':
                    loss = 10*criterion1(preds, y) + criterion2(preds, y)
                else:
                    loss = criterion1(preds, y)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
            loss_train = loss_train / len(train_loader)
            print_f(f'Epoch {epoch + 1} | Loss : {loss_train:.6f} | Time : {time.time() - t0:.4f}', file=log_file)

            # test
            model.eval()
            t1 = time.time()
            with torch.no_grad():
                loss_test = 0.0
                for X, y in tqdm(test_loader):
                    X, y = X.to(device), y.to(device)
                    preds = model(X)
                    loss = criterion1(preds, y)
                    loss = criterion1(preds, y)
                    loss_test += loss.item()
                loss_test = loss_test / len(test_loader)
                if args.scheduler == 'onecycle':
                    scheduler.step(epoch=epoch)
                else:
                    scheduler.step(epoch=epoch, metric=loss_test)
                print_f(f'Epoch {epoch + 1} | Test Loss : {loss_test:.6f} | Time : {time.time() - t1:.4f}', file=log_file)

                # save best model
                if loss_test < best_loss:
                    best_loss = loss_test
                    best_epoch = epoch + 1
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(save_path, 'best_model.pth'))
                    print_f(f'Best model saved with loss {best_loss:.6f} at epoch {epoch + 1}', file=log_file)

        # save last model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_path, 'last_model.pth'))

        # evaluate with various metrics
        checkpoint = torch.load(os.path.join(save_path, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        with torch.no_grad():
            total_mse, total_mae, total_psnr, total_ssim = 0, 0, 0, 0
            print_f('===Final Evaluation===', file=log_file)
            for X, y in tqdm(test_loader):
                X = X.to(device)
                preds = model(X)
                preds = preds.detach().cpu().numpy()
                true = y.numpy()
                preds = np.maximum(preds, 0)
                preds = np.minimum(preds, 1)

                total_mse += MSE(preds, true)
                total_mae += MAE(preds, true)
                total_psnr += cal_psnr(preds, true)
                total_ssim += cal_ssim(preds, true)
            total_mse /= len(test_loader)
            total_mae /= len(test_loader)
            total_psnr /= len(test_loader)
            total_ssim /= len(test_loader)
            print_f(f'Best Epoch {best_epoch} | MSE : {total_mse:.6f} | MAE : {total_mae:.6f} | PSNR : {total_psnr:.6f} | SSIM : {total_ssim:.6f}', file=log_file)

