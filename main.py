import argparse, os, torch
from GCGAN import GCGAN


if __name__ == '__main__':

    desc = "Pytorch implementation of GCGAN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--loss_type', type=str, default='default', choices=['default', 'ZR', 'PM', 'ZP'],
                        help='The type of loss function')
    parser.add_argument('--dataset', type=str, default='ml-100k', choices=['ml-100k', 'ml-1m'],
                        help='The name of dataset')
    parser.add_argument('--epochs', type=int, default=500, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=32, help='The size of batch')
    parser.add_argument('--Glayer', type=int, default=3, help='Number of G hidden layer')
    parser.add_argument('--Ghidden', type=int, default=500, help='Number of G node on hidden layer')
    parser.add_argument('--z_dim', type=int, default=100, help='Number of noise dimension')
    parser.add_argument('--lrG', type=float, default=0.005)
    parser.add_argument('--lrD', type=float, default=0.005)
    parser.add_argument('--save_dir', type=str, default='models', help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--eval', type=bool, default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    gan = GCGAN(args, device)

    # launch the graph in a session
    gan.train()
    print(" [*] Training finished!")
    gan.eval()
    print(" [*] Testing finished!")
    print(args)
