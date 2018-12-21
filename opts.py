import argparse
import os

def opts():
    parser = argparse.ArgumentParser(description='Train resnet on the Office-31 dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # dataset options
    parser.add_argument('--data_path_source', type=str, default='data/mnist/imgs_train_source', help='Root of the source dataset')
    parser.add_argument('--data_path_target', type=str, default='data/mnist/imgs_train_target', help='Root of the target dataset')

    # training options
    parser.add_argument("--train", default=False, action='store_true', help="True for training, False for testing (default: True)")
    parser.add_argument("--mode", type=str, default='edge', choices=['edge', 'negative'], help="Training mode for training, 'edge' or 'negative' (default: 'edge')")


    # optimization options
    parser.add_argument("--epochs", type=int, default=25000, help="Number of iterations (default: 25000)")
    parser.add_argument("--batch_size", type=int, default=16, help="The size of batch images (default: 128)")
    parser.add_argument('--schedule', type=int, nargs='+', default=[1000, 2200, 3200, 4000, 4600], help='Decrease learning rate at these epochs')
    parser.add_argument("--lr", type=float, default=0.0002, help="The learning rate of gradient descent algorithm (default: 1e-4)")
    parser.add_argument("--optimizer", type=str, default="adam", help="Type of optimizer, 'sgd' or 'adam'(default: adam)")
    parser.add_argument('--momentum_1st', type=float, default=0.5, help='The 1st momentum for Adam')
    parser.add_argument('--momentum_2nd', type=float, default=0.999, help='The 2nd momentum for Adam')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty)') # default=0.0001
    parser.add_argument('--seed', type=int, default=666, help='The random seed for pytorch (default: 666)')

    # checkpoints options
    parser.add_argument('--start_epoch', type=int, default=0, help='Manual epoch number (used for restarting)')
    parser.add_argument('--resume', type=str, default='', help='Path of the checkpoint file to resume (default empty)')
    
    # i/o options
    parser.add_argument('--log', type=str, default='log', help='Directory of log and checkpoint file')
    # parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", help="Name of checkpoint directory (default: checkpoint)")
    # parser.add_argument("--result_dir", type=str, default="result", help="Name of result directory (default: result)")
    parser.add_argument('--num_workers', type=int, default=1, help='Number of data loading workers (default: 4)')
    parser.add_argument('--test_freq', type=int, default=500, help='The frequency printing the result while testing (default: 500 epoch)')
    parser.add_argument('--print_freq', type=int, default=10, help='The frequency printing the result while training (default: 10 epoch)')
    parser.add_argument('--record_freq', type=int, default=500, help='The frequency recoding the result into the log file while training (default: 500 epoch)')

    args = parser.parse_args()

    return args