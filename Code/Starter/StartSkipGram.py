import os
import sys
import argparse
sys.path.append('..')
from DataProcess.Streamer import Streamer
from Models.SkipGram import SkipGram
from Trainer.Trainer import SkipGramTrainer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    opts = read_commands()
    streamer = Streamer(opts)
    print('Data streamer has prepared well!')
    skipgram = SkipGram(opts)
    solver = SkipGramTrainer(opts, streamer, skipgram)
    if opts.is_training:
        solver.train()
        solver.test()
    else:
        solver.test()


def read_commands():
    parser = argparse.ArgumentParser(usage='Train word2vec model!')
    root = os.path.abspath('../..')
    data_root = os.path.join(root, 'Data')
    parser.add_argument('--is_training', action='store_true', default=True)
    parser.add_argument('--graph_name', type=str, default='SkipGram')
    parser.add_argument('--tag', type=str, default='SkipGram')
    parser.add_argument('--data_set_name', type=str, default='imdb')
    parser.add_argument('--data_path', type=str, default=os.path.join(data_root, 'train/imdb_words_parallel.pkl'))
    parser.add_argument('--save_dir', type=str, default=os.path.join(data_root, 'model'))
    parser.add_argument('--log_dir', type=str, default=os.path.join(data_root, 'log'))
    parser.add_argument('--util_dir', type=str, default=os.path.join(data_root, 'util'))
    parser.add_argument('--neg_sampling_num', type=int, default=6)
    parser.add_argument('--neg_sampling_p_factor', type=float, default=0.75)
    parser.add_argument('--sub_sampling_p_factor', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--num_units', type=int, default=None)
    parser.add_argument('--min_count', type=int, default=5)
    parser.add_argument('--emb_dim', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=500000)
    parser.add_argument('--learning_rate', type=float, default=1.0)
    parser.add_argument('--learning_rate_factor', type=float, default=1.0)
    parser.add_argument('--display_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=50000)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
