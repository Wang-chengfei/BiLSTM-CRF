import argparse
import torch
import os
import random
import numpy as np


class Config(object):
    def __init__(self):
        # get init config
        args = self.__get_config()
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])

        # select device
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_cuda else "cpu")

        # set the random seed
        self.__set_seed(self.seed)

    @staticmethod
    def __set_seed(seed):
        os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # set seed for cpu
        torch.cuda.manual_seed(seed)  # set seed for current gpu
        torch.cuda.manual_seed_all(seed)  # set seed for all gpu

    @staticmethod
    def __get_config():
        parser = argparse.ArgumentParser()
        parser.description = 'config for models'

        # data directory
        parser.add_argument('--train_data_path', type=str, default='./CoNLL-2003/train.txt',
                            help='path to load data')
        parser.add_argument('--eval_data_path', type=str, default='./CoNLL-2003/valid.txt',
                            help='path to load data')
        parser.add_argument('--test_data_path', type=str, default='./CoNLL-2003/test.txt',
                            help='path to load data')
        parser.add_argument('--bert_path', type=str, default='E:/PycharmProjects/bert-base-uncased',
                            help='bert path')
        # word/char embedding
        parser.add_argument('--embedding_path', type=str, default='./model/word_embedding.pt',
                            help='path to load pre-trained word/char embedding')
        parser.add_argument('--embedding_dim', type=int, default=100,
                            help='dimension of word embedding')
        parser.add_argument('--char_embedding_dim', type=int, default=50,
                            help='dimension of character embedding')
        parser.add_argument('--min_freq', type=float, default=1,
                            help='minimum token frequency when constructing vocabulary list')
        parser.add_argument('--max_len', type=int, default=100,
                            help='max length of sentence')
        parser.add_argument('--max_char_len', type=int, default=15,
                            help='max length of word')

        # train settings
        parser.add_argument('--model_path', type=str, default='./model/BiLSTM_CRF.pt',
                            help='path to save model')
        parser.add_argument('--continue_from', type=str, default=None,
                            help='continue train from a specific model')
        parser.add_argument('--seed', type=int, default=2022,
                            help='random seed')
        parser.add_argument('--use_cuda', type=bool, default=True,
                            help='whether need to use cuda')
        parser.add_argument('--epoch', type=int, default=20,
                            help='max epochs during training')

        # hyper parameters
        parser.add_argument('--batch_size', type=int, default=100,
                            help='batch size')
        parser.add_argument('--use_pretrain_embedding', type=bool, default=True,
                            help='whether to use pretrain embedding')
        parser.add_argument('--use_crf', type=bool, default=True,
                            help='whether to use crf')
        parser.add_argument('--use_att', type=bool, default=False,
                            help='whether to use multi head attention')
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='the possiblity of dropout')
        parser.add_argument('--hidden_size', type=int, default=200,
                            help='the LSTM hidden size')
        parser.add_argument('--lr', type=float, default=2e-3,
                            help='learning rate')
        parser.add_argument('--lr_decay', type=float, default=0.95,
                            help='learning rate decay')
        parser.add_argument('--weight_decay', type=float, default=1e-5,
                            help='weight decay')

        args = parser.parse_args()
        return args

    def print_config(self):
        for key in self.__dict__:
            print(key, end=' = ')
            print(self.__dict__[key])


if __name__ == '__main__':
    config = Config()
    config.print_config()
