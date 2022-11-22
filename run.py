import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import Config
from utils import *
from dataloader import get_loader
from model import BiLSTM_CRF


class Runner(object):
    def __init__(self,
                 train_loader,
                 test_loader,
                 word2seq,
                 char2seq,
                 tag2seq,
                 config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.word2seq = word2seq
        self.char2seq = char2seq
        self.tag2seq = tag2seq
        self.seq2tag = {value: key for key, value in tag2seq.items()}
        self.config = config
        self.best_F1 = 0
        self.model = BiLSTM_CRF(word2seq=self.word2seq,
                                char2seq=self.char2seq,
                                tag2seq=self.tag2seq,
                                config=config).to(config.device)
        if config.continue_from is not None:
            self.model.load_state_dict(torch.load(config.continue_from))

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: self.config.lr_decay ** epoch)
        # 开始训练
        for epoch_idx in range(self.config.epoch):
            # train
            self.model.train()
            for index, data in tqdm(enumerate(train_loader), total=len(train_loader)):
                sentences, chars, tags = data
                sentences, chars, tags = sentences.to(self.config.device), chars.to(self.config.device), tags.to(self.config.device)
                optimizer.zero_grad()
                loss = self.model(sentences, chars, tags)
                loss.backward()
                optimizer.step()
                # if index % 10 == 0:
                #     print("EPOCH[{}/{}] INTER[{}/{}] loss: {:.3}".format(epoch_idx + 1, self.config.epoch, index, len(train_loader), loss.item()))
            lr_scheduler.step()
            self.test(epoch_idx)
        print("Best F1: {:.3f}".format(self.best_F1))
        torch.save(self.model.state_dict(), self.config.model_path)

    def test(self, epoch_idx):
        # 开始测试
        self.model.eval()
        test_loss = 0
        tp = [0] * len(self.tag2seq)
        fp = [0] * len(self.tag2seq)
        fn = [0] * len(self.tag2seq)
        for index, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            with torch.no_grad():
                sentences, chars, tags = data
                sentences, chars, tags = sentences.to(self.config.device), chars.to(self.config.device), tags.to(self.config.device)
                loss = self.model(sentences, chars, tags)
                pred = self.model(sentences, chars)
                test_loss += loss
                for key in seq2tag:
                    correct = torch.where(pred == key, key, -1).eq(torch.where(tags == key, key, -2)).sum().item()
                    tp[key] = tp[key] + correct
                    fp[key] = fp[key] + pred.eq(key).sum().item() - correct
                    fn[key] = fn[key] + tags.eq(key).sum().item() - correct
        # 打印各类 P R F1
        print()
        for i in range(len(self.tag2seq)):
            P = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) != 0 else 0
            R = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) != 0 else 0
            F1 = 2 * P * R / (P + R) if (P + R) != 0 else 0
            print(self.seq2tag[i].ljust(8), end=":")
            print("{:.3f}  {:.3f}  {:.3f}".format(P, R, F1), end="\t")
            print(tp[i], tp[i] + fp[i], tp[i] + fn[i])
        # 打印综合 P R F1
        tp = sum(tp[2:])
        fp = sum(fp[2:])
        fn = sum(fn[2:])
        P = tp / (tp + fp) if (tp + fp) != 0 else 0
        R = tp / (tp + fn) if (tp + fn) != 0 else 0
        F1 = 2 * P * R / (P + R) if (P + R) != 0 else 0
        self.best_F1 = max(self.best_F1, F1)
        print("OVERALL".ljust(8), end=":")
        print("{:.3f}  {:.3f}  {:.3f}".format(P, R, F1), end="\t")
        print(tp, tp + fp, tp + fn)
        # 计算loss
        test_loss = test_loss / len(test_loader)
        print("\nEPOCH[{}/{}] loss:{:.6f}".format(epoch_idx + 1, self.config.epoch, test_loss))


if __name__ == '__main__':
    # 加载配置
    config = Config()

    # 准备数据
    train_sentences = load_data(data_path=config.train_data_path)
    word2seq, seq2word = build_word2seq(train_sentences, min_freq=config.min_freq)
    print("words dictionary:", len(word2seq))
    char2seq, seq2char = build_char2seq(train_sentences)
    print("characters dictionary:", len(char2seq))
    tag2seq, seq2tag = build_tag2seq(train_sentences)
    print("tags dictionary:", len(tag2seq))
    train_loader = get_loader(data_path=config.train_data_path, word2seq=word2seq, char2seq=char2seq,
                              tag2seq=tag2seq, max_len=config.max_len,
                              max_char_len=config.max_char_len, batch_size=config.batch_size)
    print("total train set:", len(train_loader.dataset))
    test_loader = get_loader(data_path=config.test_data_path, word2seq=word2seq, char2seq=char2seq,
                             tag2seq=tag2seq, max_len=config.max_len,
                             max_char_len=config.max_char_len, batch_size=config.batch_size)
    print("total test set:", len(test_loader.dataset))

    # run
    runner = Runner(train_loader=train_loader, test_loader=test_loader, word2seq=word2seq, char2seq=char2seq, tag2seq=tag2seq, config=config)
    runner.train()
