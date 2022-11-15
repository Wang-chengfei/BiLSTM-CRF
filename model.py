import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from w2v import load_word2vec


class BiLSTM_CRF(nn.Module):
    def __init__(self,
                 word2seq,
                 char2seq,
                 tag2seq,
                 config,
                 num_heads=8):
        super(BiLSTM_CRF, self).__init__()
        self.word2seq = word2seq
        self.char2seq = char2seq
        self.tag2seq = tag2seq
        self.config = config

        self.char_embedding = nn.Embedding(len(self.char2seq), self.config.char_embedding_dim)
        self.char_cnn = nn.Conv2d(in_channels=self.config.max_char_len, out_channels=1, kernel_size=3, stride=1, padding=1)

        if self.config.use_pretrain_embedding:
            self.embedding = nn.Embedding.from_pretrained(load_word2vec(self.config.embedding_path, word2seq), freeze=False)
        else:
            self.embedding = nn.Embedding(len(self.word2seq), self.config.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.config.embedding_dim + self.config.char_embedding_dim, hidden_size=self.config.hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.fc = nn.Linear(self.config.hidden_size * 2, len(self.tag2seq))
        if self.config.use_crf:
            self.crf = CRF(num_tags=len(self.tag2seq), batch_first=True)
        if self.config.use_att:
            self.mul_att = nn.MultiheadAttention(embed_dim=self.config.hidden_size * 2, batch_first=True, dropout=self.config.dropout, num_heads=num_heads)

    def encode(self, sentences):
        output, (h_n, _) = self.lstm(sentences)  # output [batch_size, length, hidden_size * 2] h_n [num_lays*num_direction, batch_size, hidden_size]
        output = self.dropout(output)
        if self.config.use_att:
            output, _ = self.mul_att(output, output, output)
        emit_score = self.fc(output)  # emit_score[batch_size, length, len(tag2seq)]
        return emit_score

    def predict(self, sentences, chars):
        """
        计算预测值
        sentences:[batch_size, max_len]
        chars: [batch_size, max_len, max_char_len]
        """
        seq_len = sentences.size()[-1]
        mask = sentences != 1
        sentences = self.embedding(sentences)  # sentences:[batch_size, length, embedding_dim]
        chars = self.char_embedding(chars).transpose(1, 2)  # chars: [batch_size, max_len, max_char_len, char_embedding_dim]
        chars = self.char_cnn(chars).squeeze(1)
        sentences = torch.cat((sentences, chars), -1)
        emit_score = self.encode(sentences)
        if self.config.use_crf:
            pred = self.crf.decode(emissions=emit_score, mask=mask)
            for pred_tag in pred:
                if len(pred_tag) < seq_len:
                    pred_tag += [len(self.tag2seq) - 1] * (seq_len - len(pred_tag))
            pred = torch.tensor(pred).to(self.config.device)
        else:
            pred = emit_score.transpose(1, 2)
            pred = pred.argmax(1)
        return pred

    def forward(self, sentences, chars, tags):
        """
        计算loss
        sentences:[batch_size, max_len]
        chars: [batch_size, max_len, max_char_len]
        tags:[batch_size, max_len]
        """
        mask = sentences != 1
        sentences = self.embedding(sentences)  # sentences[batch_size, max_len, embedding_dim]
        chars = self.char_embedding(chars).transpose(1, 2)  # chars: [batch_size, max_len, max_char_len, char_embedding_dim]
        chars = self.char_cnn(chars).squeeze(1)
        sentences = torch.cat((sentences, chars), -1)
        emit_score = self.encode(sentences)
        if self.config.use_crf:
            loss = self.crf(emissions=emit_score, tags=tags, mask=mask) * -1
        else:
            pred = emit_score.transpose(1, 2)
            loss = torch.nn.functional.cross_entropy(pred, tags)
        return loss


