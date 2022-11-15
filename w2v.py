import os
import torch
import numpy as np
from gensim.models import word2vec
from utils import *
from dataloader import get_dataset
from config import Config


def train_word2vec():
    """
    训练word2vec模型
    """
    # 加载配置
    config = Config()
    # 准备数据
    train_sentences = load_data(data_path=config.train_data_path)
    word2seq, seq2word = build_word2seq(train_sentences, min_freq=config.min_freq)
    char2seq, seq2char = build_char2seq(train_sentences)
    tag2seq, seq2tag = build_tag2seq(train_sentences)
    train_dataset = get_dataset(data_path=config.train_data_path, word2seq=word2seq, char2seq=char2seq, tag2seq=tag2seq,
                                max_len=config.max_len, max_char_len=config.max_char_len, )
    # 准备句子
    sentences = []
    for sentence in train_dataset:
        sentence = [seq2word.get(word_idx) for word_idx in sentence[0]]
        sentences.append(sentence)
    # 训练word_embedding模型
    embedding_model = word2vec.Word2Vec(sentences, vector_size=config.embedding_dim, min_count=config.min_freq, window=10)
    # 保存模型
    embedding_model.save(config.embedding_path)


def load_word2vec(embedding_path, word2seq):
    """
    加载word2vec矩阵
    """
    if not os.path.exists(embedding_path):
        train_word2vec()
    embedding_model = word2vec.Word2Vec.load(embedding_path)
    word_vectors = [embedding_model.wv[word] if word in embedding_model.wv else np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                    for word in word2seq.keys()]
    word_vectors = np.array(word_vectors)
    word_vectors = torch.from_numpy(word_vectors).float()
    return word_vectors


if __name__ == '__main__':
    train_word2vec()
