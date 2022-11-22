def load_data(data_path):
    """
    根据文件路径data_path加载数据
    """
    sentences = []
    sentence = []
    for line in open(data_path, 'r', encoding="utf-8"):
        line = line.strip()
        if not line and len(sentence) > 0:
            if sentence[0][0] != '-DOCSTART-' and sentence[0][0] != '-docstart-':
                sentences.append(sentence)
            sentence = []
        else:
            word = line.split(" ")
            word = [word[0], word[-1]]
            sentence.append(word)
    return sentences


def build_word2seq(sentences, min_freq=1):
    """
    建立word与id的对应关系
    """
    count = dict()
    word2seq = dict()
    word2seq["NUK"] = 0
    word2seq["PAD"] = 1
    for sentence in sentences:
        for word in sentence:
            word = word[0].lower()
            count[word] = count.get(word, 0) + 1
    # count排序
    count = dict(sorted(count.items(), key=lambda i: i[-1], reverse=True))
    for key, value in count.items():
        if value > min_freq:
            word2seq[key] = len(word2seq)
    seq2word = {value: key for key, value in word2seq.items()}
    return word2seq, seq2word


def build_char2seq(sentences):
    """
    建立字母character与id的对应关系
    """
    count = dict()
    char2seq = dict()
    char2seq["unk"] = 0
    char2seq["pad"] = 1
    for sentence in sentences:
        for word in sentence:
            word = word[0]
            for char in word:
                count[char] = count.get(char, 0) + 1
    # count排序
    count = dict(sorted(count.items(), key=lambda i: i[-1], reverse=True))
    for key, value in count.items():
        char2seq[key] = len(char2seq)
    seq2char = {value: key for key, value in char2seq.items()}
    return char2seq, seq2char


def build_tag2seq(sentences):
    """
    建立tag与id的对应关系
    """
    count = dict()
    tag2seq = dict()
    tag2seq["<PAD>"] = 0
    for sentence in sentences:
        for word in sentence:
            word = word[1]
            count[word] = count.get(word, 0) + 1
    # count排序
    count = dict(sorted(count.items(), key=lambda i: i[-1], reverse=True))
    for key, value in count.items():
        tag2seq[key] = len(tag2seq)
    seq2tag = {value: key for key, value in tag2seq.items()}
    return tag2seq, seq2tag


if __name__ == '__main__':
    data_path = "CoNLL-2003/eng.train"
    sentences = load_data(data_path=data_path)
    word2seq, seq2word = build_word2seq(sentences, min_freq=0)
    char2seq, seq2char = build_char2seq(sentences)
    tag2seq, seq2tag = build_tag2seq(sentences)
    # print(word2seq)
    # print(tag2seq)
    print(char2seq)
    print(len(char2seq))
    print(len(word2seq))