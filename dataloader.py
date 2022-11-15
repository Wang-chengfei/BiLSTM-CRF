import torch
from utils import load_data, build_word2seq, build_char2seq, build_tag2seq
from torch.utils.data import Dataset, DataLoader


class CoNLLDataset(Dataset):
    def __init__(self,
                 data_path,
                 word2seq,
                 char2seq,
                 tag2seq,
                 max_len=100,
                 max_char_len=10):
        sentences = load_data(data_path)
        self.sentences = []
        self.characters = []
        self.labels = []
        for sentence in sentences:
            this_sentence = []
            this_characters = []
            this_label = []
            for word in sentence:
                this_sentence.append(word2seq.get(word[0].lower(), 0))
                this_characters.append([char2seq.get(character, 0) for character in word[0]])
                this_label.append(tag2seq.get(word[1], 0))
            # 裁剪填充句子至指定长度
            # 裁剪
            if len(this_sentence) >= max_len:
                this_sentence = this_sentence[:max_len]
                this_characters = this_characters[:max_len]
                this_label = this_label[:max_len]
            # 填充
            else:
                this_sentence = this_sentence + [1] * (max_len - len(this_sentence))
                this_characters.extend([[1] for i in range((max_len - len(this_characters)))])
                this_label = this_label + [len(tag2seq) - 1] * (max_len - len(this_label))
            # 裁剪填充char
            padded_this_characters = []
            for character in this_characters:
                if len(character) >= max_char_len:
                    padded_character = character[:max_char_len]
                else:
                    padded_character = character + [1] * (max_char_len - len(character))
                padded_this_characters.append(padded_character)
            self.sentences.append(this_sentence)
            self.characters.append(padded_this_characters)
            self.labels.append(this_label)

    def __getitem__(self, item):
        return self.sentences[item], self.characters[item], self.labels[item]

    def __len__(self):
        return len(self.sentences)


def collate_fn(batch):
    sentence, char, tags = list(zip(*batch))
    sentence = torch.LongTensor(sentence)
    char = torch.LongTensor(char)
    tags = torch.LongTensor(tags)
    return sentence, char, tags


def get_loader(data_path, word2seq, char2seq, tag2seq, max_len=100, max_char_len=10, batch_size=100, shuffle=True):
    dataset = CoNLLDataset(data_path=data_path, word2seq=word2seq, char2seq=char2seq, tag2seq=tag2seq, max_len=max_len, max_char_len=max_char_len)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return data_loader


def get_dataset(data_path, word2seq, char2seq, tag2seq, max_len=100, max_char_len=10):
    dataset = CoNLLDataset(data_path=data_path, word2seq=word2seq, char2seq=char2seq, tag2seq=tag2seq, max_len=max_len, max_char_len=max_char_len)
    return dataset


if __name__ == '__main__':
    data_path = "CoNLL-2003/eng.train"
    sentences = load_data(data_path=data_path)
    word2seq, seq2word = build_word2seq(sentences, min_freq=1)
    char2seq, seq2char = build_char2seq(sentences)
    tag2seq, seq2tag = build_tag2seq(sentences)
    dataset = CoNLLDataset(data_path=data_path, word2seq=word2seq, char2seq=char2seq, tag2seq=tag2seq, max_len=20)
    data_loader = get_loader(data_path=data_path, word2seq=word2seq, char2seq=char2seq, tag2seq=tag2seq, max_len=20, batch_size=3, shuffle=False)
    print(dataset[0])
    print("-" * 100)
    for index, data in enumerate(data_loader):
        sentences, chars, tags = data
        print(index)
        print(sentences)
        print(chars)
        print(tags)
        break
