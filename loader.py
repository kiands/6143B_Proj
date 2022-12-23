from torch.utils.data import Dataset
import torch.utils.data
import transformers
from config import *


class DatasetLoader(Dataset):

    @staticmethod
    def read_csv(path):
        texts = []
        labels = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                tag, text = line.strip().split(',', maxsplit=1)
                text = text.replace(' ', '')[1:-1].lower()
                if text:
                    labels.append(int(tag))
                    texts.append(text)
        return texts, labels

    def __init__(self, path):
        self.texts, self.labels = self.read_csv(path)

    def __getitem__(self, item):
        return self.texts[item], self.labels[item]

    def __len__(self):
        return len(self.texts)


if __name__ == '__main__':
    texts = []
    labels = []
    path = train_set
    dataset = DatasetLoader(train_set)
    print('Total', len(dataset))