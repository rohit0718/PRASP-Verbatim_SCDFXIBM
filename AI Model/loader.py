import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import spacy
from collections import defaultdict

SOS_token = 0
EOS_token = 1

spacy_eng = spacy.load("en_core_web_sm")

class Language:
    def __init__(self):
        self.word2index = {"<SOS>": 0, "<EOS>": 1, "<PAD>": 2, "<UNK>": 3}   # stoi
        self.word2count = {}
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<PAD>", 3: "<UNK>"}  # itos
        self.n_words = 4

    def __len__(self):
        return len(self.index2word)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def add_sentence(self, sentence_list):
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if ' ' != word:
                    self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [self.word2index[token] if token in self.word2index else self.word2index["<UNK>"] for token in tokenized_text if token != ' ']


class FlickrDataset(Dataset):
    def __init__(self, root: str, ann_file: str, transform=None):
        self.ann_file = os.path.expanduser(ann_file)
        self.transform = transform
        self.root = root

        # Read annotations and store in a dict
        self.annotations = defaultdict(list)
        with open(self.ann_file, encoding="utf8") as fh:
            for idx, line in enumerate(fh):
                if idx > 0:
                    img_id, _, caption = line.strip().split("|")
                    self.annotations[img_id.split('.')[0]].append(caption)

        self.ids = list(sorted(self.annotations.keys()))
        self.vocab = Language()
        for key in self.annotations:
            self.vocab.add_sentence(self.annotations[key])

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index]

        # Captions
        captions = self.annotations[img_id]

        # Image
        filename = os.path.join(self.root, img_id + ".jpg")
        img = Image.open(filename).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        list_num_caption = []
        for caption in captions:
            numericalized_caption = [self.vocab.word2index["<SOS>"]]
            numericalized_caption += self.vocab.numericalize(caption)
            numericalized_caption.append(self.vocab.word2index["<EOS>"])
            list_num_caption.extend(numericalized_caption)

        return img, torch.tensor(list_num_caption)

    def __len__(self):
        return len(self.ids)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets

def custom_loader(root_folder, annotation_file, transform, batch_size, shuffle):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)

    pad_idx = dataset.vocab.word2index["<PAD>"]

    loader = DataLoader(dataset, batch_size, shuffle, collate_fn=MyCollate(pad_idx), num_workers=0, pin_memory=True)

    return loader, dataset
