import sys
import warnings
warnings.simplefilter("ignore", UserWarning)

from collections import defaultdict
import spacy
import torchvision.transforms as transform
import os
import PIL.Image as Image
import torch
from model import Pipeline as Pipeline
from unique_frames import extract_unique_frames
import argparse


spacy_eng = spacy.load("en_core_web_sm")

parser = argparse.ArgumentParser(description="Train the Model")
parser.add_argument('-a', '--annotation_file_path', type=str, required=True, metavar='', help='Path to annotation_file')
parser.add_argument('-e', '--pretrained_embeddings_path', type=str, required=True, metavar='', help='Path to pretrained embeddings')
parser.add_argument('-ml', '--pretrained_model_path', type=str, required=True, metavar='', help='Path to pretrained_model')
parser.add_argument('-v', '--video_path', type=str, metavar='', help='Path to video')
parser.add_argument('-im', '--image_folder_path', type=str, metavar='', help='Path to images')
parser.add_argument('-th', '--threshold', type=float, metavar='', help='Threshold for similarity')
args = parser.parse_args()

model_file = args.pretrained_model_path
video_path = args.video_path
annotation_file = args.annotation_file_path
pretrained_embed = args.pretrained_embeddings_path
image_folder_path = args.image_folder_path
threshold = args.threshold


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


class WordBank:
    def __init__(self, ann_file):
        self.annotations = defaultdict(list)
        self.ann_file = ann_file

        with open(self.ann_file, encoding="utf8") as fh:
            for idx, line in enumerate(fh):
                if idx > 0:
                    img_id, _, caption = line.strip().split("|")
                    self.annotations[img_id.split('.')[0]].append(caption)

        self.ids = list(sorted(self.annotations.keys()))
        self.vocab = Language()
        for key in self.annotations:
            self.vocab.add_sentence(self.annotations[key])


def lstm_inference(model_file, ann_file, pretrained_embed, threshold=0.1, image_folder=None, video_path=None, transform_test=None):

    dataset = WordBank(ann_file)

    embed_size = 300  # word embeddings dimension
    hidden_size = 1024
    vocab_size = dataset.vocab.n_words

    if image_folder is None and video_path is None:
        print("NO INPUT TO MODEL")
        sys.exit()

    if image_folder is not None and video_path is not None:
        image_folder = None

    if image_folder is None and video_path is not None:
        image_folder = os.path.join(os.getcwd(), "image_extract")
        if not os.path.exists(image_folder):
            os.mkdir(image_folder)
        extract_unique_frames(video_path, embed_size, hidden_size, threshold, path_out=image_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Pipeline(embed_size, hidden_size, vocab_size, num_layers=2, vocab=dataset.vocab.word2index,
                     embed_path=pretrained_embed).to(device=device)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint["model_state"])
    optimiser.load_state_dict(checkpoint["optimiser_state"])
    model.eval()

    if transform_test is None:
        transform_test = transform.Compose(
            [
                transform.Resize([256, 256]),
                transform.ToTensor(),
                transform.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    sentences = []
    for file in os.listdir(image_folder):
        img = transform_test(Image.open(os.path.join(image_folder, file)).convert("RGB")).unsqueeze(0)
        sentence = model.inference(img.to(device), dataset.vocab)
        sentences.append(sentence[1:-1])
        with open('report.txt', 'w') as f:
            for item in sentences:
                f.write("%s\n" % item)

        print("Prediction for", file, "-->", sentence)


lstm_inference(model_file=model_file, ann_file=annotation_file, pretrained_embed=pretrained_embed,
               image_folder=image_folder_path, video_path=video_path)
