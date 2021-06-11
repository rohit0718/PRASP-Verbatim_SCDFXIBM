import torch
import torch.nn as nn
import torchvision.models as models
from load_embed import Embedding_reader

class CnnEncoder(nn.Module):
    def __init__(self, embed_size, hidden_size, train_CNN=False):
        super(CnnEncoder, self).__init__()
        self.train_CNN = train_CNN
        self.hidden_size = hidden_size
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.01)

        for name, param in self.resnet.named_parameters():
            if 'fc.weight' in name or 'fc.bias' in name:
                param.requires_grad = False
            else:
                param.requires_grad = self.train_CNN

    def forward(self, x):
        feature_out = self.dropout(self.relu(self.resnet(x)))

        return feature_out


class LstmDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, embed_path, vocab):
        super(LstmDecoder, self).__init__()

        self.embedding, embed_size = Embedding_reader.from_txt(embed_path, vocab)
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=0.3)
        self.fc = nn.Linear(hidden_size, len(vocab))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embedding(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hidden_state, _ = self.lstm(embeddings)
        outputs = self.fc(hidden_state)

        return outputs

class Pipeline(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, vocab, embed_path):
        super(Pipeline, self).__init__()
        self.CnnEncoder = CnnEncoder(embed_size, hidden_size)
        self.LstmDecoder = LstmDecoder(embed_size, hidden_size, vocab_size, num_layers, vocab=vocab, embed_path=embed_path)

    def forward(self, images, captions):
        features = self.CnnEncoder(images)
        outputs = self.LstmDecoder(features, captions)

        return outputs

    def inference(self, image, vocab, max_length=100):
        captions = []
        with torch.no_grad():
            x = self.CnnEncoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.LstmDecoder.lstm(x, states)
                output = self.LstmDecoder.fc(hiddens.squeeze(0))
                predicted = output.argmax(1)
                captions.append(predicted.item())
                x = self.LstmDecoder.embedding(predicted).unsqueeze(0)

                if vocab.index2word[predicted.item()] == '<EOS>':
                    break

        return [vocab.index2word[idx] for idx in captions]
