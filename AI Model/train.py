import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transform
from model import Pipeline
import os
from loader import custom_loader
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Train the Model")
parser.add_argument('-t', '--training_folder_path', type=str, required=True, metavar='', help='Path to training folder (Eg. Flickr30k)')
parser.add_argument('-v', '--validation_folder_path', type=str, required=True, metavar='', help='Path to validation/test folder')
parser.add_argument('-ml', '--pretrained_model', type=str, metavar='', help='Path to pretrained_model')
args = parser.parse_args()

# parameters
model_file = args.pretrained_model
training_folder = args.training_folder_path
validation_folder = args.validation_folder_path

warnings.simplefilter("ignore", UserWarning)


def save_checkpoint(epoch, model, optimiser, loss, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimiser_state': optimiser.state_dict(),
        'loss': loss,
    }

    torch.save(checkpoint, save_path)


def lstm_training(training_folder, test_folder, pretrained_model_path=None):
    transforms = transform.Compose(
        [
            transform.Resize([260, 260]),
            transform.RandomCrop([256, 256]),
            transform.ToTensor(),
            transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    dataset_folder = training_folder
    root_image_folder = os.path.join(dataset_folder, 'flickr30k_images')
    caption_file = os.path.join(dataset_folder, 'results.csv')

    data_loader, dataset = custom_loader(root_image_folder, caption_file, transforms, batch_size=32, shuffle=True)

    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device Using:", device)

    # Hyperparameters
    embed_size = 300  # word embeddings dimension
    hidden_size = 1024
    vocab_size = dataset.vocab.n_words
    num_layers = 2
    learning_rate = 0.003
    num_epochs = 30

    pretrained_file = os.path.join(os.getcwd(), 'glove_6B_300d', 'glove.6B.300d.txt')
    #pretrained_file = os.path.join(os.getcwd(), 'wiki-news-300d-1M', 'wiki-news-300d-1M.vec')
    model = Pipeline(embed_size, hidden_size, vocab_size, num_layers,
                     vocab=dataset.vocab, embed_path=pretrained_file).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.word2index["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if pretrained_model_path:
        print("Getting Pretrained model")
        load_model(pretrained_model_path, model, optimizer)
        # For a quick check of the loaded model
        print("Testing Model")
        test_folder = test_folder
        lstm_inference(model=model, device=device, dataset=dataset, image_folder=test_folder)

    model.train()

    for epoch in range(num_epochs):
        loss_list = []

        for idx, (imgs, captions) in enumerate(data_loader):

            imgs = imgs.to(device)
            captions = captions.to(device)
            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            if idx % 100 == 0:
                loss_list.append((idx, loss.item))
                print(idx, loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
            optimizer.step()

    save_path = os.path.join(os.getcwd(), "saved_model", f"model_epoch{epoch}.pth.tar")
    save_checkpoint(epoch, model, optimizer, loss_list, save_path)
    test_folder = test_folder  #'C:/Users/prama/Pictures/test_scdf1'

    lstm_inference(model=model, device=device, dataset=dataset, image_folder=test_folder)
    model.train()


def lstm_inference(model, device, image_folder, dataset, transform_test=None):
    model.eval()
    if transform_test is None:
        transform_test = transform.Compose(
            [
                transform.Resize([256, 256]),
                transform.ToTensor(),
                transform.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    for file in os.listdir(image_folder):
        img = transform_test(Image.open(os.path.join(image_folder, file)).convert("RGB")).unsqueeze(0)
        print("Prediction for", file, "-->", model.inference(img.to(device), dataset.vocab))


def load_model(model_file, model, optimiser):
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint["model_state"])
    optimiser.load_state_dict(checkpoint["optimiser_state"])


if __name__ == '__main__':
    lstm_training(training_folder, validation_folder, os.path.join(os.getcwd(),
                                                                   "captioner_resnet50_h1024_glove_300dim.pth.tar"))
