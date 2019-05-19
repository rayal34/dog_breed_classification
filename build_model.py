import torch
import utils
import argparse
from torchvision import models, transforms, utils
from torchvision.datasets import ImageFolder
from torch import nn, optim
from settings import ModelDir
from torchcontrib.optim import SWA
from PIL import Image
from metrics import accuracy

DEVICE = torch.DEVICE('cuda' if torch.cuda.is_available() else 'cpu')


    def __init__(self,
                 base_dir,
                 batch_size=32):
        train_data = create_image_folder(base_dir, 'train')
        val_data = create_image_folder(base_dir, 'val')
        train_iterator = create_iterator(train_data)
        val_iterator = create_iterator(val_data, shuffle=False)


def create_iterator(data, shuffle=True, batch_size=32):
    return torch.utils.data.DataLoader(data,
                                       shuffle=shuffle,
                                       batch_size=batch_size)


def create_image_folder(base_dir, which):

    md = ModelDir(base_dir)
    generic_transforms = [transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])]
    if which == 'train':
        specific_transforms = [transforms.RandomHorizontalFlip(),
                               transforms.RandomRotation(10),
                               transforms.RandomCrop((224, 224),
                               pad_if_needed=True)]
        path = md.train
    elif which == 'val':
        specific_transforms = [transforms.CenterCrop((224, 224))]
        path = md.val
    else:
        raise ValueError("'which' can only be 'train' or 'val'")

    final_transforms = transforms.Compose(specific_transforms + generic_transforms)

    return ImageFolder(path, final_transforms)


def train_one_epoch(model, iterator, optimizer, criterion, scheduler=None):
    epoch_loss, epoch_acc = 0, 0

    model.train()
    for x, y in iterator:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, y)
        acc = accuracy(predictions, y)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def get_default_optimizer():
    base_opt = torch.optim.SGD(model.parameters(), momentum=.9, lr=1e-1)
    optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
    return optimizer


def get_default_criterion():
    return nn.CrossEntropyLoss()


def get_default_scheduler(optimizer):
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10)


def train(model, train_iterator, val_iterator, epochs, patience,
          optimizer=None, criterion=None, scheduler=None, fine_tune_layers=None):

    if optimizer is None:
        optimizer = get_default_optimizer()

    if criterion is None:
        criterion = get_default_criterion()

    if scheduler is None:
        scheduler = get_default_scheduler(optimizer)

    best_val_loss = float('inf')

    utils.make_folder(ModelDir.model_path)

    no_improve_cnt = 0
    fine_tune = (fine_tune_layers is not None)
    history = {'train_loss': [],
               'train_acc': [],
               'val_loss': [],
               'val_acc': []}

    for epoch in range(1, epochs + 1):

        train_loss, train_acc = train_one_epoch(model, train_iterator,
                                                optimizer, criterion,
                                                scheduler)

        val_loss, val_acc = evaluate(model, val_iterator, criterion)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # fine-tuning
        if fine_tune and train_loss < val_loss:
            print('train_loss < val_loss')
            fine_tune_init = True
            if fine_tune_init:
                model = set_layer_trainable(model, fine_tune_layers)
                no_improve_cnt = 0
                best_val_loss = float('inf')
                fine_tune_init = False

            # early stopping
            if no_improve_cnt >= patience:
                print(f'No improvements after {patience} epochs.')
                break

        val_loss, best_val_loss, no_improve_cnt = compare_val_loss(model,
                                                                   val_loss,
                                                                   best_val_loss,
                                                                   no_improve_cnt)

        print(f'| Epoch: {epoch:02} '
               '| Train Loss: {train_loss:.3f} '
               '| Train Acc: {train_acc * 100:05.2f}% '
               '| Val. Loss: {val_loss:.3f} '
               '| Val. Acc: {val_acc * 100:05.2f}% |')

    return history


def evaluate(model, iterator, criterion):
    epoch_loss, epoch_acc = 0, 0

    model.eval()
    for x, y in iterator:
        x, y = x.to(DEVICE), y.to(DEVICE)

        predictions = model(x)
        loss = criterion(predictions, y)
        acc = accuracy(predictions, y)
        loss.backward()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def count_parameters(model):

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params:,} trainable parameters')


def compare_val_loss(model, val_loss, best_val_loss, no_improve_cnt):

    if val_loss < best_val_loss:
        print(f'***********Loss improved from {best_val_loss:.3f} to {val_loss:.3f}***********')
        no_improve_cnt = 0
        best_val_loss = val_loss
        torch.save(model.state_dict(), ModelDir.model_path + MODEL_NAME)
    else:
        no_improve_cnt += 1

    return val_loss, best_val_loss, no_improve_cnt


def set_model_untrainable(model):

    for param in model.parameters():
        param.requires_grad = False

    return model


def set_layer_trainable(model, trainable_layers):

    for name, param in model.named_parameters():
        for layer in trainable_layers:
            if layer in name:
                param.requires_grad = True

    return model


def predict(model, filepaths, transforms):
    model.eval()

    batch_size = len(filepaths)
    imgs = torch.zeros(batch_size, 3, 224, 224).to(device)

    for i, file in enumerate(filepaths):
        img = Image.open(file)
        img = transforms(img)
        imgs[i] = img

    scores = torch.softmax(model(imgs), dim=1)
    idx_predicts = torch.argmax(scores, dim=1)
    class_predicts = [class_map[idx.item()] for idx in idx_predicts]
    class_scores = [scores[i][idx].item() for i, (score, idx) in enumerate(zip(scores, idx_predicts))]

    for filepath, predict in zip(filepaths, class_predicts):
        plt.imshow(Image.open(filepath))
        plt.axis('off')
        plt.show()
        print(predict)

    return class_predicts, class_scores

MODEL_NAME = 'resnet152.pt'

model = models.resnet152(pretrained=True).to(DEVICE)

model = set_model_untrainable(model)

model.fc = nn.Linear(model.fc.in_features, len(train_data.classes)).to(DEVICE)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default=50)
    args = parser.parse_args()
    base_dir = args.base_dir

    train()
