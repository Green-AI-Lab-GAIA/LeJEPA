# run the TSNE of the whole dataset
from tqdm import tqdm
import torch
import numpy as np
from torchvision.datasets import CIFAR10

from transforms import common_transform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def evaluate(encoder):
    encoder.eval()

    # define the test dataset and dataloader
    dataset = CIFAR10(root='./data', train=False, transform=common_transform, download=True)
    dataloaders = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # calculate embeddings for the entire test set
    embeddings = []
    labels = []
    with torch.no_grad():
        for data in tqdm(dataloaders):
            views, batch_labels = data
            emb = encoder(views.to('cuda'))

            embeddings.append(emb.cpu().numpy())
            labels.append(batch_labels.numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Train a logistic regression classifier in 10% of the data and test on the rest
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.9, random_state=0)
    classifier = LogisticRegression(max_iter=1000, random_state=0)
    classifier.fit(X_train, y_train)

    correct_predictions = classifier.predict(X_test) == y_test
    accuracy = correct_predictions.sum() / len(y_test)

    encoder.train()
    return accuracy