import os
import argparse
import torch
import torchvision
import numpy as np
from utils import yaml_config_hook
from modules import resnet, network, transform
from evaluation import evaluation
from torch.utils import data
import copy

def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    image_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c,h = model.forward_cluster(x)
        c = c.detach()
        h = h.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        # greyscale_image_batch = np.mean(x.cpu().detach().numpy(), axis=-3, keepdims=True)
        # image_vector.extend(greyscale_image_batch.reshape(500, 50176))
        image_vector.extend(h.cpu().detach().numpy())
        if step==6:
            feature_vector = np.array(feature_vector)
            labels_vector = np.array(labels_vector)
            images_vector = np.array(image_vector)
            break

    return feature_vector, labels_vector, images_vector


def plot_fig4(data, pred_labels, epoch):

    x_data = data
    pred_data = pred_labels
    # # PCA first
    # from sklearn import decomposition
    # pca = decomposition.PCA()
    # pca.n_components = 10
    # pca_data = pca.fit_transform(x_data)
    # TSNE afterward
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=2000, random_state=0, init='pca', learning_rate=200)
    tsne_results = tsne.fit_transform(x_data)

    # visualize TSNE
    x_axis = tsne_results[:, 0]
    y_axis = tsne_results[:, 1]
    plt.scatter(x_axis, y_axis, c=pred_data, cmap=plt.cm.get_cmap("jet", 100))

    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.title(f"TSNE Visualization {epoch} EPOCH")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=True,
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=False,
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    elif args.dataset == "STL-10":
        train_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="train",
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="test",
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "ImageNet-10":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-10',
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-dogs',
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/tiny-imagenet-200/train',
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        class_num = 200
    else:
        raise NotImplementedError
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=True,
        drop_last=False,
        num_workers=args.workers,
    )

    epochs = [0, 20, 100]
    for epoch in epochs:
        res = resnet.get_resnet(args.resnet)
        model = network.Network(res, args.feature_dim, class_num)
        model_fp = os.path.join("save/ImageNet-10/checkpoint_{}.tar".format(str(epoch)))
        model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
        model.to(device)

        print("### Creating features from model %s ###"%epoch)

        X, Y, IMG = inference(data_loader, model, device)
        plot_fig4(IMG, X, epoch)
