import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
import random
import math
from datetime import timedelta
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf


def parse_args(opt):
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    random.seed(opt.random_seed)

    print(OmegaConf.to_yaml(opt))
    return opt


def get_model_and_optimizer(opt):
    model = FF_model(opt)
    if "cuda" in opt.device:
        model = model.cuda()
    print(model, "\n")

    main_model_params = [
        p
        for p in model.parameters()
        if all(p is not x for x in model.linear_classifier.parameters())
    ]
    optimizer = torch.optim.SGD(
        [
            {
                "params": main_model_params,
                "lr": opt.training.lr,
                "weight_decay": opt.training.wd,
                "momentum": opt.training.optimizer_momentum,
            },
            {
                "params": model.linear_classifier.parameters(),
                "lr": opt.training.downstream_lr,
                "weight_decay": opt.training.downstream_wd,
                "momentum": opt.training.optimizer_momentum,
            },
        ]
    )
    return model, optimizer


def get_data(opt, partition):
    dataset = FF_MNIST(opt, partition)

    g = torch.Generator()
    g.manual_seed(opt.random_seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.dataset.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
        persistent_workers=True,
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_MNIST_partition(opt, partition):
    if partition in ["train", "val", "train_val"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.dataset.path),
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    elif partition in ["test"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.dataset.path),
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    else:
        raise NotImplementedError

    if partition == "train":
        mnist = torch.utils.data.Subset(mnist, range(50000))
    elif partition == "val":
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.dataset.path),
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        mnist = torch.utils.data.Subset(mnist, range(50000, 60000))

    return mnist


def dict_to_cuda(dict):
    for key, value in dict.items():
        dict[key] = value.cuda(non_blocking=True)
    return dict


def preprocess_inputs(opt, inputs, labels):
    if "cuda" in opt.device:
        inputs = dict_to_cuda(inputs)
        labels = dict_to_cuda(labels)
    return inputs, labels


def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.training.num_epochs // 2):
        return lr * 2 * (1 + opt.training.num_epochs - epoch) / opt.training.num_epochs
    else:
        return lr


def update_learning_rate(optimizer, opt, epoch):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.lr
    )
    optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.downstream_lr
    )
    return optimizer


def get_accuracy(opt, output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.dataset.batch_size


def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="")
    print()


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict

# FF MNIST Code

class FF_MNIST(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        self.mnist = get_MNIST_partition(opt, partition)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.mnist)

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        )
        pos_sample = sample.clone()
        pos_sample[:, 0, : self.num_classes] = one_hot_label
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        classes = list(range(self.num_classes))
        classes.remove(class_label) 
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        )
        neg_sample = sample.clone()
        neg_sample[:, 0, : self.num_classes] = one_hot_label
        return neg_sample

    def _get_neutral_sample(self, z):
        z[:, 0, : self.num_classes] = self.uniform_label
        return z

    def _generate_sample(self, index):
        sample, class_label = self.mnist[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        return pos_sample, neg_sample, neutral_sample, class_label

## FF MODEL

class FF_model(torch.nn.Module):

    def __init__(self, opt):
        super(FF_model, self).__init__()

        self.opt = opt
        self.num_channels = [self.opt.network.hidden_size] * self.opt.network.num_layers
        self.act_fn = ReLU_full_grad()

        self.model = nn.ModuleList([nn.Linear(784, self.num_channels[0])])
        for i in range(1, len(self.num_channels)):
            self.model.append(nn.Linear(self.num_channels[i - 1], self.num_channels[i]))

        self.ff_loss = nn.BCEWithLogitsLoss()

        self.running_means = [
            torch.zeros(self.num_channels[i], device=self.opt.device) + 0.5
            for i in range(self.opt.network.num_layers)
        ]

        channels_for_classification_loss = sum(
            self.num_channels[-i] for i in range(self.opt.network.num_layers - 1)
        )
        self.linear_classifier = nn.Sequential(
            nn.Linear(channels_for_classification_loss, 10, bias=False)
        )
        self.classification_loss = nn.CrossEntropyLoss()

        self._init_weights()

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(
                    m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[0])
                )
                torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True)) + eps)

    def _calc_peer_normalization_loss(self, idx, z):
        mean_activity = torch.mean(z[:self.opt.dataset.batch_size], dim=0)

        self.running_means[idx] = self.running_means[
            idx
        ].detach() * self.opt.network.momentum + mean_activity * (
            1 - self.opt.network.momentum
        )

        peer_loss = (torch.mean(self.running_means[idx]) - self.running_means[idx]) ** 2
        return torch.mean(peer_loss)

    def _calc_ff_loss(self, z, labels):
        sum_of_squares = torch.sum(z ** 2, dim=-1)

        logits = sum_of_squares - z.shape[1]
        ff_loss = self.ff_loss(logits, labels.float())

        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels)
                / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy

    def forward(self, inputs, labels):
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "Peer Normalization": torch.zeros(1, device=self.opt.device),
        }

        z = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0)
        posneg_labels = torch.zeros(z.shape[0], device=self.opt.device)
        posneg_labels[: self.opt.dataset.batch_size] = 1

        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        for idx, layer in enumerate(self.model):
            z = layer(z)
            z = self.act_fn.apply(z)

            if self.opt.network.peer_norm_factor > 0:
                peer_loss = self._calc_peer_normalization_loss(idx, z)
                scalar_outputs["Peer Normalization"] += peer_loss
                scalar_outputs["Loss"] += self.opt.network.peer_norm_factor * peer_loss

            ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)
            scalar_outputs[f"loss_layer_{idx}"] = ff_loss
            scalar_outputs[f"ff_accuracy_layer_{idx}"] = ff_accuracy
            scalar_outputs["Loss"] += ff_loss
            z = z.detach()

            z = self._layer_norm(z)

        scalar_outputs = self.forward_downstream_classification_model(
            inputs, labels, scalar_outputs=scalar_outputs
        )

        return scalar_outputs

    def forward_downstream_classification_model(
        self, inputs, labels, scalar_outputs=None,
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        z = inputs["neutral_sample"]
        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        input_classification_model = []

        with torch.no_grad():
            for idx, layer in enumerate(self.model):
                z = layer(z)
                z = self.act_fn.apply(z)
                z = self._layer_norm(z)

                if idx >= 1:
                    input_classification_model.append(z)

        input_classification_model = torch.concat(input_classification_model, dim=-1)

        output = self.linear_classifier(input_classification_model.detach())
        output = output - torch.max(output, dim=-1, keepdim=True)[0]
        classification_loss = self.classification_loss(output, labels["class_labels"])
        classification_accuracy = get_accuracy(
            self.opt, output.data, labels["class_labels"]
        )

        scalar_outputs["Loss"] += classification_loss
        scalar_outputs["classification_loss"] = classification_loss
        scalar_outputs["classification_accuracy"] = classification_accuracy
        return scalar_outputs


class ReLU_full_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()