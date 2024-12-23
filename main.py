import torch
from collections import defaultdict
import hydra
from omegaconf import DictConfig
from src import utils


import time


def train(opt, model, optimizer):
    start_time = time.time()
    model_start_time = start_time
    train_loader = utils.get_data(opt, "train")
    num_steps_per_epoch = len(train_loader)

    for epoch in range(opt.training.num_epochs):
        train_results = defaultdict(float)
        optimizer = utils.update_learning_rate(optimizer, opt, epoch)

        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            optimizer.zero_grad()

            scalar_outputs = model(inputs, labels)
            scalar_outputs["Loss"].backward()

            optimizer.step()

            train_results = utils.log_results(
                train_results, scalar_outputs, num_steps_per_epoch
            )

        utils.print_results("train", time.time() - start_time, train_results, epoch)
        start_time = time.time()

        # Validate.
        if epoch % opt.training.validation_index == 0 and opt.training.validation_index != -1:
            validate_or_test(opt, model, "val", epoch=epoch)

    print(f"Total time elapsed for Training on MNIST using Forward Forward Algorithm : {time.time() - model_start_time:.2f} seconds")
    return model


def validate_or_test(opt, model, partition, epoch=None):
    test_time = time.time()
    test_results = defaultdict(float)

    data_loader = utils.get_data(opt, partition)
    num_steps_per_epoch = len(data_loader)

    model.eval()
    print(partition)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            scalar_outputs = model.forward_downstream_classification_model(
                inputs, labels
            )
            test_results = utils.log_results(
                test_results, scalar_outputs, num_steps_per_epoch
            )

    utils.print_results(partition, time.time() - test_time, test_results, epoch=epoch)
    model.train()


@hydra.main(config_path=".", config_name="config", version_base=None)
def my_main(opt: DictConfig) -> None:
    opt = utils.parse_args(opt)
    model, optimizer = utils.get_model_and_optimizer(opt)
    model = train(opt, model, optimizer)
    validate_or_test(opt, model, "val")

    if opt.training.final_test:
        validate_or_test(opt, model, "test")


if __name__ == "__main__":
    my_main()