import torch
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pickle
np.random.seed(0)
torch.manual_seed(0)
# Allow torch/cudnn to optimize/analyze the input/output shape of convolutions
# To optimize forward/backward pass.
# This will increase model throughput for fixed input shape to the network
torch.backends.cudnn.benchmark = True

# Cudnn is not deterministic by default. Set this to True if you want
# to be sure to reproduce your results
torch.backends.cudnn.deterministic = True


def to_cuda(elements):
    """
    Transfers every object in elements to GPU VRAM if available.
    elements can be a object or list/tuple of objects
    """
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.cuda() for x in elements]
        return elements.cuda()
    return elements


def plot_loss(loss_dict: dict, label: str = None, fmt="-"):
    """
    Args:
        loss_dict: a dictionary where keys are the global step and values are the given loss / accuracy
        label: a string to use as label in plot legend
    """
    global_steps = list(loss_dict.keys())
    loss = list(loss_dict.values())
    plt.plot(global_steps, loss, fmt, label=label)


def save_checkpoint(state_dict: dict,
                    filepath: pathlib.Path,
                    is_best: bool,
                    max_keep: int = 1):
    """
    Saves state_dict to filepath. Deletes old checkpoints as time passes.
    If is_best is toggled, saves a checkpoint to best.ckpt
    """
    filepath.parent.mkdir(exist_ok=True, parents=True)
    list_path = filepath.parent.joinpath("latest_checkpoint")
    torch.save(state_dict, filepath)
    if is_best:
        torch.save(state_dict, filepath.parent.joinpath("best.ckpt"))
    previous_checkpoints = get_previous_checkpoints(filepath.parent)
    if filepath.name not in previous_checkpoints:
        previous_checkpoints = [filepath.name] + previous_checkpoints
    if len(previous_checkpoints) > max_keep:
        for ckpt in previous_checkpoints[max_keep:]:
            path = filepath.parent.joinpath(ckpt)
            if path.exists():
                path.unlink()
    previous_checkpoints = previous_checkpoints[:max_keep]
    with open(list_path, 'w') as fp:
        fp.write("\n".join(previous_checkpoints))


def save_training_statistics(statistic, filepath, filename):
    filepath.parent.mkdir(exist_ok=True, parents=True)
    list_path = filepath.parent.joinpath(filename)

    f = open(list_path, 'wb')
    pickle.dump(statistic, f)
    f.close()


def load_training_statistic(filepath, filename):
    list_path = filepath.parent.joinpath(filename)
    f = open(list_path, "rb")
    statistic = pickle.load(f)
    f.close()
    return statistic


def get_previous_checkpoints(directory: pathlib.Path) -> list:
    assert directory.is_dir()
    list_path = directory.joinpath("latest_checkpoint")
    list_path.touch(exist_ok=True)
    with open(list_path) as fp:
        ckpt_list = fp.readlines()
    return [_.strip() for _ in ckpt_list]


def load_best_checkpoint(directory: pathlib.Path):
    filepath = directory.joinpath("best.ckpt")
    if not filepath.is_file():
        return None
    if torch.cuda.is_available():
        return torch.load(directory.joinpath("best.ckpt"))
    else:
        return torch.load(directory.joinpath("best.ckpt"), map_location=torch.device('cpu'))


def compute_loss(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_criterion: torch.nn.modules.loss._Loss):
    """
    Computes the average loss and the accuracy over the whole dataset
    in dataloader.
    Args:
        dataloder: Validation/Test dataloader
        model: torch.nn.Module
        loss_criterion: The loss criterion, e.g: torch.nn.CrossEntropyLoss()
    Returns:
        [average_loss, accuracy]: both scalar.
    """
    total_loss = 0
    total_seps = 0
    with torch.no_grad():
        for X_batch in dataloader:
            # Transfer images/labels to GPU VRAM, if possible
            X_batch = to_cuda(X_batch)
            # Forward pass the images through our model
            output = model(X_batch)
            total_loss += loss_criterion(output, X_batch).item()
            total_seps += 1

    average_loss = total_loss/total_seps
    return average_loss
