import argparse
import ast
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import math
import random
import time
from glob import glob
from os.path import join
from typing import Union, List, Callable

import wandb
import yaml
from easydict import EasyDict
import plotly.graph_objects as go
import plotly.express as px
import imageio
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.stats import kstest, norm
from scipy.stats import probplot
import torch
import json
from torchvision.models import resnet50
from torch.multiprocessing import Process, set_start_method
from torchvision.transforms import Normalize
from DatasetCondensation.utils import ParamDiffAug, get_loops

CLIP_MODELS = {'16-B': 'openai/clip-vit-base-patch16',
               '32-B': 'openai/clip-vit-base-patch32',
               '14-336-L': 'openai/clip-vit-large-patch14-336',
               '14-L': 'openai/clip-vit-large-patch14'}

CLIP_DEFAULT_MODEL = '32-B'
CLIP_LATENTS_DIM = 512
DATASETS_CHOICES = ["MNIST", "FashionMNIST", "SVHN", "CIFAR10", "CIFAR100", "TinyImageNet"]


def process_args(latent_distillation=False):
    args = get_args(latent_distillation=latent_distillation)
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.dsa_param = EasyDict(vars(ParamDiffAug()))
    args.dsa = True if args.method == 'DSA' else False
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
    run_name = 'latent_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) if not args.exp_name else args.exp_name
    args.save_path = os.path.join(args.save_path, run_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    save_dict_as_json(args, os.path.join(args.save_path, 'args.json'))
    if args.use_wandb:
        wandb.init(project='latent-dataset-distillation', name=run_name, config=args)

    # logging config, logging to both a file and stdout
    file_handler = logging.FileHandler(filename=f"{args.save_path}/out.txt")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(v):
    return ast.literal_eval(v)


class DefaultsArgumentParser:

    def __init__(self, parser: argparse.ArgumentParser):
        self.parser = parser

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, action=ArgparseDefaultUsedAction, **kwargs)

    def parse_args(self, *args, **kwargs):
        return self.parser.parse_args(*args, **kwargs)


def get_args(parser=None, latent_distillation=False) -> EasyDict:
    """
    Get the arguments for the experiment
    :return: an EasyDict with the arguments
    """
    if not parser:
        parser = argparse.ArgumentParser(
            prog='latent dataset distillation',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = DefaultsArgumentParser(parser)
        parser.add_argument('--use_wandb', default='false', type=str2bool, help='whether to use wandb')
        parser.add_argument('--wandb_project', default='latent-dataset-distillation', type=str,
                            help='wandb project name')
        if latent_distillation:
            parser.add_argument('--clip_model', default='32-B', choices=list(CLIP_MODELS.keys()), type=str,
                                help='which network architecture to run')
            parser.add_argument('--latents_dim', default=CLIP_LATENTS_DIM, type=int, help='latents vectors dimension')
        parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
        parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset', choices=DATASETS_CHOICES)
        parser.add_argument('--model', type=str, default='MLP', help='model')
        parser.add_argument('--config', '--c', type=str, default='', help='path to a config json/yaml file')
        parser.add_argument('--exp_name', type=str, default='',
                            help='name of the experiment for saving dir name and for wandb')
        parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
        parser.add_argument('--eval_mode', type=str, default='S',
                            help='eval_mode')  # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
        parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
        parser.add_argument('--num_eval', type=int, default=20,
                            help='the number of evaluating randomly initialized models')
        parser.add_argument('--epoch_eval_train', type=int, default=300,
                            help='epochs to train a model with synthetic data')
        parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
        parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
        parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
        parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
        parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
        parser.add_argument('--init', type=str, default='noise',
                            help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
        parser.add_argument('--dsa_strategy', type=str, default='None',
                            help='differentiable Siamese augmentation strategy')
        parser.add_argument('--data_path', type=str, default='/home/shimon/research/datasets', help='dataset path')
        parser.add_argument('--save_path', type=str, default='/home/shimon/research/latent_distillation/result', help='path to save results')
        parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    args = parser.parse_args()
    if args.config:
        with open(args.config) as f:
            if args.config.endswith('.json'):
                config = json.load(f)
            elif args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                raise ValueError(f'config file must be either json or yaml. got {args.config}')
        for k, v in config.items():
            if not hasattr(args, f'{k}_nondefault'):
                # this means that k was not used as a command line argument, therefore we can override it
                setattr(args, k, v)

    # now we remove the redundant nondefault attributes from args
    args_clean = EasyDict({k: v for k, v in vars(args).items() if not k.endswith('_nondefault')})
    return args_clean


class ArgparseDefaultUsedAction(argparse.Action):
    """
    This class is used to allow the use of default values in the argparse module, while identifying when the user has
     actually passed the default value as an argument. it works by adding the parser an attribute called
     '<name>_nondefault', where <name> is the name of the argument. This attribute is set to True if the user has passed
      the default value as an argument, and False otherwise.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest + "_nondefault", True)
        setattr(namespace, self.dest, values)


def save_dict_as_json(save_dict, save_path):
    with open(save_path, 'w') as out_j:
        json.dump(save_dict, out_j, indent=4)


def create_horizontal_bar_plot(data: Union[str, dict], out_path, **kwargs) -> None:
    """
    Given a data dictionary (or path to json containing the data), create a horizontal bar plot and save it
    in out_path
    :param data: data dictionary (or subclass of it), or path to json containing the data
    :param out_path: path to save the plot
    :param kwargs: 'title' can be passed as a string with the title
    :return: None
    """
    if isinstance(data, str):
        with open(data, 'r') as in_j:
            data = json.load(in_j)
    elif not isinstance(data, dict):
        raise ValueError('data must be a dictionary or a path to a json file')
    # Figure Size
    fig, ax = plt.subplots(figsize=(16, 9))
    # Horizontal Bar Plot
    ax.barh(list(data.keys()), list(data.values()))
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)
    # Add x, y gridlines
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)

    # Show top values
    ax.invert_yaxis()
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                 str(round((i.get_width()), 2)),
                 fontsize=10, fontweight='bold',
                 color='grey')
    if 'title' in kwargs:
        # Add Plot Title
        ax.set_title(kwargs['title'], loc='left')
    plt.savefig(out_path)
    plt.close()


def get_num_params(model: torch.nn.Module) -> int:
    """
    Returns the number of trainable parameters in the given model
    :param model: torch model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_resnet_for_binary_cls(pretrained=True, all_layers_trainable=True, device=None, num_outputs=1) -> resnet50:
    """
    Loads a pretrained resenet50 apart from the last classification layer (fc) that is changed to a new untrained layer
    with output size 1
    :param num_outputs: number of outputs of the new last layer
    :param pretrained: whether to load the pretrained weights or not
    :param device: device to load the model to
    :param all_layers_trainable: if True, all layers are trainable, otherwise only the last layer is trainable
    :return:
    """
    model = resnet50(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = all_layers_trainable
    model.fc = torch.nn.Linear(2048, num_outputs)  # 2048 is the number of activations before the fc layer
    if device:
        model = model.to(device)
    return model


def get_resnet_50_normalization():
    """
    This is the normalization according to:
    https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
    :return:
    """
    return Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def images_to_gif(path: Union[str, List[str]], out_path, duration=300, **kwargs) -> None:
    """
    Given a path to a directory containing images, create a gif from the images using Pillow save function
    :param duration: duration in milliseconds for each image
    :param path: Either a str containing a path to a directory containing only images ,or list of paths to images
    :param out_path: path to save the gif
    :param kwargs: arguments to pass to the save function
    :return: None
    """
    if isinstance(path, str):
        path = glob(f"{path}/*")
    elif not isinstance(path, list):
        raise ValueError("path must be either a str or a list of str")
    images = [Image.open(p) for p in path]
    assert len(images) > 0, "No images found in directory"
    images[0].save(out_path, save_all=True, optimize=False, append_images=images[1:], loop=0,
                   duration=duration, **kwargs)


def forward_kl_univariate_gaussians(mu_p, sigma_p, mu_q, sigma_q):
    """
    Compute forward KL(P || Q) given parameters of univariate gaussian distributions, where P is the real distribution
    and Q is the approximated learned one.
    """
    return math.log(sigma_q / sigma_p) + ((sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2)) - 0.5


def reverse_kl_univariate_gaussians(mu_p, sigma_p, mu_q, sigma_q):
    """
    Compute forward KL(Q || P) given parameters of univariate gaussian distributions, where P is the real distribution
    and Q is the approximated learned one.
        """
    return math.log(sigma_p / sigma_q) + ((sigma_q ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_p ** 2)) - 0.5


def np_gaussian_pdf(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def data_parallel2normal_state_dict(dict_path: str, out_path: str):
    state_dict = torch.load(dict_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    torch.save(new_state_dict, out_path)


def set_all_seeds(seed=37):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def normality_test(samples: Union[torch.Tensor, np.ndarray], max_size=2000) -> float:
    """
    Returns the p-value of a Kolmogorov-Smirnov test on the given samples see (more info at
    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test and
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html )
    Intuitively, the p-value means the probability of obtaining test results at least as extreme as the result
    actually observed, meaning (hand wavy) bigger p -> greater probability for normal distribution, and vice-verse.
    This value can change drastically depending on the observations, and we usually reject the null hypothesis with a
    significance level <= 0.05 (p-value <= 0.05).
    If the distirbution is too large (SW test is not applicable in these cases), we return sample max_size samples
    from the distribution
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    elif not isinstance(samples, np.ndarray):
        raise ValueError("samples must be either torch.Tensor or np.ndarray")
    if samples.size > max_size:
        samples = np.random.choice(samples, max_size, replace=False)
    cdf_func = lambda x: norm.cdf(x, loc=np.mean(samples), scale=np.std(samples))
    return kstest(samples, cdf_func)[1]


def plot_qqplot(dist_samples: Union[torch.Tensor, np.ndarray], save_path: str):
    """
    Plots a qqplot of the given samples
    """
    if isinstance(dist_samples, torch.Tensor):
        dist_samples = dist_samples.detach().cpu().numpy()
    elif not isinstance(dist_samples, np.ndarray):
        raise ValueError("dist_samples must be either torch.Tensor or np.ndarray")
    probplot(dist_samples, dist="norm", plot=plt)
    plt.savefig(save_path)
    plt.close('all')


def images2video(images: Union[str, List[str], List[np.ndarray]], video_path: str, fps=5):
    already_numpy = False
    if isinstance(images, str):
        images = glob(join(images, "*"))
    elif isinstance(images, list):
        assert len(images) > 0, f"requires at least one image but got len(images) = {len(images)}"
        if isinstance(images[0], np.ndarray):
            already_numpy = True
    else:
        raise ValueError("images must be either str or list")
    writer = imageio.get_writer(video_path, fps=fps)
    for im in images:
        writer.append_data(im if already_numpy else imageio.imread(im))
    writer.close()


def set_fig_config(fig: go.Figure,
                   font_size=14,
                   width=500,
                   height=250,
                   margin_l=5,
                   margin_r=5,
                   margin_t=5,
                   margin_b=5,
                   font_family='Serif',
                   remove_background=False):
    if remove_background:
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(width=width, height=height,
                      font=dict(family=font_family, size=font_size),
                      margin_l=margin_l, margin_t=margin_t, margin_b=margin_b, margin_r=margin_r)
    return fig


def save_fig(fig, save_path):
    fig.write_image(save_path, width=1.5 * 300, height=0.75 * 300)


def plotly_init():
    figure = "placeholder_figure.pdf"
    debug_fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    debug_fig.write_image(figure, format="pdf")
    time.sleep(2)
    debug_fig.data = []


def multiprocess_func(func: Callable, args_list: List[tuple], add_devices=False):
    if add_devices:
        n_devices = torch.cuda.device_count()
        assert len(args_list) <= n_devices
        for i, cur_args in enumerate(args_list):
            assert isinstance(cur_args, tuple)
            args_list[i] = cur_args + (torch.device(f"cuda:{i}"),)

    set_start_method('spawn', force=True)
    procs = []
    for arg in args_list:
        p = Process(target=func, args=arg)
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
