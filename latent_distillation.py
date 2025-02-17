import os
import time
import copy
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, \
    get_time, TensorDataset, epoch, DiffAugment
from typing import Tuple
import torch
from transformers import CLIPModel, CLIPProcessor
from ldd_utils import CLIP_DEFAULT_MODEL, CLIP_MODELS, process_args
from contextlib import nullcontext


def load_clip_and_process(model_name=CLIP_DEFAULT_MODEL, frozen=True) -> Tuple[CLIPModel, CLIPProcessor]:
    assert model_name in CLIP_MODELS.keys(), f"Invalid model name: {model_name}. poosible models: {CLIP_MODELS.keys()}"
    model_name = CLIP_MODELS[model_name]
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    return model, processor


def get_latents_optimizer(latents, args):
    optimizer = torch.optim.SGD([latents], lr=args.lr_img, momentum=0.5)

    return optimizer


def images_to_clip_features(images: torch.FloatTensor, processor: CLIPProcessor, model: CLIPModel,
                            with_grads=True, normalize=True, requires_grad=False,
                            device=None):
    if normalize:
        images = normalize_batch_before_clip(images, requires_grad=requires_grad)
    with torch.no_grad() if not with_grads else nullcontext():
        inputs = processor(images=images, return_tensors="pt", padding=True)
        if device is not None:
            inputs['pixel_values'] = inputs['pixel_values'].to(device)
        outputs = model.get_image_features(**inputs)
        return outputs


def normalize_batch_before_clip(batch, requires_grad=False):
    """
    Normalize a batch of images to [0,1] range
    Args:
        batch:  torch.Tensor
        requires_grad: whether to require gradients on the normalized batch
    Returns: a batch of normalized images, with gradients if requires_grad is True

    """
    # thanks to https://discuss.pytorch.org/t/how-to-efficiently-normalize-a-batch-of-tensor-to-0-1/65122/16
    # detaching to remove the normalization from the computation graph
    batch = batch.detach()
    device = batch.device
    batch_shape = batch.shape
    batch = batch.cpu().view(batch_shape[0], -1)
    batch -= batch.min(1, keepdim=True)[0]
    batch /= batch.max(1, keepdim=True)[0]
    batch = batch.view(batch_shape).to(device)
    if requires_grad:
        batch.requires_grad = True
    return batch


def main():
    args = process_args(latent_distillation=True)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_it_pool = np.arange(0, args.Iteration + 1,
                             min(500, args.Iteration // 2)).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [
        args.Iteration]  # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset,
                                                                                                         args.data_path)
    clip, processor = load_clip_and_process(args.clip_model)
    clip = clip.to(args.device)

    visited_first = False
    x, y = None, None
    for batch in tqdm(testloader):
        cur_x = images_to_clip_features(batch[0].to(args.device), processor, clip, with_grads=False, normalize=True,
                                        device=args.device)
        cur_y = batch[1]
        if not visited_first:
            x = cur_x
            y = cur_y
            visited_first = True
        else:
            x = torch.cat((x, cur_x), dim=0)
            y = torch.cat((y, cur_y), dim=0)
    testloader = DataLoader(TensorDataset(x, y), batch_size=testloader.batch_size, shuffle=False, num_workers=0)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    for exp in tqdm(range(args.num_exp), desc='Experiment number'):
        print('\n================== Exp %d ==================\n ' % exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            print('class c = %d: %d real images' % (c, len(indices_class[c])))

        def get_images(c, n):  # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f' % (
                ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

        ''' initialize the synthetic data latents '''
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float,
                                requires_grad=False, device=args.device)

        label_syn = torch.tensor([np.ones(args.ipc) * i for i in range(num_classes)], dtype=torch.long,
                                 requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')

        image_syn = images_to_clip_features(image_syn, processor, clip, with_grads=False, normalize=True,
                                            device=args.device)

        image_syn = image_syn.detach()
        image_syn.requires_grad = True
        ''' training '''
        optimizer_img = get_latents_optimizer(image_syn, args)
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins' % get_time())

        for it in tqdm(range(args.Iteration + 1), desc='global Iteration'):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in tqdm(model_eval_pool, desc='eval pool'):
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (
                        args.model, model_eval, it))
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval,
                                                        args.ipc)  # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        print('DC augmentation parameters: \n', args.dc_aug_param)

                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 300

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel=1, num_classes=num_classes, im_size=im_size,
                                               latents_size=args.latents_dim).to(
                            args.device)  # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                            label_syn.detach())  # avoid any unaware modification
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval,
                                                                 testloader, args)
                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                        len(accs), model_eval, np.mean(accs), np.std(accs)))

                    if it == args.Iteration:  # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'latents_%s_%s_%s_%dipc_exp%d_iter%d.pt' % (
                    args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                torch.save(image_syn_vis, save_name)

            ''' Train synthetic data '''
            net = get_network(args.model, channel=1, num_classes=num_classes, im_size=im_size,
                              latents_size=args.latents_dim).to(
                args.device)  # get a random model
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.

            for ol in tqdm(range(args.outer_loop), desc='outer loop'):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():  # BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train()  # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real)  # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  # BatchNorm
                            module.eval()  # fix mu and sigma of every BatchNorm layer

                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(args.device)
                for c in tqdm(range(num_classes), desc='num classes'):
                    img_real = get_images(c, args.batch_real)
                    img_real = images_to_clip_features(img_real, processor, clip, normalize=True, requires_grad=True,
                                                       device=args.device)

                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].view(
                        (args.ipc, -1))

                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss += match_loss(gw_syn, gw_real, args)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                if ol == args.outer_loop - 1:
                    break

                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                    label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True,
                                                          num_workers=0)
                for il in tqdm(range(args.inner_loop), desc='inner loop'):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug=True if args.dsa else False)

            loss_avg /= (num_classes * args.outer_loop)

            if it % 10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.Iteration:  # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path,
                                                                                               'res_%s_%s_%s_%dipc.pt' % (
                                                                                                   args.method,
                                                                                                   args.dataset,
                                                                                                   args.model,
                                                                                                   args.ipc)))

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%' % (
            args.num_exp, args.model, len(accs), key, np.mean(accs) * 100, np.std(accs) * 100))


if __name__ == '__main__':
    main()
