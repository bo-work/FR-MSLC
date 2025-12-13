# -*- coding: utf-8 -*-

import argparse
import os

import torch
import pandas as pd
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
import numpy as np
import pickle
from collections import Counter


from models.ETCModels.MLPNet.networks import MLPNet, VNet
from dataloader import CIFAR10, CIFAR100
from config_mlp import config, config_warmup
from utils import setup_realtime_logging, analyze_confusion_matrix


parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.2,
                    help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='sy',
                    help='Type of corruption ("sy" or "asy").')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--epochs_wp', default=80, type=int,
                    help='number of total epochs to warmup run')
parser.add_argument('--iters', default=60000, type=int,
                    help='number of total iters to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '--batch-size', default=100, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Resnet34', type=str,
                    help='name of experiment')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
parser.add_argument('--num_classes', type=int, default=0, help='number of label.')
parser.add_argument('--expriment_type', type=str, default='main', help='number of label.')
parser.add_argument('--meta_type', type=str, default='main', help='number of label.')
parser.set_defaults(augment=False)

args = parser.parse_args()
# # ========================
# # warmup phase
args.dataset = 'tls23'
args = config_warmup(args)
# args.dataset = 'ids17c'
# args = config_warmup(args)

# =======================


# 初始化实时日志
logger = setup_realtime_logging("script_output_MLP_pre.log")

# 用logger替代print输出
logger.info("脚本开始执行...")

logger.info(args)

use_cuda = True
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")


def TransformTwice(transform):
    if transform == 0:
        return 20
    elif transform == 1:
        return 21


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def to_tensor_var(x, requires_grad=True, type=torch.float32):
    x = torch.tensor(x, dtype=type)
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def build_dataset():
    logger.info('============ Build data')

    train_data_meta = CIFAR10(
        root=args.data_path, train=True, meta=True, epochs_wp=args.epochs_wp, corruption_prob=args.corruption_prob,
        corruption_type=args.corruption_type, num_classes=args.num_classes,
        meta_name=args.metadata_name, dataset=args.dataset)
    train_data = CIFAR10(
        root=args.data_path, train=True, meta=False, epochs_wp=args.epochs_wp, corruption_prob=args.corruption_prob,
        corruption_type=args.corruption_type, num_classes=args.num_classes, seed=args.seed,
        meta_name=args.metadata_name, dataset=args.dataset)
    test_data = CIFAR10(root=args.data_path, train=False, num_classes=args.num_classes, corruption_type=args.corruption_type,
                        meta_name=args.metadata_name, dataset=args.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    train_meta_loader = torch.utils.data.DataLoader(
        train_data_meta, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    # return train_loader, train_meta_loader, train_meta_aaec_loader, test_loader, test_aaec_loader
    return train_loader, train_meta_loader, test_loader


def build_model():
    # model = ResNet34(args.num_classes)
    model = MLPNet(input_shape=args.feature_size, num_classes=args.num_classes)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model



def save_model(save_path, model, optimizer, epoch, prediction):
    checkpoint = {
        'prediction': prediction,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, save_path)
    logger.info('Save model: Epoch: {}\n'.format(epoch))


def save_model_aaec(save_path, model, optimizer):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, save_path)


def load_model(save_path, model, optimizer):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()  # 设置模型为评估模式
    logger.info('load model finish')
    return checkpoint['epoch'], checkpoint['prediction']


def load_model_cluster(save_path):
    with open(save_path, 'rb') as f:
        model = pickle.load(f)
    logger.info('load model kmeans finish')
    return model


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epochs):
    lr = args.lr * (
                (0.1 ** int(epochs >= args.epochs_wp)) * (0.1 ** int(epochs >= args.epochs_wp + 20)))  # For WRN-28-10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




def custom_cross_entropy_with_sample_weight(inputs, targets, sample_weights=None, reduce=False , reduction='mean'):
    # 计算每个样本的交叉熵损失（不聚合）
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    # 应用样本权重
    weighted_loss = ce_loss * sample_weights
    # 按照指定方式聚合
    if not reduce:
        return weighted_loss  # 'none' 模式
    else:
        if reduction == 'mean':
            return weighted_loss.mean()
        elif reduction == 'sum':
            return weighted_loss.sum()
        elif reduction == 'none':
            return weighted_loss



def get_sample_weight(fhat_label, train_init_weight, train_cluster_label):
    return train_init_weight


def test(model, test_loader, fea_flag=False, type='model', train_loader = None):
    model.eval()
    correct = 0
    test_loss = 0
    predicteds = []
    targetss = []

    with torch.no_grad():
        for batch_idx, (inputs, fea, targets) in enumerate(test_loader):
            if fea_flag:
                inputs = fea
            inputs, targets = inputs.to(device), targets.to(device)
            if type == 'aaec':
                inputs = inputs.type(torch.float32)
                outputs = model(inputs, labels=targets)  # (bs, num_classes)
            else:
                outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            predicteds += predicted.tolist()
            targetss += targets.tolist()
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    logger.info(f'y_meta labels: {Counter(predicteds)}')

    y_gt, y_noi = np.array(targetss), np.array(predicteds)
    # y_gt, y_noi = ys['gty_meta'], ys['gty_meta']
    y_gt_name = []
    y_noi_name = []
    categories = args.categories

    for i, j in zip(y_gt, y_noi):
        y_gt_name.append(categories[i])
        y_noi_name.append(categories[j])
    y_gt_name = np.array(y_gt_name)
    y_noi_name = np.array(y_noi_name)

    df_cm = pd.crosstab(y_gt_name, y_noi_name,
                        rownames=['Actual'], colnames=['Predicted'],
                        dropna=False, margins=False, normalize='index').round(4)
    conf_matrix = confusion_matrix(y_gt, y_noi, normalize='true').round(4)
    logger.info(df_cm.to_dict())
    logger.info(conf_matrix.tolist())

    df_cm2 = confusion_matrix(y_gt, y_noi)
    w_accuracy, g_acc, accs = analyze_confusion_matrix(df_cm2)
    logger.info('Test set: Average acc: {:.4f}, Group few/mid/larg Accuracy: {:.4f}/{:.4f}/{:.4f}'.format(
        w_accuracy, g_acc[0], g_acc[1], g_acc[2]))
    print(accs)

    return accuracy, w_accuracy


def train_warmup(train_loader, test_loader, model, optimizer_model, epochs_model_wp, fea_flag):

    logger.info('\nWarmup Begin')

    # warmup train model
    best_test_acc = 0
    counter = 0
    global results_hat
    results_hat = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32)
    for epoch in range(epochs_model_wp):
        model.train()
        correct = 0
        correct_true = 0
        iteration_losses = []
        train_loss_wp = 0
        logger.info('\nWarmup model Epoch: %d' % epoch)

        # ce_weight = to_var(torch.tensor(args.ceweight, dtype=torch.float32), requires_grad=False)

        for batch_idx, ((inputs, _), inputs_fea, targets, targets_true, _, _, _, _, indexs) in enumerate(train_loader):
            input_var = to_var(inputs, requires_grad=False)
            target_var = to_var(targets, requires_grad=False).long()
            targets_true_var = to_var(targets_true, requires_grad=False).long()

            y_f = model(input_var)
            probs = F.softmax(y_f, dim=1)
            results_hat[indexs.cpu().detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()

            Loss = F.cross_entropy(y_f, target_var.long())
            # Loss = F.cross_entropy(y_f, target_var.long(), weight=ce_weight, reduce=True)
            iteration_losses.append(Loss.item())

            optimizer_model.zero_grad()
            Loss.backward()
            optimizer_model.step()
            # prec_train = accuracy(y_f.data, target_var.long().data, topk=(1,))[0]
            y_f = torch.max(y_f, dim=1)[1]
            correct += y_f.eq(target_var).sum().item()
            correct_true += y_f.eq(targets_true_var).sum().item()
            train_loss_wp += Loss.item()
            if (batch_idx + 1) % args.showiter == 0:
                logger.info('Warmup model Epoch: [%d/%d]\t'
                            'Iters: [%d/%d]\t'
                            'Loss: %.4f\t'
                            'Prec@1 %.2f\t' % (
                                epoch, epochs_model_wp, batch_idx + 1, len(train_loader.dataset) / args.batch_size,
                                (train_loss_wp / (batch_idx + 1)),
                                correct / (batch_idx + 1) / args.batch_size))

        train_loader.dataset.label_update(results_hat, epoch)
        logger.info(
            f"Epoch {epoch} | ACC {correct / len(train_loader.dataset)} | ACC_true {correct_true / len(train_loader.dataset)} | Loss {np.mean(iteration_losses)}")

        test_acc, test_w_acc = test(model=model, test_loader=test_loader)

        if best_test_acc < test_acc:
            best_test_acc = test_acc
            # save_model(args.warmup_dir_path+'/model_checkpoint_.pth', model, optimizer_model, epoch, train_loader.dataset.prediction)
            counter += 1
            logger.info(str(counter))
            save_model(args.warmup_dir_path + '/model_checkpoint_' + str(counter) + '.pth', model, optimizer_model,
                       epoch, train_loader.dataset.prediction)
            logger.info('Epoch: {}, Best Test Accuracy: ({:.4f}%)\n'.format(epoch, best_test_acc))
    logger.info('Warmup Training model End')

    return model

def train(train_loader, train_meta_loader, model, vnet1, vnet2, optimizer_model, optimizer_vnet1, optimizer_vnet2, train_kmeans_info_set, epoch, epoch_wp):
    logger.info('\nEpoch: %d' % epoch)

    train_loss = 0
    meta_loss = 0
    aaec_loss = [0, 0, 0, 0]

    correct = 0
    correct_o = 0
    prec_aaec_all = 0
    prec_aaec_meta = 0
    prec_aaec_meta_true = 0
    prec_model_all = 0
    prec_model_all_true = 0
    train_meta_loader_iter = iter(train_meta_loader)

    results_fhat = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32)
    results_fhat1 = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32)
    results_soft1 = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32)
    results_soft2 = np.zeros((len(train_loader.dataset)), dtype=np.float32)

    train_init_weight = train_kmeans_info_set['init_weight']
    train_kmeans_soft_labels = train_kmeans_info_set['train_soft_label']
    train_kmeans_pre_labels = train_kmeans_info_set['train_pre_label']
    train_cluster_label = train_kmeans_info_set['train_cluster_label']
    # train_fea_ex = train_kmeans_info_set['X_train_ex']

    train_sample_weight = get_sample_weight(train_loader.dataset.fhat_label_1, train_init_weight, train_cluster_label)


    for batch_idx, (
    (inputs, _), inputs_fea, targets, targets_true, fhat_labels, fhat_labels_1, soft_labels_1, soft_labels_2,
    indexs) in enumerate(train_loader):
        model.train()

        input_var = to_var(inputs, requires_grad=False)
        inputs_fea_var = to_var(inputs_fea, requires_grad=False)
        target_var = to_var(targets, requires_grad=False).long()
        targets_true_var = to_var(targets_true, requires_grad=False).long()
        fhat_labels_var = to_var(fhat_labels, requires_grad=False).long()
        fhat_labels_1_var = to_var(fhat_labels_1, requires_grad=False).long()
        soft_labels_1_var = to_var(soft_labels_1, requires_grad=False)
        soft_labels_2_var = to_var(soft_labels_2, requires_grad=False).long()

        soft_labels_kmeans_var = to_tensor_var(train_kmeans_soft_labels[indexs], requires_grad=False, type=torch.float32)
        pre_labels_kmeans_var = to_tensor_var(train_kmeans_pre_labels[indexs], requires_grad=False, type=torch.int64).long()
        input_sample_weight_var = to_tensor_var(train_sample_weight[indexs], requires_grad=False, type=torch.float32)


        meta_model = build_model()
        meta_model.cuda()

        meta_model.load_state_dict(model.state_dict())

        y_f_hat = meta_model(input_var)

        results_fhat1[indexs.cpu().detach().numpy().tolist()] = y_f_hat.cpu().detach().numpy().tolist()
        prec_train = accuracy(y_f_hat.data, target_var.data, topk=(1,))[0]
        prec_train_true = accuracy(y_f_hat.data, targets_true_var.data, topk=(1,))[0]

        # cal g   target_var:n    y_f_hat:n,c    fhat_label_2:n  soft_labels_1:n,c   soft_labels_2:n

        # cost1 = F.cross_entropy(y_f_hat, target_var, reduce=False)
        # cost1 = F.cross_entropy(y_f_hat, target_var, weight=ce_weight, reduce=False)
        cost1 = custom_cross_entropy_with_sample_weight(y_f_hat, target_var, sample_weights=input_sample_weight_var, reduce=False)
        cost1_v = torch.reshape(cost1, (len(cost1), 1))
        l_lambda = vnet1(cost1_v.data)
        l1 = torch.sum(cost1_v * l_lambda) / len(cost1_v)

        prec_aaec_all += targets_true_var.eq(pre_labels_kmeans_var).sum().item()
        # cost2 = F.cross_entropy(y_f_hat, pre_labels_kmeans_var, reduce=False)
        # cost2 = F.cross_entropy(y_f_hat, pre_labels_kmeans_var, weight=ce_weight, reduce=False)
        cost2 = custom_cross_entropy_with_sample_weight(y_f_hat, pre_labels_kmeans_var, sample_weights=input_sample_weight_var, reduce=False)
        cost2_v = torch.reshape(cost2, (len(cost2), 1))
        l_beta = vnet2(cost2_v.data)

        # y_f_hat_soft_m = (F.softmax(soft_labels_1_var, dim=1) + F.softmax(fhat_labels_1_var, dim=1) + F.softmax(y_f_hat_m, dim=1)) / 3
        y_f_hat_soft = (F.softmax(soft_labels_1_var, dim=1) + F.softmax(y_f_hat, dim=1)) / 2
        z = torch.max(y_f_hat_soft, dim=1)[1].long().cuda()
        # cost3 = F.cross_entropy(y_f_hat, z, weight=ce_weight, reduce=False)
        cost3 = custom_cross_entropy_with_sample_weight(y_f_hat, z, sample_weights=input_sample_weight_var, reduce=False)
        cost3_v = torch.reshape(cost3, (len(cost3), 1))

        l2 = torch.sum(cost2_v * (l_beta) * (1 - l_lambda)) / len(cost2_v) + torch.sum(
            cost3_v * (1 - l_beta) * (1 - l_lambda)) / (len(cost3_v))

        l_f_meta = l1 + l2

        y_g_wide2 = ((1 - l_beta) * y_f_hat_soft) + (l_beta *soft_labels_kmeans_var.float().cuda())
        target_var_nc = torch.zeros(inputs.size()[0], args.num_classes).scatter_(1, targets.view(-1, 1), 1)
        y_g_wide = y_g_wide2.cuda() * (1 - l_lambda.cuda()) + l_lambda.cuda() * target_var_nc.cuda()
        soft_labels = torch.max(y_g_wide, dim=1)[1].long().cuda()
        results_soft2[indexs.cpu().detach().numpy().tolist()] = soft_labels.cpu().detach().numpy().tolist()
        correct_o += targets_true_var.eq(soft_labels).sum().item()

        # fake update
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_lr = args.lr * ((0.1 ** int(epoch >= args.epochs_g_delay)))
        meta_model.update_params(lr_inner=meta_lr, source_params=grads)
        del grads

        # meta fake loss
        try:
            input_validation, _, target_validation, target_true_validation = next(train_meta_loader_iter)
        except StopIteration:
            train_meta_loader_iter = iter(train_meta_loader)
            input_validation, _, target_validation, target_true_validation = next(train_meta_loader_iter)
        input_validation_var = to_var(input_validation, requires_grad=False)
        target_validation_var = to_var(target_validation.type(torch.LongTensor), requires_grad=False)

        y_f_meta_hat = meta_model(input_validation_var)
        prec_meta = accuracy(y_f_meta_hat.data, target_validation_var.data, topk=(1,))[0]
        l_f_meta = F.cross_entropy(y_f_meta_hat, target_validation_var)
        # l_f_meta = F.cross_entropy(y_f_meta_hat, target_validation_var, weight=ce_weight)

        Loss_g = l_f_meta

        # corrector update
        optimizer_vnet1.zero_grad()
        optimizer_vnet2.zero_grad()
        Loss_g.backward()
        optimizer_vnet1.step()
        optimizer_vnet2.step()

        # cal real g  with model
        y_f_hat_m = model(input_var)

        # cost1_m = F.cross_entropy(y_f_hat_m, target_var, reduce=False)
        # cost1_m = F.cross_entropy(y_f_hat_m, target_var, weight=ce_weight, reduce=False)
        cost1_m = custom_cross_entropy_with_sample_weight(y_f_hat_m, target_var, sample_weights=input_sample_weight_var, reduce=False)
        cost1_m_v = torch.reshape(cost1_m, (len(cost1_m), 1))
        # cost2_m = F.cross_entropy(y_f_hat_m, y_f_hat_a, weight=ce_weight, reduce=False)
        cost2_m = custom_cross_entropy_with_sample_weight(y_f_hat_m, pre_labels_kmeans_var, sample_weights=input_sample_weight_var,
                                                          reduce=False)
        cost2_m_v = torch.reshape(cost2_m, (len(cost2_m), 1))
        # y_f_hat_soft_m = (F.softmax(soft_labels_1_var, dim=1) + F.softmax(y_f_hat, dim=1) + F.softmax(y_f_hat_m, dim=1)) / 3
        y_f_hat_soft_m = (F.softmax(soft_labels_1_var, dim=1) + F.softmax(y_f_hat_m, dim=1)) / 2
        z_m = torch.max(y_f_hat_soft_m, dim=1)[1].long().cuda()
        # cost3_m = F.cross_entropy(y_f_hat_m, z_m, weight=ce_weight, reduce=False)
        cost3_m = custom_cross_entropy_with_sample_weight(y_f_hat_m, z_m, sample_weights=input_sample_weight_var,
                                                          reduce=False)
        cost3_m_v = torch.reshape(cost3_m, (len(cost3_m), 1))

        with torch.no_grad():
            l_lambda_new = vnet1(cost1_m_v.data)
            l_beta_new = vnet2(cost2_m_v.data)
        y_g_wide2_new = (l_beta_new * soft_labels_kmeans_var.float().cuda()) + ((1 - l_beta_new) * y_f_hat_soft_m)
        y_g_wide_new = y_g_wide2_new.cuda() * (1 - l_lambda_new.cuda()) + l_lambda_new.cuda() * target_var_nc.cuda()
        results_soft1[indexs.cpu().detach().numpy().tolist()] = y_g_wide_new.cpu().detach().numpy().tolist()
        soft_labels_new = torch.max(y_g_wide_new, dim=1)[1].long().cuda()

        correct += targets_true_var.eq(soft_labels_new).sum().item()

        # real update
        l1_new = torch.sum(l_lambda_new * (cost1_m_v)) / len(cost1_m_v)
        l2_new = torch.sum((cost2_m_v) * (l_beta_new) * (1 - l_lambda_new)) / (
            len(cost2_m_v)) + torch.sum(
            (cost3_m_v) * (1 - l_beta_new) * (1 - l_lambda_new)) / (len(cost3_m_v))
        Loss_f = l1_new + l2_new

        optimizer_model.zero_grad()
        Loss_f.backward()
        optimizer_model.step()

        with torch.no_grad():
            y_f_hat_new = model(input_var)
        results_fhat[indexs.cpu().detach().numpy().tolist()] = y_f_hat_new.cpu().detach().numpy().tolist()

        y_f_hat_new_one = torch.max(y_f_hat_new, dim=1)[1].long().cuda()
        prec_model_all += target_var.eq(y_f_hat_new_one).sum().item()
        prec_model_all_true += targets_true_var.eq(y_f_hat_new_one).sum().item()

        train_loss += Loss_f.item()
        meta_loss += Loss_g.item()

        if (batch_idx + 1) % args.showiter == 0 or (batch_idx + 1) % 800 == 0:
            # if (batch_idx + 1) % 30 == 0 or (batch_idx + 1) % 2500 == 1500:
            logger.info('After Warmup Epoch: [%d/%d]\t'
                        'Iters: [%d/%d]\t'
                        'Loss: %.4f\t'
                        'MetaLoss:%.4f\t'
                        'Prec@1 %.2f\t'
                        'Prec_tilnow@1 %.4f\t'
                        'Prec_true@1 %.2f\t'
                        'Prec_true_tilnow@1 %.4f\t'
                        'Prec_cor_tillnow@1 %.4f\t'
                        'Prec_meta@1 %.2f' % (
                            (epoch), args.epochs, batch_idx + 1, len(train_loader.dataset) / args.batch_size,
                            (train_loss / (batch_idx + 1)),
                            (meta_loss / (batch_idx + 1)), prec_train, prec_model_all / (batch_idx + 1) / len(indexs),
                            prec_train_true, prec_model_all_true / (batch_idx + 1) / len(indexs),
                            correct / (batch_idx + 1) / len(indexs), prec_meta))
        # break

    logger.info(f'y_hat in traindata labels: {Counter(torch.max(torch.tensor(results_fhat), dim=1)[1].tolist())}')
    train_loader.dataset.label_update(results_fhat1, epoch, results_fhat, results_soft1, results_soft2)

    logger.info('Epoch: [%d/%d]\t'
                'Prec_aaec_o@1 %.4f\t'
                'Prec_aaec/true: [%d/%d/%d] %.2f, %.2f\t'
                'Prec_model_ture: [%d/%d] %.4f\t'
                'correct_o: [%d/%d] %.4f\t'
                'correct: [%d/%d] %.4f' % (
                    (epoch), args.epochs, prec_aaec_all / len(train_loader.dataset), prec_aaec_meta,
                    prec_aaec_meta_true,
                    len(train_meta_loader.dataset), prec_aaec_meta / len(train_meta_loader.dataset),
                    prec_aaec_meta_true / len(train_meta_loader.dataset), prec_model_all_true,
                    len(train_loader.dataset), prec_model_all_true / len(train_loader.dataset),
                    correct_o, len(train_loader.dataset), correct_o / len(train_loader.dataset), correct,
                    len(train_loader.dataset), correct / len(train_loader.dataset)))


def evaluate(results, train_loader, evaluator):
    model.eval()
    correct = 0
    test_loss = 0
    evaluator.reset()
    with torch.no_grad():
        # for batch_idx, (inputs, targets) in enumerate(test_loader):
        for batch_idx, ((inputs, inputs_u), targets, targets_true, soft_labels, indexs) in enumerate(train_loader):
            outputs = model(inputs)
            # pred = torch.max(outputs,dim=1)[1]#.cuda()
            evaluator.add_batch(targets_true, results)
    return evaluator.confusion_matrix


# train_loader, train_meta_loader, train_meta_aaec_loader, test_loader, test_aaec_loader = build_dataset()
train_loader, train_meta_loader, test_loader = build_dataset()


# create model
model = build_model()
logger.info('============ Build model MLP')

# # init DC strategy
# dc_max = max(0, int(len(train_loader.dataset) * args.dc_expend_ratio) - len(train_meta_loader.dataset))
# data_clean_stra = Data_Clean(train_loader.dataset, dc_max, strategy_type=args.dc_strategy_type, delay=args.dc_delay,
#                              replace=args.dc_replace)

vnet1 = VNet(1, 100, 1).cuda()
vnet2 = VNet(1, 100, 1).cuda()

optimizer_model = torch.optim.Adam(model.params(), args.lr,
                                  weight_decay=args.weight_decay)
optimizer_vnet1 = torch.optim.Adam(vnet1.params(), args.lr_vent,
                                   weight_decay=1e-4)
optimizer_vnet2 = torch.optim.Adam(vnet2.params(), args.lr_vent,
                                   weight_decay=1e-4)


def main():
    best_acc = 0
    best_w_acc = 0
    logger.info('============ Begin Train')

    if not args.ifload:
        train_warmup(train_loader, test_loader, model, optimizer_model, args.epochs_model_wp, args.fea_flag)
    else:
        warmup_epoch, prediction = load_model(args.model_results_path, model, optimizer_model)
        # model_kmeans = load_model_cluster(args.model_kmeans_results_path)
        train_loader.dataset.set_prediction(prediction, warmup_epoch, args.epochs_wp)
        train_kmeans_info_set = np.load(args.train_kmeans_info_results_path)

    test_acc, test_w_acc = test(model=model, test_loader=test_loader)
    for epoch in range(args.epochs_wp, args.epochs + 1):
        adjust_learning_rate(optimizer_model, epoch)

        train(train_loader, train_meta_loader, model, vnet1, vnet2, optimizer_model, optimizer_vnet1, optimizer_vnet2, train_kmeans_info_set, epoch, args.epochs_wp)
        test_acc, test_w_acc = test(model=model, test_loader=test_loader)
        if test_acc >= best_acc:
            best_acc = test_acc
        if test_w_acc >= best_w_acc:
            best_w_acc = test_w_acc
        logger.info(f'best accuracy: {best_acc}, best w_acc: {best_w_acc}')


if __name__ == '__main__':
    main()
