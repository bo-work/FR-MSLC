import time
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from sklearn.cluster import KMeans
import csv
import math

from .LNL import Data_Process, wave_tran_4
from .model import MLPAE_for_DeepRe
from .plot import TSNE_plot
from .model_MoCo import MoCo

import pandas as pd
import os
import numpy as np
import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, silhouette_samples, confusion_matrix
from scipy.optimize import linear_sum_assignment

import warnings
warnings.filterwarnings("ignore")
#
# config = dict(algorithm='MCRe',
#               # dataset='Malicious_TLS',
#               # # data = "D:\\Jupyter\\UCL\\data\\CICIDS2017.csv",
#               # data="G:\\tls_features\\malicious_TLS_4_paper\\label_encodered_malicious_TLS.csv",
#               # # data = "D:\\Jupyter\\UCL\\data\\iot23--1444674--12classes.csv",
#               #
#               # savedir='./results',
#               dataset='IDS18',
#               datadir="/home/maybo/Documents/Project/Paper/etc-exmslc-py/dataset/ids18-0.1/noisedata/",
#               savedir='/home/maybo/Documents/Project/Paper/etc-exmslc-py/val-MCRE/MCRe-main/results/',
#               categories = ['Benign', 'Bot', 'BruteForce', 'DoS Hulk', 'HTTP DDoS', 'Infilteration'],
#
#
#               # noise_pattern='sy',  ##asy or sy
#               noise_pattern='ask',  ##asy or sy
#               # INCV_C_list=[0.5, 0.7, 0.3, 0, 0.8],  # [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
#               INCV_C_list=[0.1],  # [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
#               percent=0.7,
#               # seed = 1,
#
#               batch_size=256,
#               num_workers=1,
#               # epochs=200,
#               epochs=50,
#               adjust_lr=1,
#               learning_rate=1e-2,
#
#               embedding_size=128,
#               # start_clean_epoch = 100,
#               # epoch_contrast = 50,
#
#               moco_queue=8192,
#               moco_m=0.999,
#               temperature=0.1,
#               alpha=0.5,
#               pseudo_th=0.8,
#               proto_m=0.999,
#               lr=0.05,
#               cos=False,
#               schedule=[40, 80],
#               w_proto=1,
#               w_inst=1,
#               print_freq=300,
#
#               # num_class=23,  #
#               num_classes=6,  #
#               low_dim=16,
#               train_size=0,
#               val_size=0,
#               # input_dim=117,
#               input_dim=83,
#
#               )


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate based on schedule"""
    lr = config['lr']
    if config['cos']:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / config['epochs']))
    else:  # stepwise lr schedule
        for milestone in config['schedule']:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def acc(y_true, y_pred, num_cluster, categories):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size

    w = np.zeros((num_cluster, num_cluster))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    accuracy = 0.0
    for i in ind[0]:
        accuracy = accuracy + w[i, ind[1][i]]


    row_sums = w.sum(axis=1, keepdims=True)
    matrix_norm = w / np.where(row_sums != 0, row_sums, 1)
    # categories = ['Benign', 'BruteForce', 'DoS Hulk', 'Infilteration', 'HTTP DDoS', 'Bot']
    df = pd.DataFrame(matrix_norm)
    df.index = categories
    col = list(range(len(categories)))
    for i in range(len(categories)) :
        col[ind[1][i]] = categories[i]
    df.columns = col
    df = df[categories]
    print(df.to_dict())

    return accuracy / y_pred.size,  np.mean(np.diag(df.to_numpy()))


def save_model(save_path, model, optimizer, epoch):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, save_path)

def load_model(save_path, model, optimizer):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()  # 设置模型为评估模式
    print('load model finish')
    return checkpoint['epoch']


def Kmeans_model_evaluation_Discrete(model, dataloader, dataset='train'):
    if dataset == 'train':
        datasize = config['train_size']
    elif dataset == 'val':
        datasize = config['val_size']

    model.eval()
    datas = np.zeros([datasize, config['embedding_size']])
    label_true = np.zeros(datasize)
    ii = 0
    for i, (x, target, indexes) in enumerate(dataloader):
        x = x.reshape(config['batch_size'], 1, -1)
        x = Variable(x).cuda()
        target = Variable(target).cuda()

        _, u = model(x)
        u = u.cpu()
        datas[ii * config['batch_size']:(ii + 1) * config['batch_size'], :] = u.data.numpy()
        label_true[ii * config['batch_size']:(ii + 1) * config['batch_size']] = target.cpu().numpy().reshape((1, -1))
        ii = ii + 1

    kmeans = KMeans(n_clusters=config['num_classes'], random_state=0).fit(datas)
    centers = kmeans.cluster_centers_  ##
    print(len(kmeans.labels_))
    print(kmeans.labels_)

    label_pred = kmeans.labels_
    print(label_true)
    ACC = acc(label_true, label_pred, config['num_classes'])
    return ACC, centers


def Kmeans_model_evaluation(config, model, train_dataloader, val_dataloader):
    train_datasize = config['train_size']
    val_datasize = config['val_size']

    # model.eval()
    ##train_data
    train_datas = np.zeros([train_datasize, 32])  # config['embedding_size']
    train_label_observed = np.zeros(train_datasize)
    ii = 0
    for i, (x, target, indexes) in enumerate(train_dataloader):
        x = x.reshape(config['batch_size'], 1, -1)
        x = Variable(x).cuda()
        target = Variable(target).cuda()

        _, _, _, _, _, u = model(x, target, config)
        u = u.cpu()
        train_datas[ii * config['batch_size']:(ii + 1) * config['batch_size'], :] = u.data.numpy()
        train_label_observed[ii * config['batch_size']:(ii + 1) * config['batch_size']] = target.cpu().numpy().reshape(
            (1, -1))
        ii = ii + 1

    ##val_data
    val_datas = np.zeros([val_datasize, 32])  # config['embedding_size']
    val_label_true = np.zeros(val_datasize)
    ii = 0
    for i, (x, target, indexes) in enumerate(val_dataloader):
        x = x.reshape(config['batch_size'], 1, -1)
        x = Variable(x).cuda()
        target = Variable(target).cuda()

        _, _, _, _, _, u = model(x, target, config)
        u = u.cpu()
        val_datas[ii * config['batch_size']:(ii + 1) * config['batch_size'], :] = u.data.numpy()
        val_label_true[ii * config['batch_size']:(ii + 1) * config['batch_size']] = target.cpu().numpy().reshape(
            (1, -1))
        ii = ii + 1

    kmeans = KMeans(n_clusters=config['num_classes'], random_state=0).fit(train_datas)
    centers = kmeans.cluster_centers_  ##

    train_label_pred = kmeans.labels_

    train_ACC, train_w_ACC = acc(train_label_observed, train_label_pred, config['num_classes'], categories = list(config['categories_dict'].keys()))

    val_label_pred = kmeans.predict(val_datas)
    # kmeans_val = KMeans(n_clusters=config['num_classes'], random_state=0).fit(val_datas)
    # val_label_pred = kmeans_val.labels_
    val_ACC, val_w_ACC = acc(val_label_true, val_label_pred, config['num_classes'], categories = list(config['categories_dict'].keys()))

    return train_ACC, train_w_ACC, val_ACC, val_w_ACC, centers


def clusterloss(config, u, rho=1.0, reduction='none'):
    kmeans = KMeans(n_clusters=config['num_classes'], random_state=0).fit(u.cpu().detach())
    loss = silhouette_score(u.cpu().detach(), kmeans.labels_)

    return loss


def DeepRe(config, train_dataset, val_dataset, logger):
    cls_acc = []
    kmeans_acc = []
    best_val_acc = 0
    best_kmeans_val_acc = 0

    config['train_size'] = len(train_dataset)
    config['val_size'] = len(val_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config['batch_size'],
                                               num_workers=config['num_workers'],
                                               drop_last=True,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=config['batch_size'],
                                             num_workers=config['num_workers'],
                                             drop_last=True,
                                             shuffle=False)

    ##############################################################################

    logger.info('building model...')

    model = MoCo(MLPAE_for_DeepRe, config)
    model.cuda()

    criterion2 = nn.CrossEntropyLoss()  # .cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'],
                                momentum=0.9,
                                weight_decay=1e-4)

    load_epoch = 0
    if config['reload']:
        load_epoch = load_model(os.path.join(config['savedir'], config['reload_file']), model, optimizer)
        best_kmeans_val_acc = config['reload_acc']

    for epoch in range(1, config['epochs']):
        if epoch <= load_epoch:
            continue
        else:
            logger.info(f'epoch: {epoch}')

            adjust_learning_rate(optimizer, epoch, config)

            # train(train_loader, model, criterion, optimizer, epoch, args, logger)

            batch_time = AverageMeter('Time', ':1.2f')
            data_time = AverageMeter('Data', ':1.2f')
            acc_cls = AverageMeter('Acc@Cls', ':2.2f')
            acc_proto = AverageMeter('Acc@Proto', ':2.2f')
            # acc_inst = AverageMeter('Acc@Inst', ':2.2f')

            progress = ProgressMeter(
                len(train_loader),
                [batch_time, data_time, acc_cls, acc_proto],
                prefix="Epoch: [{}]".format(epoch))

            ##开始训练
            model.train()
            end = time.time()

            for i, (x, target_, indexes) in enumerate(tqdm.tqdm(train_loader)):
                x = x.reshape(config['batch_size'], 1, -1)
                x = Variable(x).cuda()
                target_ = Variable(target_).cuda()

                data_time.update(time.time() - end)

                loss = 0

                # compute model output
                cls_out, target, logits, x_q, logits_proto, u = \
                    model(x, target_, config, is_proto=(epoch > 0))

                loss_proto = criterion2(logits_proto, target.squeeze())
                acc = accuracy(logits_proto, target)[0]
                acc_proto.update(acc[0])

                loss_cls = criterion2(cls_out, target.squeeze())
                loss_clus = clusterloss(config, u)
                loss_clus = torch.Tensor([loss_clus]).cuda()
                loss_AE = nn.MSELoss()(x, x_q)
                x_1 = x.reshape(config['batch_size'], -1)
                loss = loss_cls + config['w_proto'] * loss_proto + loss_AE + loss_clus

                # log accuracy
                acc = accuracy(cls_out, target)[0]
                acc_cls.update(acc[0])

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()  #######
                optimizer.step()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # if i % config['print_freq'] == 0:
                # progress.display(i)


            with torch.no_grad():
                logger.info('==> Evaluation...')
                model.eval()
                top1_acc = AverageMeter("Top1")
                top5_acc = AverageMeter("Top5")

                predicteds = []
                targetss = []

                # evaluate on webvision val set
                for batch_idx, (x, target_, indexes) in enumerate(tqdm.tqdm(val_loader)):
                    x = x.reshape(config['batch_size'], 1, -1)
                    x = Variable(x).cuda()
                    target_ = Variable(target_).cuda()

                    outputs, _, target = model(x, target_, config, is_eval=True)
                    acc1 = accuracy(outputs, target)
                    top1_acc.update(acc1[0])
                    _, predicted = outputs.max(1)
                    predicteds += predicted.tolist()
                    targetss += target_.tolist()

                # average across all processes
                acc_tensors = torch.Tensor([top1_acc.avg]).cuda()

                logger.info('Accuracy is %.2f%%', acc_tensors[0].data.cpu().numpy())

                y_gt, y_noi = np.array(targetss), np.array(predicteds)
                y_gt_name = []
                y_noi_name = []
                # categories = ['Benign', 'BruteForce', 'DoS Hulk', 'Infilteration', 'HTTP DDoS', 'Bot']
                # categories = ['Benign', 'Bot', 'BruteForce', 'DoS Hulk', 'HTTP DDoS', 'Infilteration']
                categories = list(config['categories_dict'].keys())
                for i, j in zip(y_gt, y_noi):
                    y_gt_name.append(categories[i])
                    y_noi_name.append(categories[j])
                y_gt_name = np.array(y_gt_name)
                y_noi_name = np.array(y_noi_name)
                df_cm = pd.crosstab(y_gt_name, y_noi_name,
                                    rownames=['Actual'], colnames=['Predicted'],
                                    dropna=False, margins=False, normalize='index').round(4)
                conf_matrix = confusion_matrix(y_gt, y_noi, normalize='true').round(4)
                logger.info('firest val  ')
                logger.info(df_cm.to_dict())
                logger.info(conf_matrix.tolist())

                train_ACC, val_ACC, centers = Kmeans_model_evaluation(config, model=model, train_dataloader=train_loader,
                                                                      val_dataloader=val_loader)

                if val_ACC > best_kmeans_val_acc:
                    best_kmeans_val_acc = val_ACC
                    save_model(os.path.join(config['savedir'], 'model_mcre_checkpoint_'+config['noise_pattern']+str(config['noise_ratio']) + '_' + str(epoch) + 'ep.pth'), model, optimizer, epoch)
                    logger.info('Save mcre model: Epoch: {}\n'.format(epoch))
                # save_model(os.path.join(config['savedir'], 'model_mcre_checkpoint_' + config['noise_pattern'] + str(
                #     config['noise_ratio']) + '_' + str(epoch) + 'ep_.pth'), model, optimizer, epoch)
                # save_model(os.path.join(config['savedir'], 'model_mcre_checkpoint_' + config['noise_pattern'] + str(
                #     config['noise_ratio']) + '_lastep.pth'), model, optimizer, epoch)

            cls_acc.append(acc_tensors[0].data.cpu().numpy())
            kmeans_acc.append(val_ACC)

    f = open(config['savedir'] + 'record_acc_' + config['noise_pattern'] + '_' + str(config['noise_ratio']) + '_res.csv', 'a', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(cls_acc)
    csv_writer.writerow(kmeans_acc)
    f.close()



def DeepRe_Reload_Emb(config, train_dataset, logger):
    # batch_size = 4096
    batch_size = 512
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=config['num_workers'],
                                               drop_last=False,
                                               shuffle=False)

    logger.info('rebuilding model...')
    model = MoCo(MLPAE_for_DeepRe, config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'],
                                momentum=0.9,
                                weight_decay=1e-4)
    _ = load_model(os.path.join(config['savedir'], config['reload_file']), model, optimizer)
    model.cuda()

    with torch.no_grad():
        logger.info('==> Evaluation...')
        model.eval()
        ##train_data
        train_emb = np.zeros([len(train_dataset), 32])  # config['embedding_size']
        train_noi_label = np.zeros(len(train_dataset))
        train_gt_label = train_dataset.train_labels.detach().numpy()
        ii = 0
        for i, (x, target, indexes) in enumerate(train_loader):
            current_size = x.shape[0]
            if current_size < batch_size:
                # 需要补充的样本数量
                need = batch_size - current_size
                # 从现有样本中随机选择需要补充的样本（可替换为其他策略）
                fill_samples = [x[i % current_size] for i in range(need)]
                fill_samples = torch.stack(fill_samples)
                x = torch.cat((x, fill_samples), dim=0)

            x = x.reshape(batch_size, 1, -1)
            x = Variable(x).cuda()
            target = Variable(target).cuda()

            _, _, _, _, _, u = model(x, target, config)
            u = u.cpu()
            u = u[:current_size]
            train_emb[ii * batch_size:(ii + 1) * batch_size, :] = u.data.numpy()
            train_noi_label[ii * batch_size:(ii + 1) * batch_size] = target.cpu().numpy().reshape(
                (1, -1))
            ii = ii + 1

    return train_emb, train_gt_label, train_noi_label




    f = open(config['savedir'] + 'record_acc_' + config['noise_pattern'] + '_' + str(config['noise_ratio']) + '_res.csv', 'a', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(cls_acc)
    csv_writer.writerow(kmeans_acc)
    f.close()





def reload_test_DeepRe(config, train_dataset, test_dataset, logger):
    cls_acc = []
    kmeans_acc = []
    best_val_acc = 0
    best_kmeans_val_acc = 0

    config['train_size'] = len(train_dataset)
    config['val_size'] = len(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config['batch_size'],
                                               num_workers=config['num_workers'],
                                               drop_last=True,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=config['batch_size'],
                                             num_workers=config['num_workers'],
                                             drop_last=True,
                                             shuffle=False)

    ##############################################################################

    model = MoCo(MLPAE_for_DeepRe, config)
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'],
                                momentum=0.9,
                                weight_decay=1e-4)

    if config['reload']:
        load_epoch = load_model(os.path.join(config['savedir'], config['reload_file']), model, optimizer)

        train_ACC, train_w_ACC, val_ACC, val_w_ACC, _ = Kmeans_model_evaluation(config, model=model,
                                                              train_dataloader=train_loader,
                                                              val_dataloader=val_loader)
        print(f'train acc:{train_ACC}, train_w_acc : {train_w_ACC}')
        print(f'val acc:{val_ACC}, val_w_acc : {val_w_ACC}')

# for INCV_c in config['INCV_C_list']:
#     # INCV_c = 0.9
#     # INCV_b get label but we dont care
#
#     if config['noise_pattern'] == 'asy':
#         INCV_b = 0.1
#     else:
#         INCV_b = INCV_c
#
#     logger.info(f"noise_pattern:{config['noise_pattern']}")
#     print("INCV_c:", INCV_c)
#     DeepRe()