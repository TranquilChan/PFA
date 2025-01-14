import torch.nn as nn
import torch
from .tools import *
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from .dataset import ScribbleClassData
import torch.nn.functional as F
import random


def data_loader(train, args):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    if train:
        train_transform = Compose([
            RandScale([0.5, 2.0]),
            RandRotate([-10, 10], padding=mean, ignore_label=args.dataset.ignore_label),
            RandomGaussianBlur(),
            RandomHorizontalFlip(),
            Crop([args.dataset.crop_size, args.dataset.crop_size],
                 crop_type='rand', padding=mean,
                 ignore_label=args.dataset.ignore_label),
            ToTensor(),
            Normalize(mean=mean, std=std)])
        train_dataset = ScribbleClassData(data_list=args.dataset.train_data_list,
                                          data_root=args.dataset.data_root,
                                          transform=train_transform,
                                          path=args.dataset.label_path)
        return train_dataset
    else:
        val_transform = Compose([
            Crop([args.dataset.crop_size, args.dataset.crop_size],
                 crop_type='center', padding=mean, ignore_label=args.dataset.ignore_label),
            ToTensor(),
            Normalize(mean=mean, std=std)])

        val_dataset = ScribbleClassData(data_list=args.dataset.val_data_list,
                                        data_root=args.dataset.data_root,
                                        transform=val_transform,
                                        path=args.dataset.label_path)
        return val_dataset


def get_loader(is_train, args):
    if is_train:
        return DataLoader(data_loader(train=is_train, args=args), num_workers=args.dataset.num_workers,
                          batch_size=args.dataset.batch_size,
                          shuffle=args.dataset.shuffle, pin_memory=args.dataset.pin_memory)
    else:
        return DataLoader(data_loader(train=is_train, args=args), num_workers=args.dataset.num_workers,
                          batch_size=args.dataset.batch_size,
                          shuffle=args.dataset.shuffle, pin_memory=args.dataset.pin_memory)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False


def check_values(array, num):
    for value in array:
        if value < num:
            return False
    return True


def extract_prototype(img_feat, img_pred, labels, infer=False):
    if infer:
        indices = labels
    else:
        indices = [torch.nonzero(row).flatten().tolist() for row in labels]
    c_feat = img_feat.shape[1]
    img_pred = F.relu(img_pred)
    n, c, h, w = img_pred.shape

    max_ = torch.max(img_pred.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1)
    min_ = torch.min(img_pred.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1)

    img_pred[img_pred < min_ + 1e-5] = 0
    norm_img_pred = (img_pred - min_ - 1e-5) / (max_ - min_ + 1e-5)

    img_pred = norm_img_pred

    img_feat = img_feat.permute(0, 2, 3, 1).reshape(n, -1, c_feat)

    img_pred = img_pred.reshape(n, c, -1)

    prototype = torch.zeros(n, c, c_feat)
    for i in range(n):
        top_values, top_indices = torch.topk(img_pred[i], k=h * w // 12, dim=-1)

        for j in indices[i]:
            top_feature = img_feat[i][top_indices[j]]
            prototype[i][j] = torch.sum(top_values[j].unsqueeze(-1) * top_feature, dim=0) / torch.sum(top_values[j])

    prototype = F.normalize(prototype, dim=-1)
    return prototype
