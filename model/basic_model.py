import torch
import torch.nn as nn
from model import mix_tr
from model.seg_head import SegFormerHead
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from einops import rearrange


class MixTrModel(nn.Module):
    def __init__(self, args, backbone=None):
        super().__init__()
        self.num_classes = args.dataset.num_classes
        self.embedding_dim = args.model.decoder_embedding_dim
        self.stride = args.model.stride

        self.encoder = getattr(mix_tr, args.model.backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        if args.model.pretrained:
            state_dict = torch.load('./' + args.work_dir.pre_weight + '/' + args.model.backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        self.decoder = SegFormerHead(in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim,
                                     num_classes=self.num_classes)

        self.feature_aug = feature_aug(args=args)

    def get_param_groups(self):

        param_groups = [[], [], [], []]

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)

        for param in list(self.feature_aug.parameters()):
            param_groups[3].append(param)

        return param_groups

    def forward(self, x, prototypes=None, labels=None, is_global_prototype=False,
                is_infer=False):
        if prototypes is None:
            x, _ = self.encoder(x)
            x_feat, x_pred = self.decoder(x)
            return x_feat, x_pred
        else:
            feat_aug = self.feature_aug(x, prototypes, labels, is_global_prototype, is_infer)
            pred_aug = self.decoder(feat_aug, True)
            return feat_aug, pred_aug


class feature_aug(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.relu = nn.ReLU()
        self.prototype_fc = nn.Linear(args.prototype.dim, args.prototype.dim // 2)
        self.feature_fc = nn.Linear(args.prototype.dim, args.prototype.dim // 2)
        self.fc3 = nn.Linear(args.prototype.dim, args.prototype.dim // 2)
        self.fc4 = nn.Linear(args.prototype.dim // 2, args.prototype.dim)

    def forward(self, x, prototypes=None, labels=None, is_global_prototype=False, is_infer=False):
        if is_global_prototype:
            feat_aug = self.global_prototype_based_feature_aug(x, prototypes, labels, is_infer)
        else:
            feat_aug = self.local_prototype_based_feature_aug(x, prototypes, labels, is_infer)
        return feat_aug

    def local_prototype_based_feature_aug(self, img_features, prototypes, labels, is_infer):

        n, c, h, w = img_features.size(0), img_features.size(1), img_features.size(2), img_features.size(3)
        features = rearrange(img_features, "n c h w -> n (h w) c")
        features_agg = []
        for i in range(n):
            if is_infer:
                indices = labels[i]
            else:
                indices = torch.nonzero(labels[i]).squeeze()
            cur_prototype = prototypes[i][indices]
            if cur_prototype.dim() == 1:
                cur_prototype = torch.unsqueeze(cur_prototype, 0)
            feature = features[i]
            prototype_proj = self.prototype_fc(cur_prototype)
            feature_proj = self.feature_fc(feature)
            prototype_proj_ = prototype_proj.permute(1, 0)
            attention = torch.matmul(feature_proj, prototype_proj_)
            attention = F.softmax(attention, dim=1)
            prototype_aug = torch.matmul(attention, prototype_proj)
            feature_agg = self.relu(self.fc3(torch.cat([prototype_aug, feature_proj], dim=-1)))

            features_agg.append(feature_agg)
        features_agg_ = torch.stack(features_agg)
        features_agg_ = self.relu(features + self.fc4(features_agg_))
        features_agg_ = rearrange(features_agg_, "n (h w) c -> n c h w", h=h, w=w)

        return features_agg_

    def global_prototype_based_feature_aug(self, img_features, prototypes, labels, is_infer):

        n, c, h, w = img_features.size(0), img_features.size(1), img_features.size(2), img_features.size(3)
        features = rearrange(img_features, "n c h w -> n (h w) c")
        features_agg = []
        for i in range(n):
            if is_infer:
                indices = labels[i]
            else:
                indices = torch.nonzero(labels[i]).squeeze()
            cur_prototype = prototypes[indices].reshape(-1, c)
            feature = features[i]
            prototype_proj = self.prototype_fc(cur_prototype)
            feature_proj = self.feature_fc(feature)
            prototype_proj_ = prototype_proj.permute(1, 0)
            attention = torch.matmul(feature_proj, prototype_proj_)
            attention = F.softmax(attention, dim=1)
            prototype_aug = torch.matmul(attention, prototype_proj)
            feature_agg = self.relu(self.fc3(torch.cat([prototype_aug, feature_proj], dim=-1)))

            features_agg.append(feature_agg)
        features_agg_ = torch.stack(features_agg)
        features_agg_ = self.relu(features + self.fc4(features_agg_))
        features_agg_ = rearrange(features_agg_, "n (h w) c -> n c h w", h=h, w=w)

        return features_agg_
