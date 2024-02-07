#from backbone.renset import ResNet
#from decoder import Decoder
#from head import Head
#from fpn import FPN
#from loss.utils import map2coords
import torch
from torch import nn
import torch.nn.functional as F



Norm = nn.BatchNorm2d


class Conv1x1(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.conv = nn.Conv2d(num_in, num_out, kernel_size=1, bias=False)
        self.norm = Norm(num_out)
        self.active = nn.ReLU(True)
        self.block = nn.Sequential(self.conv, self.norm, self.active)

    def forward(self, x):
        return self.block(x)


class Conv3x3(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.conv = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1,
                              bias=False)
        self.norm = Norm(num_out)
        self.active = nn.ReLU(True)
        self.block = nn.Sequential(self.conv, self.norm, self.active)

    def forward(self, x):
        return self.block(x)


class FPN(nn.Module):
    def __init__(self, inplanes, outplanes=512):
        super(FPN, self).__init__()

        self.laterals = nn.Sequential(*[Conv1x1(inplanes // (2 ** c), outplanes) for c in range(4)])
        self.smooths = nn.Sequential(*[Conv3x3(outplanes * c, outplanes * c) for c in range(1, 5)])
        self.pooling = nn.MaxPool2d(2)

    def forward(self, features):
        laterals = [l(features[f]) for f, l in enumerate(self.laterals)]

        map4 = laterals[3]
        map3 = laterals[2] + nn.functional.interpolate(map4, scale_factor=2,
                                                       mode="nearest")
        map2 = laterals[1] + nn.functional.interpolate(map3, scale_factor=2,
                                                       mode="nearest")
        map1 = laterals[0] + nn.functional.interpolate(map2, scale_factor=2,
                                                       mode="nearest")

        map1 = self.smooth[0](map1)
        map2 = self.smooth[1](torch.cat([map2, self.pooling(map1)], dim=1))
        map3 = self.smooth[2](torch.cat([map3, self.pooling(map2)], dim=1))
        map4 = self.smooth[3](torch.cat([map4, self.pooling(map3)], dim=1))
        return map4


class Head(nn.Module):
    def __init__(self, num_classes=80, channel=64):
        super(Head, self).__init__()

        self.cls_head = nn.Sequential(
            nn.Conv2d(256, channel,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes,
                      kernel_size=1, stride=1, padding=0))
        self.wh_head = self.ConvReluConv(256, 2)
        self.reg_head = self.ConvReluConv(256, 2)

    def ConvReluConv(self, in_channel, out_channel, bias_fill=False, bias_value=0):
        feat_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        relu = nn.ReLU()
        out_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        if bias_fill:
            out_conv.bias.data.fill_(bias_value)
        return nn.Sequential(feat_conv, relu, out_conv)

    def forward(self, x):
        hm = self.cls_head(x).sigmoid()
        wh = self.wh_head(x).relu()
        offset = self.reg_head(x)
        return hm, wh, offset



class Decoder(nn.Module):
    def __init__(self, inplanes, bn_momentum=0.1):
        super(Decoder, self).__init__()
        self.bn_momentum = bn_momentum
        # backbone output: [b, 2048, _h, _w]
        self.inplanes = inplanes
        self.deconv_with_bias = False
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 256, 256],
            num_kernels=[4, 4, 4],
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            padding = 0 if kernel == 2 else 1
            output_padding = 1 if kernel == 3 else 0
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)


def map2coords(h, w, stride):
    shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations

def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap


def convert_to_inplace_relu(model):
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = True

class ResNet(nn.Module):
    def __init__(self, slug='r50', pretrained=True):
        super().__init__()
        if not pretrained:
            print("Caution, not loading pretrained weights.")

        if slug == 'r18':
            self.resnet = models.resnet18(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif slug == 'r34':
            self.resnet = models.resnet34(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif slug == 'r50':
            self.resnet = models.resnet50(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'r101':
            self.resnet = models.resnet101(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'r152':
            self.resnet = models.resnet152(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'rx50':
            self.resnet = models.resnext50_32x4d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'rx101':
            self.resnet = models.resnext101_32x8d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'r50d':
            self.resnet = timm.create_model('gluon_resnet50_v1d',
                                            pretrained=pretrained)
            convert_to_inplace_relu(self.resnet)
            num_bottleneck_filters = 2048
        elif slug == 'r101d':
            self.resnet = timm.create_model('gluon_resnet101_v1d',
                                            pretrained=pretrained)
            convert_to_inplace_relu(self.resnet)
            num_bottleneck_filters = 2048

        else:
            assert False, "Bad slug: %s" % slug

        self.outplanes = num_bottleneck_filters

    def forward(self, x):
        size = x.size()
        assert size[-1] % 32 == 0 and size[-2] % 32 == 0, \
            "image resolution has to be divisible by 32 for resnet"

        enc0 = self.resnet.conv1(x)
        enc0 = self.resnet.bn1(enc0)
        enc0 = self.resnet.relu(enc0)
        enc0 = self.resnet.maxpool(enc0)

        enc1 = self.resnet.layer1(enc0)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)

        return enc1, enc2, enc3, enc4

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def freeze_stages(self, stage):
        if stage >= 0:
            self.resnet.bn1.eval()
            for m in [self.resnet.conv1, self.resnet.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, stage + 1):
            layer = getattr(self.resnet, 'layer{}'.format(i))
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False


class CenterNet(nn.Module):
    def __init__(self, cfg):
        super(CenterNet, self).__init__()
        self.backbone = ResNet(cfg.slug)
        if cfg.fpn:
            self.fpn = FPN(self.backbone.outplanes)
        self.upsample = Decoder(self.backbone.outplanes if not cfg.fpn else 2048, cfg.bn_momentum)
        self.head = Head(channel=cfg.head_channel, num_classes=cfg.num_classes)

        self._fpn = cfg.fpn
        self.down_stride = cfg.down_stride
        self.score_th = cfg.score_th
        self.CLASSES_NAME = cfg.CLASSES_NAME

    def forward(self, x):
        feats = self.backbone(x)
        if self._fpn:
            feat = self.fpn(feats)
        else:
            feat = feats[-1]
        return self.head(self.upsample(feat))

    @torch.no_grad()
    def inference(self, img, infos, topK=40, return_hm=False, th=None):
        feats = self.backbone(img)
        if self._fpn:
            feat = self.fpn(feats)
        else:
            feat = feats[-1]
        pred_hm, pred_wh, pred_offset = self.head(self.upsample(feat))

        _, _, h, w = img.shape
        b, c, output_h, output_w = pred_hm.shape
        pred_hm = self.pool_nms(pred_hm)
        scores, index, clses, ys, xs = self.topk_score(pred_hm, K=topK)

        reg = gather_feature(pred_offset, index, use_transform=True)
        reg = reg.reshape(b, topK, 2)
        xs = xs.view(b, topK, 1) + reg[:, :, 0:1]
        ys = ys.view(b, topK, 1) + reg[:, :, 1:2]

        wh = gather_feature(pred_wh, index, use_transform=True)
        wh = wh.reshape(b, topK, 2)

        clses = clses.reshape(b, topK, 1).float()
        scores = scores.reshape(b, topK, 1)

        half_w, half_h = wh[..., 0:1] / 2, wh[..., 1:2] / 2
        bboxes = torch.cat([xs - half_w, ys - half_h, xs + half_w, ys + half_h], dim=2)

        detects = []
        for batch in range(b):
            mask = scores[batch].gt(self.score_th if th is None else th)

            batch_boxes = bboxes[batch][mask.squeeze(-1), :]
            # batch_boxes[:, [0, 2]] *= infos[batch]['raw_width'] / output_w
            # batch_boxes[:, [1, 3]] *= infos[batch]['raw_height'] / output_h
            batch_boxes[:, [0, 2]] *= w / output_w
            batch_boxes[:, [1, 3]] *= h / output_h

            batch_scores = scores[batch][mask]

            batch_clses = clses[batch][mask]
            batch_clses = [self.CLASSES_NAME[int(cls.item())] for cls in batch_clses]

            detects.append([batch_boxes, batch_scores, batch_clses, pred_hm[batch] if return_hm else None])
        return detects

    def pool_nms(self, hm, pool_size=3):
        pad = (pool_size - 1) // 2
        hm_max = F.max_pool2d(hm, pool_size, stride=1, padding=pad)
        keep = (hm_max == hm).float()
        return hm * keep

    def topk_score(self, scores, K):
        batch, channel, height, width = scores.shape

        # get topk score and its index in every H x W(channel dim) feature map
        topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = (index / K).int()
        topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
        topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
        topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
