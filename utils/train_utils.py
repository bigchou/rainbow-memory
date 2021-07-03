import torch_optimizer
from easydict import EasyDict as edict
from torch import optim
import torch
import torch.nn as nn
import pretrainedmodels as ptm
from models.layers import ConvBlock, InitialBlock, FinalBlock
from models import mnist, cifar, imagenet


def select_optimizer(opt_name, lr, model, sched_name="cos"):
    if opt_name == "adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    elif opt_name == "radam":
        opt = torch_optimizer.RAdam(model.parameters(), lr=lr, weight_decay=0.00001)
    elif opt_name == "sgd":
        opt = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")

    if sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2, eta_min=lr * 0.01
        )
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            opt, gamma=0.1, milestones=[1000,10000]#[30, 60, 80, 90]
        )
    else:
        raise NotImplementedError(
            "Please select the sched_name [cos, anneal, multistep]"
        )

    return opt, scheduler

class PretrainedFrozenResNet50(torch.nn.Module):
    def __init__(self, opt, frozen=True):
        super(PretrainedFrozenResNet50, self).__init__()
        self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
        if frozen:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])
        self.embedding = nn.Linear(2048, opt.feature_size)#<--- embedding layer to control embedding size
        self.fc = FinalBlock(opt=opt, in_channels=opt.feature_size)

    def forward(self, x, feat=False):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)
        emb = self.embedding(x)
        if feat: return emb
        out = self.fc(emb)
        return out


def select_model(model_name, dataset, num_classes=None, pretrain=False):
    opt = edict(
        {
            "depth": 18,
            "num_classes": num_classes,
            "in_channels": 3,
            "bn": True,
            "normtype": "BatchNorm",
            "activetype": "ReLU",
            "pooltype": "MaxPool2d",
            "preact": False,
            "affine_bn": True,
            "bn_eps": 1e-6,
            "compression": 0.5,
            "feature_size": 128,
        }
    )
    if pretrain:
        print("!!!!! [Load ImageNet-pretrained model] !!!!!")
        model = PretrainedFrozenResNet50(edict({
            "num_classes":num_classes,
            "bn": True,
            "normtype": "BatchNorm",
            "activetype": "ReLU",
            "pooltype": "MaxPool2d",
            "preact": False,
            "affine_bn": True,
            "bn_eps": 1e-6,
            "compression": 0.5,
            "feature_size": 128,
        }))
        return model

    if "mnist" in dataset:
        model_class = getattr(mnist, "MLP")
    elif "cifar" in dataset:
        model_class = getattr(cifar, "ResNet")
    elif "imagenet" in dataset.lower():#to match TinyImagenet
        model_class = getattr(imagenet, "ResNet")
    else:
        raise NotImplementedError(
            "Please select the appropriate datasets (mnist, cifar10, cifar100, imagenet)"
        )

    if model_name == "resnet18":
        opt["depth"] = 18
    elif model_name == "resnet32":
        opt["depth"] = 32
    elif model_name == "resnet34":
        opt["depth"] = 34
    elif model_name == "mlp400":
        opt["width"] = 400
    else:
        raise NotImplementedError(
            "Please choose the model name in [resnet18, resnet32, resnet34]"
        )

    model = model_class(opt)

    return model
