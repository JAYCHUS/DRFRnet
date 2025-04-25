def DualResNet_imagenet(cfg, pretrained=False):
    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=19, planes=32, spp_planes=128, head_planes=64,
                       augment=True)
    if pretrained:
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if
                            (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)

        model.load_state_dict(model_dict, strict=False)
    return model


def get_seg_model(cfg, **kwargs):
    model = DualResNet_imagenet(cfg, pretrained=True)
    return model


if __name__ == '__main__':
    x = torch.rand(4, 3, 800, 800)
    net = DualResNet_imagenet(pretrained=True)
    y = net(x)
    print(y.shape)