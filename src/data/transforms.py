from ._register import TV_TRANSFORMS


@ TV_TRANSFORMS.register_module
def define_transforms(kwargs: list):
    transforms_list = []
    for t in kwargs:
        t = {k: v for k, v in t.items()}
        t_type = t.pop('type')
        transforms_list.append(TV_TRANSFORMS.get_module(t_type)(**t))
    return TV_TRANSFORMS.get_module("Compose")(transforms_list)


@ TV_TRANSFORMS.register_module
def syn_train_trans(**kwargs):
    cfg_train = [
        dict(type="RandomCrop", size=(224, 224), pad_if_needed=True),
        dict(type="RandomHorizontalFlip"),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711), inplace=True)
    ]
    return TV_TRANSFORMS.get_module("define_transforms")(cfg_train)


@ TV_TRANSFORMS.register_module
def syn_test_trans(**kwargs):
    cfg_test = [
        dict(type="CenterCrop", size=(256, 256), pad_if_needed=True),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711), inplace=True)
    ]
    return TV_TRANSFORMS.get_module("define_transforms")(cfg_test)

