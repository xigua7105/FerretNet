import os
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from . import DATASETS, TV_TRANSFORMS


def get_trans(cfg):
    # train transforms
    if cfg.data.train_transforms is not None:
        trans_terms = cfg.data.train_transforms.copy()
        name = trans_terms.pop('name')
        train_trans = TV_TRANSFORMS.get_module(name)(**trans_terms)
    else:
        train_trans = None
    print(f"---\nTrain transforms\n{train_trans}\n---")

    # train target transforms
    if cfg.data.train_target_transforms is not None:
        trans_terms = cfg.data.train_target_transforms.copy()
        name = trans_terms.pop('name')
        train_target_trans = TV_TRANSFORMS.get_module(name)(**trans_terms)
    else:
        train_target_trans = None
    print(f"---\nTrain target transforms\n{train_target_trans}\n---")

    # test transforms
    if cfg.data.test_transforms is not None:
        trans_terms = cfg.data.test_transforms.copy()
        name = trans_terms.pop('name')
        test_trans = TV_TRANSFORMS.get_module(name)(**trans_terms)
    else:
        test_trans = None
    print(f"---\nTest transforms\n{test_trans}\n---")

    # test target transforms
    if cfg.data.test_target_transforms is not None:
        trans_terms = cfg.data.test_target_transforms.copy()
        name = trans_terms.pop('name')
        test_target_trans = TV_TRANSFORMS.get_module(name)(**trans_terms)
    else:
        test_target_trans = None
    print(f"---\nTest target transforms\n{test_target_trans}\n---")
    return train_trans, train_target_trans, test_trans, test_target_trans


def _get_datasets(cfg):
    assert cfg.data.dataset_type in DATASETS.dict.keys()
    dataset = DATASETS.get_module(cfg.data.dataset_type)
    train_transforms, train_target_transforms, test_transforms, test_target_transforms = get_trans(cfg)

    train_set = dataset(root=cfg.data.train_set_dir, cfg=cfg, is_train=True, transform=train_transforms, target_transform=train_target_transforms)
    test_set = dataset(root=cfg.data.test_set_dir, cfg=cfg, is_train=False, transform=test_transforms, target_transform=test_target_transforms)

    cfg.data.train_length = train_set.__len__()
    cfg.data.test_length = test_set.__len__()
    return train_set, test_set


def _get_dataloaders(cfg):
    train_set, test_set = _get_datasets(cfg)

    if cfg.dist:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        test_sampler = DistributedSampler(test_set, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(dataset=train_set,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              batch_size=cfg.trainer.batch_size_per_gpu,
                              num_workers=cfg.trainer.num_workers_per_gpu,
                              pin_memory=cfg.trainer.pin_memory,
                              drop_last=True)
    test_loader = DataLoader(dataset=test_set,
                             shuffle=False,
                             sampler=test_sampler,
                             batch_size=cfg.trainer.batch_size_per_gpu_test,
                             num_workers=cfg.trainer.num_workers_per_gpu,
                             pin_memory=cfg.trainer.pin_memory,
                             drop_last=False)
    return train_loader, test_loader


def _get_multi_datasets(cfg):
    """
    Dataset struct are as fellow:
        |---train
            |---TrainSet-A
            |--- ...
            |---TrainSet-N
        |---test
            |---TestSet-A
            |--- ...
            |---TestSet-N
    Return: TrainLoader: DataLoader, TestLoaders: dict
    """
    assert cfg.data.dataset_type in DATASETS.dict.keys()
    dataset = DATASETS.get_module(cfg.data.dataset_type)
    train_transforms, train_target_transforms, test_transforms, test_target_transforms = get_trans(cfg)

    # train data
    train_sets, train_lengths, train_length = [], {}, 0
    train_dir = cfg.data.train_set_dir
    for dir_name in os.listdir(train_dir):
        root = os.path.join(train_dir, dir_name)
        train_set = dataset(root=root, cfg=cfg, is_train=True, transform=train_transforms, target_transform=train_target_transforms)
        train_sets.append(train_set)
        train_lengths[dir_name] = train_set.__len__()
        train_length += train_set.__len__()
    train_set = ConcatDataset(train_sets)
    cfg.data.train_lengths = train_lengths
    cfg.data.train_length = train_length

    # test data
    test_dir = cfg.data.test_set_dir
    test_sets, test_lengths, test_length = {}, {}, 0
    for dir_name in os.listdir(test_dir):
        root = os.path.join(test_dir, dir_name)
        test_sets[dir_name] = dataset(root=root, cfg=cfg, is_train=False, transform=test_transforms, target_transform=test_target_transforms)
        test_lengths[dir_name] = test_sets[dir_name].__len__()
        test_length += test_sets[dir_name].__len__()
    cfg.data.test_lengths = test_lengths
    cfg.data.test_length = test_length
    return train_set, test_sets


def _get_multi_dataloaders(cfg):
    train_set, test_sets = _get_multi_datasets(cfg)

    if cfg.dist:
        sampler = DistributedSampler

        train_sampler = sampler(train_set, shuffle=True)
        test_samplers = dict()
        for k, test_set in test_sets.items():
            test_samplers[k] = sampler(test_set, shuffle=False)
    else:
        train_sampler = None
        test_samplers = dict()
        for k, v in test_sets.items():
            test_samplers[k] = None

    train_loader = DataLoader(dataset=train_set,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              batch_size=cfg.trainer.batch_size_per_gpu,
                              num_workers=cfg.trainer.num_workers_per_gpu,
                              pin_memory=cfg.trainer.pin_memory,
                              drop_last=True)
    test_loaders = dict()
    for k, test_set in test_sets.items():
        test_loaders[k] = DataLoader(dataset=test_set,
                                     shuffle=False,
                                     sampler=test_samplers[k],
                                     batch_size=cfg.trainer.batch_size_per_gpu_test,
                                     num_workers=cfg.trainer.num_workers_per_gpu,
                                     pin_memory=cfg.trainer.pin_memory,
                                     drop_last=False)
    return train_loader, test_loaders


def get_dataloader(cfg):
    if cfg.data.is_multi_loader:
        return _get_multi_dataloaders(cfg)
    return _get_dataloaders(cfg)


# Not Train, Only Test
def get_trans_test(cfg):
    if cfg.data.test_transforms is not None:
        trans_terms = cfg.data.test_transforms.copy()
        name = trans_terms.pop('name')
        input_transforms = TV_TRANSFORMS.get_module(name)(**trans_terms)
    else:
        input_transforms = None
    print(f"---\nTest input transforms\n{input_transforms}\n---")

    # test target transforms
    if cfg.data.test_target_transforms is not None:
        trans_terms = cfg.data.target_transforms.copy()
        name = trans_terms.pop('name')
        target_transforms = TV_TRANSFORMS.get_module(name)(**trans_terms)
    else:
        target_transforms = None
    print(f"---\nTest target transforms\n{target_transforms}\n---")
    return input_transforms, target_transforms


def _get_datasets_test(cfg):
    assert cfg.data.dataset_type in DATASETS.dict.keys()
    input_transforms, target_transforms = get_trans_test(cfg)
    dataset = DATASETS.get_module(cfg.data.dataset_type)
    test_set = dataset(root=cfg.data.test_set_dir, cfg=cfg, is_train=False, transform=input_transforms, target_transform=target_transforms)
    cfg.data.test_length = test_set.__len__()
    return test_set


def _get_multi_datasets_test(cfg):
    assert cfg.data.dataset_type in DATASETS.dict.keys()
    dataset = DATASETS.get_module(cfg.data.dataset_type)
    input_transforms, target_transforms = get_trans_test(cfg)

    # test data
    test_sets, test_lengths, test_length = {}, {}, 0
    for dir_name in os.listdir(cfg.data.test_set_dir):
        root = os.path.join(cfg.data.test_set_dir, dir_name)
        test_sets[dir_name] = dataset(root=root, cfg=cfg, is_train=False, transform=input_transforms, target_transform=target_transforms)
        test_lengths[dir_name] = test_sets[dir_name].__len__()
        test_length += test_sets[dir_name].__len__()
    cfg.data.test_lengths = test_lengths
    cfg.data.test_length = test_length
    return test_sets


def _get_dataloader_test(cfg):
    test_set = _get_datasets_test(cfg)

    if cfg.dist:
        test_sampler = DistributedSampler(test_set, shuffle=False)
    else:
        test_sampler = None

    test_loader = DataLoader(dataset=test_set,
                             shuffle=False,
                             sampler=test_sampler,
                             batch_size=cfg.tester.batch_size_per_gpu_test,
                             num_workers=cfg.tester.num_workers_per_gpu,
                             pin_memory=cfg.tester.pin_memory,
                             drop_last=False)
    return test_loader


def _get_multi_dataloader_test(cfg):
    test_sets = _get_multi_datasets_test(cfg)

    if cfg.dist:
        sampler = DistributedSampler
        test_samplers = dict()
        for k, test_set in test_sets.items():
            test_samplers[k] = sampler(test_set, shuffle=False)
    else:
        test_samplers = dict()
        for k, v in test_sets.items():
            test_samplers[k] = None

    test_loaders = dict()
    for k, test_set in test_sets.items():
        test_loaders[k] = DataLoader(dataset=test_set,
                                     shuffle=False,
                                     sampler=test_samplers[k],
                                     batch_size=cfg.tester.batch_size_per_gpu_test,
                                     num_workers=cfg.tester.num_workers_per_gpu,
                                     pin_memory=cfg.tester.pin_memory,
                                     drop_last=False)
    return test_loaders


def get_dataloader_test(cfg):
    if cfg.data.is_multi_loader:
        return _get_multi_dataloader_test(cfg)
    return _get_dataloader_test(cfg)
