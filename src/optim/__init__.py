from ._register import OPTIMIZERS, SCHEDULERS


def get_optimizer(model, cfg):
    params = model.parameters()
    optim_terms = cfg.optim.optimizer.copy()
    model_name = optim_terms.pop('name')
    return OPTIMIZERS.get_module(model_name)(params, **optim_terms)


def get_scheduler(optimizer, cfg):
    total_iter = cfg.total_epochs * (cfg.data.train_length // cfg.trainer.batch_size)
    cfg.trainer.total_iter = total_iter
    cfg.trainer.warmup_iter = 0

    if cfg.optim.scheduler is not None:
        scheduler_terms = cfg.optim.scheduler.copy()
        name = scheduler_terms.pop('name')

        if cfg.warmup_epochs is not None and isinstance(cfg.warmup_epochs, int):
            warmup_iter = cfg.warmup_epochs * (cfg.data.train_length // cfg.trainer.batch_size)
            cfg.trainer.warmup_iter = warmup_iter
            warmup_scheduler = SCHEDULERS.get_module('WarmUpLR')(optimizer, warmup_iter)
            if name == 'Cosine':
                scheduler = SCHEDULERS.get_module(name)(optimizer, T_max=int(total_iter - warmup_iter), **scheduler_terms)
            else:
                milestones = scheduler_terms.pop('milestones')
                milestones = [x*cfg.trainer.batch_size-warmup_iter for x in milestones]
                scheduler = SCHEDULERS.get_module(name)(optimizer, milestones=milestones, **scheduler_terms)
        else:
            warmup_scheduler = None
            if name == 'Cosine':
                scheduler = SCHEDULERS.get_module(name)(optimizer, T_max=total_iter, **scheduler_terms)
            else:
                milestones = scheduler_terms.pop('milestones')
                milestones = [x*cfg.trainer.batch_size for x in milestones]
                scheduler = SCHEDULERS.get_module(name)(optimizer, milestones=milestones, **scheduler_terms)
        return warmup_scheduler, scheduler
    else:
        return None, None
