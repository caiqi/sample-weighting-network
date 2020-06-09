from functools import partial

import mmcv
import numpy as np
from six.moves import map, zip
from torch.optim.optimizer import Optimizer


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret


class CombineOptimizer(Optimizer):
    def __init__(self, optimizers):
        # torch._C._log_api_usage_once("python.optimizer")
        self.optimizers = optimizers
        self.defaults = {k: v for d in optimizers for k, v in d.defaults.items()}
        self.state = {k: v for d in optimizers for k, v in d.state.items()}
        self.param_groups = [j for d in optimizers for j in d.param_groups]

    def __setstate__(self, state):
        super(CombineOptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for m in self.optimizers:
            m.step()
        return loss
