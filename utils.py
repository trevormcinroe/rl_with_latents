import torch
import numpy as np
from torchvision import transforms
import random

GPU = 'cuda:0'

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def generate_encoder_eps(encoder, env, n_eps):
    """

    Args:
        agent:
        encoder:
        n_eps:

    Returns:

    """
    n_collected = 0
    collected_eps = []

    actions = [1, 2, 3, 4, 5, 5, 5, 5]

    while n_collected < n_eps:
        inner_eps = []

        s = env.reset()
        s = np.moveaxis(s, -1, 0)
        # s = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)
        inner_eps.append(torch.tensor(s))
        # s = encoder(
        #     transforms.ToTensor()(s).unsqueeze(0).type(torch.float).to(GPU)
        # ).detach().cpu().numpy()[0]

        done = False

        while not done:
            a = np.random.choice(actions)
            s_, reward, picked_up, t, _ = env.step(a)
            s_ = np.moveaxis(s_, -1, 0)

            inner_eps.append(torch.tensor(s_))

            if t:
                done = True

            s = s_

        n_collected += 1
        collected_eps.append(inner_eps)
            # last_stored_frame_idx = agent.memory.store_frame(s)
            # obs = agent.memory.encode_recent_observation()
            # a = agent.choose_action(torch.tensor(obs).unsqueeze(0).to(GPU), concat=True, norm=True)
            # s_, r, t, _ = env.step(a.item())
            # # s_ = cv2.cvtColor(s_, cv2.COLOR_RGB2GRAY)
            # inner_eps.append(torch.tensor(s_).unsqueeze(0))
            # r = np.clip(r, -1.0, 1.0)
            # s_ = encoder(
            #     transforms.ToTensor()(s_).unsqueeze(0).type(torch.float).to(GPU)
            # ).detach().cpu().numpy()[0]
            #
            # agent.memory.store_effect(last_stored_frame_idx, a, r, t)
            #
            # if t:
            #     done = True
            #
            # s = s_

    # Shuffling for train/val split
    random.shuffle(collected_eps)
    return collected_eps[:int(n_eps * 0.8)], collected_eps[int(n_eps * 0.8):]
