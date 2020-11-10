import numpy as np


class GradientClipper:
    def __init__(self, clip_to_val=1.0):
        self.clip_to = clip_to_val

    def _clip(self, grads):
        return np.clip(grads, amin=-self.clip_to, amax=self.clip_to)

    def __call__(self, grads):
        return self._clip(grads)