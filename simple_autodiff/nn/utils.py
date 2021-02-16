import numpy as np


class GradientClipper:
    def __init__(self, clip_to_val=1.0):
        self.clip_to = clip_to_val

    def _clip(self, grad):
        return np.clip(grad, amin=-self.clip_to, amax=self.clip_to)

    def __call__(self, grad):
        return self._clip(grad)