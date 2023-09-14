import random

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry import register_transform
from super_gradients.training.transforms.keypoint_transforms import KeypointTransform, PoseEstimationSample
from super_gradients.training.transforms.transforms import augment_hsv

logger = get_logger(__name__)


@register_transform()
class KeypointsHSV(KeypointTransform):
    """
    Apply color change in HSV color space to the input image.

    :attr prob:            Probability to apply the transform.
    :attr hgain:           Hue gain.
    :attr sgain:           Saturation gain.
    :attr vgain:           Value gain.
    """

    def __init__(self, prob: float, hgain: float, sgain: float, vgain: float):
        """

        :param prob:            Probability to apply the transform.
        :param hgain:           Hue gain.
        :param sgain:           Saturation gain.
        :param vgain:           Value gain.
        """
        super().__init__()
        self.prob = prob
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        if sample.image.shape[2] != 3:
            raise ValueError("HSV transform expects image with 3 channels, got: " + str(sample.image.shape[2]))

        if random.random() < self.prob:
            image_copy = sample.image.copy()
            augment_hsv(image_copy, self.hgain, self.sgain, self.vgain, bgr_channels=(0, 1, 2))
            sample.image = image_copy
        return sample
