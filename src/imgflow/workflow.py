from ultralytics import YOLO
from ultralytics.utils.ops import scale_image
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Type

YOLO_MODEL = 'yolov8s-seg.pt'


class ImgTransform(ABC):
    """
    Abstract image transformation step.
    """
    def __init__(self, src: Type['Source'], index:int):
        """
        Default constructor for a transformation object

        Args:
            src: source object with connection to the source image
            index: i-th detected in source image
        """
        self.source = src
        self.index = index

    @abstractmethod
    def transform(self, X: np.array) -> np.array:
        """
        Abstract transformation. Derived classes will implement the transformation logic.

        Args:
            X: image before the transformation step, np.array of size H (height) x W (width) x K (no of channels)

        Returns: image after the transformation step

        """
        pass


class Source:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        self.result = None
        self.color_sample = None

    def from_jpg(self, filename: str):
        """
            Call this to run object detection on image specified by file name. Update result attribute after prediction.
        Args:
            filename: path to image file

        Returns: None

        """
        self.result = self.model(filename)[0]

    def box(self, i:int) -> np.array:
        """
        Get the coordinates of i-th object's bounding box.

        Args:
            i: index of the object being detected from source image, starts from 0

        Returns: np.array of dimension 1 x 4

        """
        return self.result.boxes.xyxy.round().numpy().astype(int)[i]

    def get_num_detection(self) -> int:
        """
        Get the number of detected objects in source image

        Returns: number of detected objects

        """
        return self.result.boxes.xyxy.shape[0]

    def mask(self, i:int) -> np.array:
        """
        Get the mask matrix of same size as source image segmenting i-th object in source image.

        Args:
            i: i-th object

        Returns: np.array of size H x W x 1

        """
        return scale_image(self.result.masks.data.numpy()[i,:,:], self.result.orig_img.shape[0:-1])

    def orig_img(self) -> np.array:
        """
        Get the matrix representing the source image with 3 channels RGB

        Returns: np.array of size H x W x 3

        """
        return self.result.orig_img

    def contour(self, i:int) -> np.array:
        """
        Get the mask matrix of same size as source image where 1's represent contour of the object. Otherwise, default
        entries take value 0's

        Args:
            i: i-th object

        Returns: np.array of size H x W x 1

        """
        A = self.mask(i)
        B = np.array([A, np.roll(A,1, axis=0), np.roll(A, -1, axis=0), np.roll(A, 1, axis=1), np.roll(A, -1, axis=1)])
        C1 = np.min(B, axis=0)
        C2 = np.max(B, axis=0)
        D1 = C1==0
        D2 = C2>0
        return (D1*D2>0)*1

    def sample_color(self, i: int, d:int = 5, n_sample:int = 10):
        """
        Sample background color in the original image and update attribute color_sample. Specifying a square of length
        d with center lying on the contour then calculating average color in the part of region outside the object. The
        number of samples can be specified using parameter n_sample which generate points of equidistant along the
        contour.

        Args:
            i: i-th object
            d: square's half length
            n_sample: number of sampling points

        Returns: np.array of shape 1x3

        """
        contour = self.contour(i)
        mask_inverted = (self.mask(i)==0) * 1
        x,y,_ = np.where(contour==1)

        colors = []
        for i in tqdm(range(0,len(x),round(len(x)/n_sample))):
            patch = np.zeros(mask_inverted.shape)
            xmax, ymax, _ = mask_inverted.shape
            patch[max(x[i]-d,0):min(xmax, x[i]+d), max(y[i]-d,0):min(y[i]+d,ymax), 0]=1
            patch_outside = patch * mask_inverted
            avg_color = (self.orig_img() * patch_outside).sum(axis=(0,1))/patch_outside.sum()
            colors.append(avg_color)

            # store sampled background color
            self.color_sample = np.array(colors).mean(axis=0)
        return np.array(colors).mean(axis=0)

class Workflow:
    def __init__(self):
        """
        Call the constructor to define the new workflow.
        """
        self.source = Source()
        self.steps = []

    def add_transform_step(self, t: ImgTransform, **kwargs):
        """
        Add transform object which defines a transform step. Once calling the run method, steps will be executed in the
        order of being added.

        Args:
            t: the next transformation step
            **kwargs:

        Returns: None

        """
        self.steps.append((t, kwargs))

    def run(self):
        """
        Invoke this to run the work flow. Final result is a list of array each represents one final image.

        Returns: list of final images
        """
        K = self.source.get_num_detection()
        X = self.source.orig_img()
        Xs = [X for _ in range(K)]
        for step, kwargs in self.steps:
            for i in range(K):
                Xs[i] = step(self.source, i, **kwargs).transform(Xs[i])
        return Xs
