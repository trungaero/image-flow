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
        self.color_sample = {}
        self.index_distinct_objects = None

    def from_jpg(self, filename: str):
        """
            Call this to run object detection on image specified by file name. Update result attribute after prediction.
        Args:
            filename: path to image file

        Returns: None

        """
        self.result = self.model(filename, save_conf=True)[0]

    @staticmethod
    def len_intersect(b11: float, b12: float, b21: float, b22: float) -> float:
        """
            Calculate 1-d intersection length of two input lengths (b11,b12) and (b21,b22).

        Args:
            b11: coordinate of the first end-point of b1
            b12: coordinate of the second end-point of b1
            b21: coordinate of the first end-point of b2
            b22: coordinate of the second end-point of b2

        Returns:
            length of intersection
        """
        if (b12 < b21 or b22 < b11):
            return 0
        elif (b11 < b21 and b12 > b22):
            return b22 - b21
        elif (b21 < b11 and b22 > b12):
            return b12 - b11
        elif (b11 < b21):
            return b12 - b21
        else:
            return b22 - b11

    @staticmethod
    def area_box(b: np.array) -> float:
        """
            Calculate area of box b

        Args:
            b: np.array of shape (1 x 4)

        Returns:
            box area
        """
        return (b[2] - b[0]) * (b[3] - b[1])

    @staticmethod
    def has_overlap(b1: np.array, b2: np.array, threshold=0.8) -> bool:
        """
            Return true if two boxes has overlapping area greater than threshold (default = 0.8 of total mask area)
        Args:
            b1: coordinates of first box, np.array of shape 1 x 4
            b2: coordinates of second box, np.array of shape 1 x 4
            threshold: proportion of overlapping area / total area

        Returns:
            True / False
        """
        dx = Source.len_intersect(b1[0], b1[2], b2[0], b2[2])
        dy = Source.len_intersect(b1[1], b1[3], b2[1], b2[3])
        return dx * dy / (Source.area_box(b1) + Source.area_box(b2) - dx * dy) > threshold


    def check_distinct(self) -> None:
        """
           Update attribute index_distinct_objects which is a list containing indices of distinct objects.

        Returns: None

        """
        if self.get_num_detection() == 1:
            self.index_distinct_objects = [0]
            return
        elif self.get_num_detection() == 0:
            return

        box_indices = [_ for _ in range(self.get_num_detection())]
        b = [1 for _ in box_indices]
        from itertools import combinations

        # evaluate boxes in pairs
        for i, j in list(combinations(box_indices, 2)):
            # skip evaluation if either box was excluded
            if b[i]==0 or b[j]==0:
                continue

            b1 = self.box(i)
            b2 = self.box(j)

            # mark zero each time a box is ignored in a pair
            if Source.has_overlap(self.box(i), self.box(j)):
                if Source.area_box(b1) >= Source.area_box(b2):
                    b[j] = 0
                else:
                    b[i] = 0

        # exclude boxes being ignored in at least one evaluated pair
        self.index_distinct_objects = np.where(np.array(b)==1)


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

        Returns: np.array of size H x W

        """
        A = self.mask(i)
        B = np.array([A, np.roll(A,1, axis=0), np.roll(A, -1, axis=0), np.roll(A, 1, axis=1), np.roll(A, -1, axis=1)])
        C1 = np.min(B, axis=0)
        C2 = np.max(B, axis=0)
        D1 = C1==0
        D2 = C2>0
        E = (D1*D2>0)*1
        return E.squeeze()

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
        mask_inverted = mask_inverted.squeeze()
        x,y = np.where(contour==1)

        colors = []
        for j in tqdm(range(0,len(x),round(len(x)/n_sample))):
            patch = np.zeros(mask_inverted.shape)
            xmax, ymax = mask_inverted.shape
            patch[max(x[j]-d,0):min(xmax, x[j]+d), max(y[j]-d,0):min(y[j]+d,ymax)]=1
            patch_outside = patch * mask_inverted
            avg_color = (self.orig_img() * patch_outside[:,:,None]).sum(axis=(0,1))/patch_outside.sum()
            colors.append(avg_color)

        # store sampled background color
        self.color_sample[i] = np.array(colors).mean(axis=0)
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
