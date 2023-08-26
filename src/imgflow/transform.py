import numpy as np
from imgflow.workflow import ImgTransform
import cv2


class TransformBoxCut(ImgTransform):
    """
    Return new image with boundaries specified by YOLO's detected box.
    """
    def transform(self, X: np.array) -> np.array:
        boxes = self.source.box(self.index)
        return X[boxes[1]:boxes[3], boxes[0]:boxes[2], :]


class TransformInvert(ImgTransform):
    """
    Return new image flipping over both vertical and horizontal axes.
    """
    def transform(self, X: np.array) -> np.array:
        return X[::-1, ::-1]


class TransformRemoveBackground(ImgTransform):
    """
    Return new image with black color on region outside of the detected object.
    """
    def transform(self, X: np.array) -> np.array:
        return X * (self.source.mask(self.index) > 0)


class TransformRemoveBWBackgound(ImgTransform):
    def transform(self, X: np.array) -> np.array:
        tol = 1e-4
        mask_background = self.source.mask(self.index) < tol
        avg_bg = np.mean(X[mask_background].flatten())
        X[self.source.mask(self.index) < tol] = avg_bg
        return X


class TransformRemoveColorBackground(ImgTransform):
   def transform(self, X: np.array) -> np.array:
        tol = 1e-4
        mask_bg = self.source.mask(self.index) < tol
        X_r = X[:,:,0]
        X_g = X[:,:,1]
        X_b = X[:,:,2]
        mask_bg = mask_bg.squeeze()
        avg_r = X_r[mask_bg].flatten().mean()
        avg_g = X_g[mask_bg].flatten().mean()
        avg_b = X_b[mask_bg].flatten().mean()
        X_r[mask_bg] = avg_r
        X_g[mask_bg] = avg_g
        X_b[mask_bg] = avg_b
        X = np.array([X_r, X_g, X_b])
        return np.transpose(X, (1, 2, 0))


class TransformRemoveBackgroundAvg(ImgTransform):
    """
    Return new image replacing background with average sampled color near object's contour.
    """
    def transform(self, X: np.array) -> np.array:
        tol = 1e-4
        mask_bg = self.source.mask(self.index) < tol
        mask_bg = mask_bg.squeeze()
        avg_color = self.source.sample_color(self.index)
        n_channel = X.shape[-1]
        Xs = []
        for i_channel in range(n_channel):
            X[:, :, i_channel][mask_bg] = avg_color[i_channel]
            Xs.append(X[:,:,i_channel])
        X = np.array(Xs)
        return np.transpose(X, (1, 2, 0))


class TransformToBW(ImgTransform):
    """
    Return new image with grayscale.
    """
    def transform(self, X: np.array) -> np.array:
        return np.expand_dims(cv2.cvtColor(X, cv2.COLOR_BGR2GRAY), axis=2)


class TransformSquare(ImgTransform):
    """
    Return new image with 1:1 aspect ratio but do not skew the image. Use the averaged sampled colors to fill the background.
    """
    def transform(self, X: np.array) -> np.array:
        h, w, k = X.shape
        color = self.source.color_sample[self.index].reshape((1,1,3)).astype(np.uint8)
        if h>w:
            d = np.round((h-w)/2).astype('int')
            fill_left = np.tile(color, (h, d, 1))
            fill_right = np.tile(color, (h, h-d-w, 1))
            X = np.hstack((fill_left, X, fill_right))
        else:
            d = np.round((w-h)/2).astype('int')
            fill_top = np.tile(color, (d, w, 1))
            fill_bottom = np.tile(color, (w-d-h, w, 1))
            X = np.vstack((fill_top, X, fill_bottom))
        return X


class TransformResize(ImgTransform):
    """
    Scale image to the specified square of new size.
    """
    def __init__(self, src, i, new_size=100):
        super().__init__(src, i)
        self.new_size = new_size

    def transform(self, X: np.array) -> np.array:
        height0, width0, _ = X.shape
        return cv2.resize(X, (self.new_size, self.new_size), interpolation = cv2.INTER_AREA)