import torch
from torchvision.transforms import Compose, Resize
from torchvision.models.detection.faster_rcnn import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
import PIL
from typing import List
import cv2


class ModelApp:
    """ Uses trained model to predict bounding boxes on given image and returns the most probable.
    """

    def __int__(self):
        self.__model = torch.load("fasterrcnn_complete_model.pth")
        self.__model.eval()

    def __transform_image(self, image: cv2.UMat) -> torch.Tensor:
        """Transforms given image so the model can process it

        Args:
            image (cv2.UMat): image to transform.

        Returns:
            Transformed image as tensor.
        """
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        transform = Compose([
            Resize((320, 320)),
            weights.transforms()
        ])
        image = transform(image)
        return image

    def predict_box(self, image: cv2.UMat) -> List[int] | None:
        """Computes most probable bounding box in given image.

        Args:
            image (cv2.UMat): image to compute bounding box for.

        Returns:
            Most probable bounding box or None if highest score is lower than 0.25.
        """
        image = self.__transform_image(image)
        preds = self.__model(image)
        most_probable = None
        if len(preds["boxes"]) > 0:
            if preds["scores"][0] > 0.25:
                most_probable = preds["boxes"][0]
        return most_probable
