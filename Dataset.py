import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pathlib
from typing import Tuple, List


class FaceImageDataset(Dataset):
    """Dataset class for the WIDERFACE dataset.

    Attributes:
      data_dir (str): path to the directory with image directories in it.
      transform: transformation to perform on images.
    """

    def __init__(self, data_dir: str, ground_truth_file: str, transform=None) -> None:
        """Constructs Dataset class.

        Attributes:
          data_dir (str): path to the directory with image directories in it.
          transform: transformation to perform on images.
        """
        # Paths to the images
        self.__paths = list(map(lambda path: str(path), pathlib.Path(data_dir).glob("*/*.jpg")))
        self.__ground_truth_file = ground_truth_file
        self.__transform = transform
        self.__label_data = self.__get_label_data()

    def __get_label_data(self) -> dict:
        """Return dictionary with number of faces and bounding boxes for all images.
        """
        label_file = self.__get_label_file()
        label_data = {}
        index = 0
        index_image = 0
        while index < len(label_file):
          if len(label_file[index].split(".")) == 2:
            image_name = label_file[index].strip()
            index += 1
            num_faces = int(label_file[index].strip())
            boxes = []
            image = self.__load_image(index_image)
            index_image += 1
            original_height, original_width = image.shape[1:]
            for _ in range(num_faces):
              index += 1
              # only num_faces / x, y, width, height
              box_data = list(map(lambda x: int(x), label_file[index].split(" ")))[:4]
              if box_data[2] > 0 and box_data[3] > 0:
                x_0 = box_data[0] / original_width * 320
                x_1 = (box_data[0] + box_data[2]) / original_width * 320
                y_0 = box_data[1] / original_height * 320
                y_1 = (box_data[1] + box_data[3]) / original_height * 320
                box = [min(x_0, x_1), min(y_0, y_1), max(x_0, x_1), max(y_0, y_1)]
                boxes.append(box)
              else:
                num_faces -= 1
            label = {"labels": torch.tensor([1 for _ in range(num_faces)], dtype=torch.int64), "boxes": torch.tensor(boxes)}
            label_data[image_name] = label
          index += 1
        return label_data

    def __get_label_file(self) -> List[str]:
        """Returns list containing the lines of the ground truth file.
        """
        label_file_list = []
        with open(self.__ground_truth_file) as file:
          for line in file:
            label_file_list.append(line.strip())
        return label_file_list

    def __len__(self) -> int:
        """Returns the number of images.
        """
        return len(self.__paths)

    def __get_label(self, index: int) -> List:
        """Returns list with number of faces and bounding boxes for image at position index.

        Attributes:
          index (int): index of the image to get label data for.
        """
        image_path = self.__paths[index]
        path_parts = image_path.split("/")
        image_name = path_parts[-2] + "/" + path_parts[-1]
        return self.__label_data[image_name]

    def __load_image(self, index: int) -> Image.Image:
        """Return Image in Tensor form.

        Attributes:
          index (int): index of the image to load.
        """
        image_path = self.__paths[index]
        image = Image.open(image_path)
        tensor_transform = ToTensor()
        tensor_image = tensor_transform(image)
        return tensor_image

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List]:
        """Returns (transformed) image and label data for given index.

        Attributes:
            index (int): index for the item to get.
        """
        image = self.__load_image(index)
        label = self.__get_label(index)
        if not self.__transform:
            raise ValueError("No transform specified!")
        image = self.__transform(image)
        item = {"image": image, "targets": label}
        if len(label["boxes"]) == 0:
            return None
        else:
            return item
