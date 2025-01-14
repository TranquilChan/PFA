import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_image_label_list_from_npy(data_root, img_name_list, path='None'):
    if 'pascal' in path:
        npy_path = data_root + '/pascal_context_class.npy'
    else:
        npy_path = data_root + '/cls_labels.npy'
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


class ScribbleData(Dataset):
    def __init__(self, data_root=None, data_list=None, transform=None, path='ScribbleLabels'):
        self.indices = open('{}/ImageSets/{}'.format(data_root, data_list), 'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        image_path = self.img_names[index]
        label_path = self.lab_names[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = Image.open(label_path)  # GRAY 1 channel ndarray with shape H * W
        label = np.array(label)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label, image_path


class ScribbleClassData(ScribbleData):
    def __init__(self, data_root=None, data_list=None, transform=None, path='ScribbleLabels'):
        super().__init__(data_root=data_root, data_list=data_list, transform=transform, path=path)
        self.label_list = load_image_label_list_from_npy(data_root, self.indices, path)

    def __getitem__(self, index):
        image, mask_label, image_path = super().__getitem__(index)
        class_label = torch.from_numpy(self.label_list[index])
        return image, mask_label, class_label, image_path




