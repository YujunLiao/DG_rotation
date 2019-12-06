import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random



class JigsawDataset(data.Dataset):
    def __init__(self, data_paths, labels, jig_classes=100, img_transformer=None, tile_transformer=None, is_patch_based_or_not=True, percent_of_original_image=None):
        self._data_path = ""
        self._data_paths = data_paths
        self._labels = labels

        self.number_of_data_in_dataset = len(self._data_paths)
        # the shape of self.permutations is 30*9
        # self.permutations = self.__retrieve_permutations(jig_classes)
        # self.grid_size = 3
        self.bias_whole_image = percent_of_original_image
        if is_patch_based_or_not:
            self.patch_size = 64
        self._image_transformer = img_transformer
        self._augment_tile = tile_transformer
        if is_patch_based_or_not:
            self.returnFunc = lambda x: x
        # else:
        #
        #     def make_grid(x):
        #         """
        #
        #         :param x:
        #         :return:
        #         """
        #         return torchvision.logger.make_grid(x, self.grid_size, padding=0)
        #     self.returnFunc = make_grid
        #



    # def get_tile(self, img, n):
    #     """Return the augmentation tile of picture
    #
    #     :param img:
    #     :param n:
    #     :return:
    #     """
    #
    #     w = float(img.size[0]) / self.grid_size
    #     y = int(n / self.grid_size)
    #     x = n % self.grid_size
    #     tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
    #     tile = self._augment_tile(tile)
    #     return tile
    
    def get_image(self, index):
        framename = self._data_path + '/' + self._data_paths[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img)
        
    def __getitem__(self, index):
        # TODO(lyj):wait to check
        """

        :param index:
        :return:
        """
        img = self.get_image(index)
        # n_grids = self.grid_size ** 2
        # tiles = [None] * n_grids
        # for n in range(n_grids):
        #     tiles[n] = self.get_tile(img, n)

        # order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        order = np.random.randint(4)

        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0
        if order == 0:
            # data = tiles
            data = self._augment_tile(img)
        else:
            # data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]
            data = self._augment_tile(img.transpose(order + 1))
            #

            # TODO(lyj):transform the data according to the order

        # data = torch.stack(data, 0)
        # return self.returnFunc(data), int(order), int(self.labels[index])
        return data , int(order), int(self._labels[index])

    def __len__(self):
        return len(self._data_paths)

    # def __retrieve_permutations(self, classes):
    #     all_perm = np.load('permutations_%d.npy' % (classes))
    #     # from range [1,9] to [0,8]
    #     if all_perm.min() == 1:
    #         all_perm = all_perm - 1
    #
    #     return all_perm


class JigsawTestDataset(JigsawDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self._data_path + '/' + self._data_paths[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), 0, int(self._labels[index])


class JigsawTestDatasetMultiple(JigsawDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        self._image_transformer = transforms.Compose([
            transforms.Resize(255, Image.BILINEAR),
        ])
        self._image_transformer_full = transforms.Compose([
            transforms.Resize(225, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._augment_tile = transforms.Compose([
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        framename = self._data_paths + '/' + self._data_paths[index]
        _img = Image.open(framename).convert('RGB')
        img = self._image_transformer(_img)

        w = float(img.size[0]) / self.grid_size
        n_grids = self.grid_size ** 2
        images = []
        jig_labels = []
        tiles = [None] * n_grids
        for n in range(n_grids):
            y = int(n / self.grid_size)
            x = n % self.grid_size
            tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
            tile = self._augment_tile(tile)
            tiles[n] = tile
        for order in range(0, len(self.permutations)+1, 3):
            if order==0:
                data = tiles
            else:
                data = [tiles[self.permutations[order-1][t]] for t in range(n_grids)]
            data = self.returnFunc(torch.stack(data, 0))
            images.append(data)
            jig_labels.append(order)
        images = torch.stack(images, 0)
        jig_labels = torch.LongTensor(jig_labels)
        return images, jig_labels, int(self._labels[index])
