from trainer_utils.data_loader.helper.data_helper import get_train_transformers, get_val_transformer
from os.path import join, dirname
from trainer_utils.data_loader.helper.JigsawLoader import JigsawDataset, JigsawTestDataset, get_split_dataset_info, _dataset_info
from trainer_utils.data_loader.helper.concat_dataset import ConcatDataset
from trainer_utils.data_loader.helper.data_helper import Subset
from random import sample, random


class MyDataset:
    """Return train, validation, test datasets.

    Implementation:
        train_dataset:Content: train_data_paths, train_labels
        validation_dataset:Content: validation_data_paths, validation_labels
        test_dataset

    """
    def __init__(self, my_training_arguments, is_patch_based_or_not):
        self.train_dataset, self.validation_dataset = self._get_train_and_validation_dataset(my_training_arguments, is_patch_based_or_not)
        self.test_dataset = self._get_test_dataset(my_training_arguments, is_patch_based_or_not)

    def _get_train_and_validation_dataset(self, my_training_arguments, is_patch_based_or_not=False):
        """Return train and validation dataset

        :param my_training_arguments:
        :param is_patch_based_or_not:
        :return:
            train_dataset:Content: train_data_paths, train_labels
            validation_dataset:Content: validation_data_paths, validation_labels
        """
        training_arguments = my_training_arguments.training_arguments
        img_transformer, tile_transformer = get_train_transformers(training_arguments)
        train_dataset_list = []
        validation_dataset_list = []
        max_number_of_train_dataset = training_arguments.limit_source
        source_domains_names_list = training_arguments.source
        assert isinstance(source_domains_names_list, list)

        for domain_name in source_domains_names_list:

            # path_of_txt_list_of_data refer to where are the txt files that record all images' path, for example,
            # /home/lyj/Files/project/pycharm/DG_rotation/trainer_utils/data_loader/txt_lists/photo_train.txt
            path_of_txt_list = join(dirname(__file__), 'txt_lists', '%s_train.txt' % domain_name)
            data_paths, labels = self._get_data_paths_and_labels_from_txt_list(path_of_txt_list)
            # training_arguments.val_size refer to the percent of validation dataset, for example,
            # val_size=0.1 means 10% data is used for validation, 90% data is used for training.
            train_data_paths, validation_data_paths, train_labels, validation_labels = self._split_dataset_randomly(
                data_paths,
                labels,
                training_arguments.val_size
            )



            # train_data_paths, validation_data_paths, train_labels, validation_labels = \
            #     get_split_dataset_info(
            #     path_of_txt_list,
            #     training_arguments.val_size
            # )

            train_dataset = JigsawDataset(
                train_data_paths,
                train_labels,
                patches=is_patch_based_or_not,
                img_transformer=img_transformer,
                tile_transformer=tile_transformer,
                jig_classes=4,
                bias_whole_image=training_arguments.bias_whole_image
            )

            if max_number_of_train_dataset:
                train_dataset = Subset(train_dataset, max_number_of_train_dataset)

            train_dataset_list.append(train_dataset)

            validation_dataset_list.append(
                JigsawTestDataset(
                    validation_data_paths,
                    validation_labels,
                    img_transformer=get_val_transformer(training_arguments),
                    patches=is_patch_based_or_not,
                    jig_classes=4)
            )

        train_dataset = ConcatDataset(train_dataset_list)
        validation_dataset = ConcatDataset(validation_dataset_list)
        return train_dataset, validation_dataset

    def _get_test_dataset(self, my_training_arguments, is_patch_based_or_not):
        training_arguments = my_training_arguments.training_arguments
        max_number_of_test_dataset = training_arguments.limit_target

        # path_of_txt_list_of_data refer to where are the txt files that record all images' path, for example,
        # /home/lyj/Files/project/pycharm/DG_rotation/trainer_utils/data_loader/txt_lists/photo_train.txt
        path_of_txt_list = join(dirname(__file__), 'txt_lists', '%s_test.txt' % training_arguments.target)
        data_paths, labels = self._get_data_paths_and_labels_from_txt_list(path_of_txt_list)
        img_tr = get_val_transformer(training_arguments)
        test_dataset = JigsawTestDataset(
            data_paths,
            labels,
            patches=is_patch_based_or_not,
            img_transformer=img_tr,
            jig_classes=4)

        if max_number_of_test_dataset and len(test_dataset) > max_number_of_test_dataset:
            test_dataset = Subset(test_dataset, max_number_of_test_dataset)
            print("Using %d subset of val dataset" % training_arguments.limit_target)

        return test_dataset

    def _get_data_paths_and_labels_from_txt_list(self, txt_labels):
        with open(txt_labels, 'r') as f:
            lines = f.readlines()

        data_paths = []
        labels = []
        for line in lines:
            line = line.split(' ')
            data_paths.append(line[0])
            labels.append(int(line[1]))

        return data_paths, labels

    def _split_dataset_randomly(self, data_paths, labels, percent):
        """

        :param data_paths: list of images paths
        :param labels:  list of labels
        :param percent: 0 < float < 1
        :return:
        """
        number_of_data = len(data_paths)
        # random_indexes is a list of index of (number_of_data * percent) samples from original data list.
        random_indexes = sample(range(number_of_data), int(number_of_data * percent))
        validation_data_paths = [data_paths[k] for k in random_indexes]
        train_data_paths = [v for k, v in enumerate(data_paths) if k not in random_indexes]
        validation_labels = [labels[k] for k in random_indexes]
        train_labels = [v for k, v in enumerate(labels) if k not in random_indexes]
        return train_data_paths, validation_data_paths, train_labels, validation_labels