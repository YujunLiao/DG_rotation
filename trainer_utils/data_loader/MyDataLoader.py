from data.rotation import data_helper


class MyDataLoader:
    """Return source, validation, test data loaders.

    Implementation:
        source_data_loader
        validation_data_loader
        test_data_loader

    """
    def __init__(self, my_training_arguments, is_patch_based_or_not):
        """

        :param my_training_arguments:
        :param is_patch_based_or_not:
        """
        self.source_data_loader, self.validation_data_loader = data_helper.get_train_dataloader(
            my_training_arguments,
            patches=is_patch_based_or_not
        )
        self.test_data_loader = data_helper.get_val_dataloader(
            my_training_arguments,
            patches=is_patch_based_or_not
        )