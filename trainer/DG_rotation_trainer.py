
# Complex input should be an entity of class.
# Name of method in a class should contain an obvious verbal, like get_something().
# set attribute within __init__(self) function.

import torch
from torch import nn
from torch.nn import functional as F

from trainer_utils.model.MyModel import MyModel

from trainer_utils.logger.Logger import Logger
from trainer_utils.training_argument.MyTrainingArgument import MyTrainingArgument
from trainer_utils.data_loader.MyDataLoader import MyDataLoader
from trainer_utils.optimizer.MyOptimizer import MyOptimizer
from trainer_utils.scheduler.MyScheduler import MyScheduler


class Trainer:
    def __init__(self, my_training_arguments, my_model, my_data_loader, my_optimizer, my_scheduler, device):
        self.training_arguments = my_training_arguments.training_arguments
        self.device = device
        self.model = my_model.model.to(device)
        ##
        self.unsupervised_task_loss_weight = self.training_arguments.jig_weight
        self.classify_only_ordered_images_or_not = self.training_arguments.classify_only_sane
        self.number_of_images_classes = self.training_arguments.n_classes

        self.train_data_loader = my_data_loader.train_data_loader
        self.validation_data_loader = my_data_loader.validation_data_loader
        self.test_data_loader = my_data_loader.test_data_loader
        ##
        self.test_loaders = {"val": self.validation_data_loader, "test": self.test_data_loader}
        print("Dataset size: trainer %d, val %d, test %d" % (len(self.train_data_loader.dataset), len(self.validation_data_loader.dataset), len(self.test_data_loader.dataset)))

        self.optimizer = my_optimizer.optimizer
        self.scheduler = my_scheduler.scheduler

        if self.training_arguments.target in self.training_arguments.source:
            self.target_domain_index = self.training_arguments.source.index(self.training_arguments.target)
            print("Target in source: %d" % self.target_domain_index)
            print(self.training_arguments.source)
        else:
            self.target_domain_index = None

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()
        # Set the mode of the model to trainer, then the parameters can begin to be trained
        self.model.train()

        # domain_index_of_images_in_this_patch is target domain index in the source domain list
        for i, ((data, rotation_label, class_label), domain_index_of_images_in_this_patch) in enumerate(self.train_data_loader):
            data, rotation_label, class_label, domain_index_of_images_in_this_patch = data.to(self.device), rotation_label.to(self.device), class_label.to(self.device), domain_index_of_images_in_this_patch.to(self.device)
            self.optimizer.zero_grad()

            rotation_predict_label, class_predict_label = self.model(data)  # , lambda_val=lambda_val)
            unsupervised_task_loss = criterion(rotation_predict_label, rotation_label)

            if self.classify_only_ordered_images_or_not:
                if self.target_domain_index is not None:
                    # images_should_be_selected_or_not is a 128*1 list containing True or False.
                    images_should_be_selected_or_not = (rotation_label == 0) & (domain_index_of_images_in_this_patch != self.target_domain_index)
                    supervised_task_loss = criterion(
                        class_predict_label[images_should_be_selected_or_not],
                        class_label[images_should_be_selected_or_not]
                    )
                else:
                    supervised_task_loss = criterion(class_predict_label[rotation_label == 0], class_label[rotation_label == 0])

            elif self.target_domain_index:
                supervised_task_loss = criterion(class_predict_label[domain_index_of_images_in_this_patch != self.target_domain_index], class_label[domain_index_of_images_in_this_patch != self.target_domain_index])
            else:
                supervised_task_loss = criterion(class_predict_label, class_label)
            _, cls_pred = class_predict_label.max(dim=1)
            _, jig_pred = rotation_predict_label.max(dim=1)
            # _, domain_pred = domain_logit.max(dim=1)
            loss = supervised_task_loss + unsupervised_task_loss * self.unsupervised_task_loss_weight

            loss.backward()
            self.optimizer.step()

            self.logger.log(
                i,
                len(self.train_data_loader),
                {
                    "jigsaw": unsupervised_task_loss.item(),
                    "class": supervised_task_loss.item()
                 },
                {
                    "jigsaw": torch.sum(jig_pred == rotation_label.data).item(),
                    "class": torch.sum(cls_pred == class_label.data).item(),
                 },
                data.shape[0]
            )
            del loss, supervised_task_loss, unsupervised_task_loss, rotation_predict_label, class_predict_label
        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                if loader.dataset.isMulti():
                    jigsaw_correct, class_correct, single_acc = self.do_test_multi(loader)
                    print("Single vs multi: %g %g" % (float(single_acc) / total, float(class_correct) / total))
                else:
                    jigsaw_correct, class_correct = self.do_test(loader)
                jigsaw_acc = float(jigsaw_correct) / total
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {"jigsaw": jigsaw_acc, "class": class_acc})
                self.results[phase][self.current_epoch] = class_acc


    def do_test(self, loader):
        jigsaw_correct = 0
        class_correct = 0
        domain_correct = 0
        for it, ((data, jig_l, class_l), _) in enumerate(loader):
            data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)
            jigsaw_logit, class_logit = self.model(data)
            _, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
            jigsaw_correct += torch.sum(jig_pred == jig_l.data)
        return jigsaw_correct, class_correct


    def do_test_multi(self, loader):
        jigsaw_correct = 0
        class_correct = 0
        single_correct = 0
        for it, ((data, jig_l, class_l), d_idx) in enumerate(loader):
            data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)
            n_permutations = data.shape[1]
            class_logits = torch.zeros(n_permutations, data.shape[0], self.number_of_images_classes).to(self.device)
            for k in range(n_permutations):
                class_logits[k] = F.softmax(self.model(data[:, k])[1], dim=1)
            class_logits[0] *= 4 * n_permutations  # bias more the original image
            class_logit = class_logits.mean(0)
            _, cls_pred = class_logit.max(dim=1)
            jigsaw_logit, single_logit = self.model(data[:, 0])
            _, jig_pred = jigsaw_logit.max(dim=1)
            _, single_logit = single_logit.max(dim=1)
            single_correct += torch.sum(single_logit == class_l.data)
            class_correct += torch.sum(cls_pred == class_l.data)
            jigsaw_correct += torch.sum(jig_pred == jig_l.data[:, 0])
        return jigsaw_correct, class_correct, single_correct


    def do_training(self):
        self.logger = Logger(self.training_arguments, update_frequency=30)  # , "domain", "lambda"
        self.results = {"val": torch.zeros(self.training_arguments.epochs), "test": torch.zeros(self.training_arguments.epochs)}
        for self.current_epoch in range(self.training_arguments.epochs):
            print("current epoch:%d", self.current_epoch)
            self.scheduler.step()
            self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch()
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        #print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
        print(self.training_arguments.target)
        print("Highest accuracy on validation set appears on epoch ", val_res.argmax().data)
        print("Highest accuracy on test set appears on epoch ",  test_res.argmax().data)
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger, self.model


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    my_training_arguments = MyTrainingArgument()
    my_model = MyModel(my_training_arguments)
    is_patch_based_or_not = my_model.model.is_patch_based()
    my_data_loader = MyDataLoader(my_training_arguments, is_patch_based_or_not)
    my_optimizer = MyOptimizer(my_training_arguments, my_model)
    my_scheduler = MyScheduler(my_training_arguments, my_optimizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(my_training_arguments, my_model, my_data_loader, my_optimizer, my_scheduler, device)
    trainer.do_training()