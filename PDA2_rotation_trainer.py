
# Complex input should be an entity of class.
# Name of method in a class should contain an obvious verbal, like get_something().
# set attribute within __init__(self) function.
# 每天学一点，每天改一点

import torch
from torch import nn
from torch.nn import functional as F
from time import time, strftime, localtime

from trainer_utils.model.MyModel import MyModel

from trainer_utils.logger.Logger import Logger
from trainer_utils.training_argument.DARotationTrainingArgument import DARotationTrainingArgument
from trainer_utils.data_loader.PDA2RotationDataLoader import PDARotationDataLoader
from trainer_utils.optimizer.MyOptimizer import MyOptimizer
from trainer_utils.scheduler.MyScheduler import MyScheduler
from trainer_utils.output_manager.OutputManager import OutputManager
from trainer_utils.lazy_man.LazyMan import LazyMan, LazyMan2
import itertools
import torch.nn.functional as func
import socket


class PDARotationTrainer:
    def __init__(self, my_training_arguments, my_model, my_data_loader, my_optimizer, my_scheduler, device, output_manager):
        self.training_arguments = my_training_arguments.training_arguments
        self.device = device
        self.model = my_model.model.to(device)
        self.output_manager=output_manager

        self.source_domain_train_data_loader = my_data_loader.source_domain_train_data_loader
        self.target_domain_train_data_loader = my_data_loader.target_domain_train_data_loader
        self.validation_data_loader = my_data_loader.validation_data_loader
        self.test_data_loader = my_data_loader.test_data_loader

        self.optimizer = my_optimizer.optimizer
        self.scheduler = my_scheduler.scheduler

        self.number_of_images_classes = self.training_arguments.n_classes
        self.test_loaders = {"val": self.validation_data_loader, "test": self.test_data_loader}

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
        for i, (data_from_source_domain, data_from_target_domain) \
                in enumerate(zip(self.source_domain_train_data_loader, itertools.cycle(self.target_domain_train_data_loader))):

            (data, rotation_label, class_label), domain_index_of_images_in_this_patch = data_from_source_domain
            data, rotation_label, class_label, domain_index_of_images_in_this_patch = \
                data.to(self.device), rotation_label.to(self.device), class_label.to(self.device), domain_index_of_images_in_this_patch.to(self.device)

            (target_domain_data, target_domain_rotation_label, _), _ = data_from_target_domain
            target_domain_data, target_domain_rotation_label = target_domain_data.to(self.device), target_domain_rotation_label.to(self.device)


            self.optimizer.zero_grad()

            rotation_predict_label, class_predict_label = self.model(data)  # , lambda_val=lambda_val)
            # unsupervised_task_loss = criterion(rotation_predict_label, rotation_label)

            target_domain_rotation_predict_label, target_domain_class_predict_label = self.model(target_domain_data)  # , lambda_val=lambda_val)
            target_domain_unsupervised_task_loss = criterion(target_domain_rotation_predict_label, target_domain_rotation_label)
            target_domain_entropy_loss = self._entropy_loss(target_domain_class_predict_label[target_domain_rotation_label==0])

            if self.training_arguments.classify_only_ordered_images_or_not:
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
            loss = supervised_task_loss \
            + target_domain_unsupervised_task_loss * self.training_arguments.target_domain_unsupervised_task_loss_weight\
            + target_domain_entropy_loss * self.training_arguments.entropy_loss_weight

            loss.backward()
            self.optimizer.step()

            self.logger.log(
                i,
                len(self.source_domain_train_data_loader),
                {
                    # "jigsaw": unsupervised_task_loss.item(),
                    "class": supervised_task_loss.item(),
                    "t_rotation": target_domain_unsupervised_task_loss.item(),
                    "entropy": target_domain_entropy_loss.item()
                 },
                {
                    "jigsaw": torch.sum(jig_pred == rotation_label.data).item(),
                    "class": torch.sum(cls_pred == class_label.data).item(),
                 },
                data.shape[0]
            )
            del loss, supervised_task_loss, rotation_predict_label, class_predict_label
            del target_domain_rotation_predict_label, target_domain_class_predict_label
            del target_domain_unsupervised_task_loss, target_domain_entropy_loss

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

    def _entropy_loss(self, x):
        return torch.sum(-func.softmax(x, 1) * func.log_softmax(x, 1), 1).mean()

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
        print('******************Start training**************************************')
        print('--------------------------------------------------------')
        print("Dataset size: training %d, validation %d, testing %d" % (len(self.source_domain_train_data_loader.dataset), len(self.validation_data_loader.dataset), len(self.test_data_loader.dataset)))
        print('--------------------------------------------------------')
        print(self.training_arguments)
        print('--------------------------------------------------------')

        # TODO(lyj):
        self.logger = Logger(self.training_arguments, update_frequency=30)  # , "domain", "lambda"
        self.results = {"val": torch.zeros(self.training_arguments.epochs), "test": torch.zeros(self.training_arguments.epochs)}
        for self.current_epoch in range(self.training_arguments.epochs):
            self.scheduler.step()
            lrs = self.scheduler.get_lr()
            self.logger.new_epoch(lrs)
            print('--------------------------------------------------------')
            print("current epoch:%d", self.current_epoch)
            print("New epoch - lr: %s" % ", ".join([str(lr) for lr in lrs]))
            self._do_epoch()
        val_res = self.results["val"]
        test_res = self.results["test"]
        # idx_best = val_res.argmax()
        idx_best = self._smooth(val_res.tolist(), 0.6)[1]-1

        # print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
        print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
        print(self.training_arguments.target)
        print(self.training_arguments.source)
        print("unsupervised_task_weight:", self.training_arguments.unsupervised_task_weight)
        # TODO(change bias whole image)
        print("bias_hole_image:", self.training_arguments.bias_whole_image)
        print("target_rotation_weight:", self.training_arguments.target_domain_unsupervised_task_loss_weight)
        print("entropy_weight", self.training_arguments.entropy_loss_weight)
        print("only_classify the ordered image:", self.training_arguments.classify_only_ordered_images_or_not)
        print("Highest accuracy on validation set appears on epoch ", val_res.argmax().data)
        print("Highest accuracy on test set appears on epoch ",  test_res.argmax().data)
        print("Accuracy on test set when the accuracy on validation set is highest:%.3f" %test_res[idx_best])
        print("Highest accuracy on test set:%.3f" %test_res.max())
        self.logger.save_best(test_res[idx_best], test_res.max())

        self.output_manager.write_to_output_file([
            '--------------------------------------------------------',
            str(strftime("%Y-%m-%d %H:%M:%S", localtime()) ),
            self.training_arguments.source,
            "target domain:" + self.training_arguments.target,
            "jigweight:" + str(self.training_arguments.unsupervised_task_weight),
            "bias_hole_image:"+ str(self.training_arguments.bias_whole_image),
            "target_rotation_weight:" + str(self.training_arguments.target_domain_unsupervised_task_loss_weight),
            "entropy_weight:" + str(self.training_arguments.entropy_loss_weight),
            "only_classify the ordered image:"+str(self.training_arguments.classify_only_ordered_images_or_not),
            "batch_size:"+str(self.training_arguments.batch_size)+" learning_rate:"+str(self.training_arguments.learning_rate),
            "Highest accuracy on validation set appears on epoch "+ str(val_res.argmax().data),
            "Highest accuracy on test set appears on epoch "+ str(test_res.argmax().data),
            str("Accuracy on test set when the accuracy on validation set is highest:%.3f" % test_res[idx_best]),
            str("Highest accuracy on test set:%.3f" % test_res.max()),
            str("It took %g" % (time() - self.logger.start_time))
        ])

        return self.logger, self.model

    def _smooth(self, scalars, weight):  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(float(smoothed_val))  # Save it
            last = smoothed_val  # Anchor the last smoothed value

        smoothed_array = []
        for i in smoothed:
            smoothed_array.append(i)

        massimo = max(smoothed_array)
        epoch = smoothed_array.index(massimo)
        epoch = epoch + 1
        return smoothed_array, epoch

def lazy_train(my_training_arguments, output_manager):
    my_model = MyModel(my_training_arguments)
    is_patch_based_or_not = my_model.model.is_patch_based()
    my_data_loader = PDARotationDataLoader(my_training_arguments, is_patch_based_or_not)
    my_optimizer = MyOptimizer(my_training_arguments, my_model)
    my_scheduler = MyScheduler(my_training_arguments, my_optimizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = PDARotationTrainer(my_training_arguments, my_model, my_data_loader, my_optimizer, my_scheduler, device, output_manager)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True


    DA_rotation_training_argument = DARotationTrainingArgument()
    # my_training_arguments.training_arguments.classify_only_ordered_images_or_not=True
    DA_rotation_training_argument.training_arguments.TTA = False
    DA_rotation_training_argument.training_arguments.nesterov = False

    for parameter_pair in DA_rotation_training_argument.training_arguments.parameters_lists:

        DA_rotation_training_argument.training_arguments.unsupervised_task_weight=parameter_pair[0]
        DA_rotation_training_argument.training_arguments.bias_whole_image=parameter_pair[1]
        DA_rotation_training_argument.training_arguments.target_domain_unsupervised_task_loss_weight=parameter_pair[2]
        DA_rotation_training_argument.training_arguments.entropy_loss_weight=parameter_pair[3]

            # lazy_man = LazyMan(['CALTECH', 'LABELME', 'PASCAL', 'SUN'])
        # lazy_man = LazyMan(
        #     ['art_painting', 'cartoon', 'sketch', 'photo'],
        #     ['art_painting', 'cartoon', 'sketch', 'photo']
        # )
        lazy_man = LazyMan(
            DA_rotation_training_argument.training_arguments.domains_list,
            DA_rotation_training_argument.training_arguments.target_domain_list
        )

        for source_and_target_domain in lazy_man.source_and_target_domain_permutation_list:
            DA_rotation_training_argument.training_arguments.source=source_and_target_domain['source_domain']
            DA_rotation_training_argument.training_arguments.target=source_and_target_domain['target_domain']

            output_manager = OutputManager(
                output_file_path=\
                '/home/giorgio/Files/pycharm_project/DG_rotation/trainer_utils/output_manager/output_file/' + \
                socket.gethostname() + "/PDA2_rotation/" + \
                DA_rotation_training_argument.training_arguments.network + '/'+ \
                str(DA_rotation_training_argument.training_arguments.unsupervised_task_weight) + '_' + \
                str(DA_rotation_training_argument.training_arguments.bias_whole_image) + '_' + \
                str(DA_rotation_training_argument.training_arguments.target_domain_unsupervised_task_loss_weight) + '_' + \
                str(DA_rotation_training_argument.training_arguments.entropy_loss_weight) + '/',
                output_file_name=DA_rotation_training_argument.training_arguments.source[0]+'_'+ DA_rotation_training_argument.training_arguments.target


            )
            for i in range(int(DA_rotation_training_argument.training_arguments.repeat_times)):
                lazy_train(DA_rotation_training_argument, output_manager)



