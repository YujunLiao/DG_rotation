import argparse
import torch
from IPython.core.debugger import set_trace
from torch import nn
from torch.nn import functional as F
# from data import data_helper
# rotation
from data.rotation import data_helper
# from IPython.core.debugger import set_trace
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Arguments for training
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="caffenet")
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--n_classes", "-c", type=int, default=31, help="Number of classes")
    # parser.add_argument("--jigsaw_n_classes", "-jc", type=int, default=31, help="Number of classes for the jigsaw task")
    parser.add_argument("--jig_weight", type=float, default=0.1, help="Weight for the jigsaw puzzle")

    parser.add_argument("--ooo_weight", type=float, default=0, help="Weight for odd one out task")
    # image_size: For example, an image's size is 3*225*225
    parser.add_argument("--image_size", type=int, default=225, help="Image size")
    # In general, in each epoch, all the samples are used for one time.
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    # batch_size (int, optional): The number of samples used for training in each iteration.
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")

    # Argument for logger.
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")

    #
    parser.add_argument("--limit_source", default=None, type=int, help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int, help="If set, it will limit the number of testing samples")
    parser.add_argument("--bias_whole_image", default=None, type=float, help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--classify_only_sane", default=False, type=bool, help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=False, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")

    # Arguments for Test-Time Augmentation
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.0, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.0, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float,
                        help="Chance of randomly greyscaling a tile")

    return parser.parse_args()


class TraningArgument(argparse.ArgumentParser):
    def __init__(self):
        super(TraningArgument, self).__init__(
            description="Script to launch jigsaw training",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    def get_arguments_from_console(self):
        # Arguments for training
        self.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use",
                            default="caffenet")
        self.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
        self.add_argument("--target", choices=available_datasets, help="Target")
        self.add_argument("--n_classes", "-c", type=int, default=31, help="Number of classes")
        # parser.add_argument("--jigsaw_n_classes", "-jc", type=int, default=31, help="Number of classes for the jigsaw task")
        self.add_argument("--jig_weight", type=float, default=0.1, help="Weight for the jigsaw puzzle")

        self.add_argument("--ooo_weight", type=float, default=0, help="Weight for odd one out task")
        # image_size: For example, an image's size is 3*225*225
        self.add_argument("--image_size", type=int, default=225, help="Image size")
        # In general, in each epoch, all the samples are used for one time.
        self.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
        # batch_size (int, optional): The number of samples used for training in each iteration.
        self.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
        self.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
        self.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")

        # Argument for logger.
        self.add_argument("--tf_logger", type=bool, default=True,
                            help="If true will save tensorboard compatible logs")
        self.add_argument("--folder_name", default=None, help="Used by the logger to save logs")

        #
        self.add_argument("--limit_source", default=None, type=int,
                            help="If set, it will limit the number of training samples")
        self.add_argument("--limit_target", default=None, type=int,
                            help="If set, it will limit the number of testing samples")
        self.add_argument("--bias_whole_image", default=None, type=float,
                            help="If set, will bias the training procedure to show more often the whole image")
        self.add_argument("--classify_only_sane", default=False, type=bool,
                            help="If true, the network will only try to classify the non scrambled images")
        self.add_argument("--train_all", default=False, type=bool,
                            help="If true, all network weights will be trained")
        self.add_argument("--suffix", default="", help="Suffix for the logger")
        self.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")

        # Arguments for Test-Time Augmentation
        self.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
        self.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
        self.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
        self.add_argument("--random_horiz_flip", default=0.0, type=float, help="Chance of random horizontal flip")
        self.add_argument("--jitter", default=0.0, type=float, help="Color jitter amount")
        self.add_argument("--tile_random_grayscale", default=0.1, type=float,
                            help="Chance of randomly greyscaling a tile")

    def arguments_dictionary(self):
        return self.parse_args()



class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        # TODO(LYJ):Wait to change.
        model = model_factory.get_network(args.network)(
            # Jigsaw class of 0 refers to original picture, apart from the original one, there
            # are another 30 classes, in total 31 classes of jigsaw pictures.
            # jigsaw_classes=args.jigsaw_n_classes + 1,

            # When using rotation technology as the unsupervised task, there are in total
            # 4 classes, which are original one, 90, 180, 270 degree.
            jigsaw_classes = 4,
            classes=args.n_classes
        )
        self.model = model.to(device)

        # TODO(LYJ):patch based
        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
        # self.source_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())

        self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)

        print("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all, nesterov=args.nesterov)
        self.jig_weight = args.jig_weight # Weight for the jigsaw puzzle
        self.only_non_scrambled = args.classify_only_sane # if true only classify the orderd images
        self.n_classes = args.n_classes
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()
        # Set the mode of the model to train, then the parameters can begin to be trained
        self.model.train()

        # Code for rotation


        # Code for Jiasaw
        # d_idx is domain index
        for it, ((data, jig_l, class_l), d_idx) in enumerate(self.source_loader):
            data, jig_l, class_l, d_idx = data.to(self.device), jig_l.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            # absolute_iter_count = it + self.current_epoch * self.len_dataloader
            # p = float(absolute_iter_count) / self.args.epochs / self.len_dataloader
            # lambda_val = 2. / (1. + np.exp(-10 * p)) - 1
            # if domain_error > 2.0:
            #     lambda_val  = 0
            # print("Shutting down LAMBDA to prevent implosion")

            self.optimizer.zero_grad()

            jigsaw_logit, class_logit = self.model(data)  # , lambda_val=lambda_val)
            jigsaw_loss = criterion(jigsaw_logit, jig_l)
            # domain_loss = criterion(domain_logit, d_idx)
            # domain_error = domain_loss.item()
            if self.only_non_scrambled:
                if self.target_id is not None:
                    idx = (jig_l == 0) & (d_idx != self.target_id)
                    class_loss = criterion(class_logit[idx], class_l[idx])
                else:
                    class_loss = criterion(class_logit[jig_l == 0], class_l[jig_l == 0])

            elif self.target_id:
                class_loss = criterion(class_logit[d_idx != self.target_id], class_l[d_idx != self.target_id])
            else:
                class_loss = criterion(class_logit, class_l)
            _, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)
            # _, domain_pred = domain_logit.max(dim=1)
            loss = class_loss + jigsaw_loss * self.jig_weight  # + 0.1 * domain_loss

            loss.backward()
            self.optimizer.step()

            self.logger.log(it, len(self.source_loader),
                            {"jigsaw": jigsaw_loss.item(), "class": class_loss.item()  # , "domain": domain_loss.item()
                             },
                            # ,"lambda": lambda_val},
                            {"jigsaw": torch.sum(jig_pred == jig_l.data).item(),
                             "class": torch.sum(cls_pred == class_l.data).item(),
                             # "domain": torch.sum(domain_pred == d_idx.data).item()
                             },
                            data.shape[0])
            del loss, class_loss, jigsaw_loss, jigsaw_logit, class_logit
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
            class_logits = torch.zeros(n_permutations, data.shape[0], self.n_classes).to(self.device)
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
        self.logger = Logger(self.args, update_frequency=30)  # , "domain", "lambda"
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            print("current epoch:%d", self.current_epoch)
            self.scheduler.step()
            self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch()
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        #print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
        print(self.args.target)
        print("Highest accuracy on validation set appears on epoch ", val_res.argmax().data)
        print("Highest accuracy on test set appears on epoch ",  test_res.argmax().data)
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger, self.model

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    training_arguments = TraningArgument()
    training_arguments.get_arguments_from_console()
    # args = get_args()
    args = training_arguments.arguments_dictionary()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()