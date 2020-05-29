import numpy as np
from torchvision import datasets, transforms
import torch

class Loader():
    def __init__(self, validation=False, inputSize=3, n_classes=10, baseModel=[True, False, False],
                 modifiedModel=[True, False, False, False], dropOut=True, BN=False, bestModel_allLR=False,
                 lrPair=[0.1, 0.05, 0.01], epochNum=350):
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.valida_folder = None
        self.validation = validation
        self.baseModel = baseModel
        self.modifiedModel = modifiedModel
        self.BN = BN
        self.dropOut = dropOut
        self.global_loss = 100000000.0
        self.global_acc = 0
        self.global_model_path = ''
        self.lr = 0
        self.cuda = torch.cuda.is_available()
        self.train_batch_size = 256
        self.test_batch_size = 64
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.dataset_path = './cifar10'
        self.gsync_save = True
        #self.gsync = utils.GDriveSync()
        self.bestModel_allLR = bestModel_allLR
        self.lrPair = lrPair
        self.epochNum = epochNum
        self.inputSize=inputSize
        self.n_classes=n_classes

    def createDataset(self):
        CIFAR10_Train = datasets.CIFAR10(root=self.dataset_path, train=True, download=True)
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}

        if self.validation:
            np.random.seed(1)
            indexSet = np.random.permutation(50000)
            prob = 0.95
            self.indexTrain = indexSet[0:int(50000 * prob)]
            self.indexValid = indexSet[int(50000 * prob):-1]
            train_mean = CIFAR10_Train.data[self.indexTrain].mean(
                axis=(0, 1, 2)) / 255  # [0.49139968  0.48215841  0.44653091]
            train_std = CIFAR10_Train.data[self.indexTrain].std(
                axis=(0, 1, 2)) / 255  # [0.24703223  0.24348513  0.26158784]
        else:
            train_mean = CIFAR10_Train.data.mean(axis=(0, 1, 2)) / 255  # [0.49139968  0.48215841  0.44653091]
            train_std = CIFAR10_Train.data.std(axis=(0, 1, 2)) / 255  # [0.24703223  0.24348513  0.26158784]

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

        del CIFAR10_Train
        CIFAR10_Train = datasets.CIFAR10(root=self.dataset_path, train=True, download=True, transform=transform_train)
        if self.validation:
            trainSet = torch.utils.data.Subset(CIFAR10_Train, self.indexTrain)
            validSet = torch.utils.data.Subset(CIFAR10_Train, self.indexValid)
            self.train_loader = torch.utils.data.DataLoader(trainSet,
                                                            batch_size=self.train_batch_size, shuffle=True, **kwargs)
            self.valida_folder = torch.utils.data.DataLoader(validSet,
                                                             batch_size=self.test_batch_size, shuffle=False, **kwargs)
        else:
            self.train_loader = torch.utils.data.DataLoader(CIFAR10_Train,
                                                            batch_size=self.train_batch_size, shuffle=True, **kwargs)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])
        self.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=self.dataset_path, train=False, download=True,
                             transform=transform_test),
            batch_size=self.test_batch_size, shuffle=False, **kwargs)
        if not self.validation:
            self.valida_folder = self.test_loader