import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchsummary import summary
from cnn import Cnn
from classifier import Classifier
#import utils
from sift import Sift
import os
from test_models import Model


class Training():
    def __init__(self, validation=False, inputSize=3, n_classes=10, baseModel=[True, False, False],
                 modifiedModel=[False, False, False, True], dropOut=True, BN=False, bestModel_allLR=False,
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


    def Procedure(self):

        # Save the best model among all learning rate
        if self.bestModel_allLR:
            self.global_loss = 100000000.0
            self.global_acc = 0
            self.global_model_path = ''

        # Training
        for self.lr in self.lrPair:
            print("Training with ", self.lr)

            # Save the best model with each learning rate
            if not self.bestModel_allLR:
                self.global_loss = 100000000.0
                self.global_acc = 0
                self.global_model_path = ''

            # Refresh the memory
            try:
                del self.model
            except:
                print('')
            try:
                del self.criterion
                del self.optimizer
                del self.scheduler
            except:
                print('')
            try:
                torch.cuda.empty_cache()
            except:
                print('')

            # Initialize the model
            self.model = Classifier(1152, 10)#Model(inputSize=self.inputSize, n_classes=self.n_classes, baseModel=self.baseModel,
                 #modifiedModel=self.modifiedModel, dropOut=self.dropOut, BN=self.BN)
            #self.model = Cnn()
            if self.cuda:
                self.model.cuda()

            # Setting for the training
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[200, 250, 300], gamma=0.1)

            # Print model architecture
            print(self.model)
            #summary(self.model(1152, 10), input_size=(3, 32, 32))

            # Start Training
            for self.epoch in range(self.epochNum):
                self.scheduler.step()
                self.train()
                #self.valid()

                # If keep diverging, stop at 100 epoch
                # if (self.epoch > 100) & (self.correct.item() > 0) & (
                #         (self.correct.item() / len(self.test_loader.dataset)) <= 0.1):
                #     print("Stop at acc of test set:", self.correct.item() / len(self.test_loader.dataset))
                #     break

            print('===================Result=========================')
            if self.validation:
                print('Validation Set')
            else:
                print('Test set')
            print('Error: ', self.test_loss.item())
            print('Acc', self.correct.item() / len(self.valida_folder.dataset))

            print('========================================================')

        if self.bestModel_allLR:
            if self.validation:
                print('Validation Set')
            else:
                print('Test set')
            print('Error: ', self.global_loss.item())
            print('Acc', self.global_acc.item() / len(self.valida_folder.dataset))
            print('Model: ', self.global_model_path)
            print('========================================================')

        # if use the valid set, evaluate the test set
        if self.validation:

            # Refresh the memory
            try:
                del self.model
            except:
                print('')
            try:
                del self.criterion
                del self.optimizer
                del self.scheduler
            except:
                print('')
            try:
                torch.cuda.empty_cache()
            except:
                print('')

            # Load the model
            #self.model = Model(inputSize=self.inputSize, n_classes=self.n_classes, baseModel=self.baseModel,
                               #modifiedModel=self.modifiedModel, dropOut=self.dropOut, BN=self.BN)
            self.model.load_state_dict(torch.load(self.global_model_path))
            if self.cuda:
                self.model.cuda()
            self.criterion = nn.CrossEntropyLoss()

            # Evaluate
            self.evaluate()

        print('==================Finish==================================')

    def train(self):
        self.model.train()
        sift = Sift()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.cuda:
                data, target = data, target.cuda()
            data, target = data, Variable(target)
            #print("data shape: ", data.shape)
            self.optimizer.zero_grad()
            output = self.model(data, sift, data.shape[0])
            #print(output.shape)
            #print(target.shape)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * len(data), len(self.train_loader.dataset),
                                100. * batch_idx / len(self.train_loader), loss.data))

    def valid(self):
        self.model.eval()
        self.test_loss = 0
        self.correct = 0

        with torch.no_grad():
            for data, target in self.valida_folder:
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                output = self.model(data)
                # sum up batch loss
                self.test_loss += self.criterion(output, target).data
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                self.correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        self.test_loss /= len(self.valida_folder.dataset)
        print(
            '\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                self.test_loss, self.correct, len(self.valida_folder.dataset), 100. * self.correct /
                                                                               len(self.valida_folder.dataset)))

        if self.test_loss < self.best_loss:
            self.best_epoch = self.epoch
            self.best_loss = self.test_loss
            torch.save(self.model, "best_" + str(self.lr) + ".pt")
            '''
            try:
                if self.gsync_save:
                    self.gsync.update_file_to_folder("best_" + str(self.lr) + ".pt")
            except:
                print('Failed to gsync_save.')
            '''
            # Save the best model among three different lr
            if self.test_loss < self.global_loss:

                try:
                    model_save_name = 'best_' + str(self.lr) + '_' + str(
                        np.where(self.model.baseModel)[0][0]) + '_' + str(
                        np.where(self.model.modifiedModel)[0][0]) + '_'
                    path = F"/content/drive/My Drive/dl-reproducibility-project/model/{model_save_name}"
                    torch.save(self.model.state_dict(), path + '.epoch-{}.pt'.format(self.epoch))
                except:
                    print('Failed to save best model to personal google drive')

                self.global_acc = self.correct
                self.global_loss = self.test_loss

                try:
                    os.remove(self.global_model_path)
                except:
                    print('Failed to delete the file')

                self.global_model_path = self.global_model_path = path + '.epoch-{}.pt'.format(self.epoch)

    def evaluate(self):
        self.model.eval()

        prediction = []
        self.test_loss = 0
        self.correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                output = self.model(data)
                prediction += [output]
                # sum up batch loss
                self.test_loss += self.criterion(output, target).data
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                self.correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        self.test_loss /= len(self.test_loader.dataset)
        print('Test set')
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                self.test_loss, self.correct, len(self.test_loader.dataset), 100. * self.correct /
                                                                             len(self.test_loader.dataset)))
        print('========================================================')

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