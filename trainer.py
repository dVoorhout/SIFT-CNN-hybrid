import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchsummary import summary
from classifier import Classifier
from sift import Sift
import os
from cnn import Cnn


class Training():
    def __init__(self, train_loader, test_loader, valida_folder, validation=False, sift_size=1152, n_classes=10, dropOut=True, bestModel_allLR=False,
                 lrPair=[0.05, 0.05, 0.01], epochNum=350):
        self.model = None
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.valida_folder = valida_folder
        self.validation = validation
        self.dropOut = dropOut
        self.global_loss = 100000000.0
        self.global_acc = 0
        self.global_model_path = './'
        self.lr = 0
        self.cuda = torch.cuda.is_available()
        self.train_batch_size = 256
        self.test_batch_size = 64
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.dataset_path = './cifar10'
        self.gsync_save = True
        self.bestModel_allLR = bestModel_allLR
        self.lrPair = lrPair
        self.epochNum = epochNum
        self.sift_size = sift_size
        self.n_classes = n_classes
        self.sift = Sift()


    def procedure(self):

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
            self.model = Classifier(1152, 10, self.sift)
            if self.cuda:
                self.model.cuda()

            # Setting for the training
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[200, 250, 300], gamma=0.1)

            # Print model architecture
            print(self.model)
            summary(self.model.cnn, input_size=(3, 32, 32))
            summary(self.model.classifier, input_size=(256, 192 * 5 * 5 + 1152))

            # Start Training
            for self.epoch in range(self.epochNum):
                self.scheduler.step()
                self.train()
                self.valid()

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
            self.model = Classifier(self.sift_size, self.n_classes)
            self.model.load_state_dict(torch.load(self.global_model_path))
            if self.cuda:
                self.model.cuda()
            self.criterion = nn.CrossEntropyLoss()

            # Evaluate
            self.evaluate()

        print('==================Finish==================================')

    def train(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.cuda:
                data, target = data, target.cuda()
            data, target = data, Variable(target)

            self.optimizer.zero_grad()
            output = self.model(data)
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
                    data, target = data, target.cuda()
                data, target = data, Variable(target)

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
            torch.save(self.model.cnn, "best_" + str(self.lr) + ".pt")
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
                    # model_save_name = 'best_' + str(self.lr) + '_' + str(
                    #     np.where(self.model.baseModel)[0][0]) + '_' + str(
                    #     np.where(self.model.modifiedModel)[0][0]) + '_'
                    path = F"./"
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