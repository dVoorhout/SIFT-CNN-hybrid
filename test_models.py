import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    def __init__(self, inputSize=3, n_classes=10, baseModel=[True, False, False],
                 modifiedModel=[True, False, False, False], dropOut=True, BN=False, **kwargs):
        super(Model, self).__init__()
        self.inputSize = inputSize
        self.n_classes = n_classes
        self.modifiedModel = modifiedModel
        self.baseModel = baseModel
        self.dropOut = dropOut
        self.BN = BN
        self.layers = []
        self.model = nn.Sequential()
        # self.description=[]

        # Debug of baseModel
        if np.sum(self.baseModel) >= 2 or np.sum(self.baseModel) == 0:
            print('Error in baseModel. Choose model A')
            self.baseModel[0] = True
            self.baseModel[1] = False
            self.baseModel[2] = False

        # Debug of modifiedModel
        if np.sum(self.modifiedModel) >= 2 or np.sum(self.modifiedModel) == 0:
            print('Error in modifiedModel. Choose model without modification')
            self.modifiedModel[0] = True
            self.modifiedModel[1] = False
            self.modifiedModel[2] = False
            self.modifiedModel[3] = False

        #
        # Intialize Layers and build the model
        #

        # Top Layers
        self.conv_3_192_192_1 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv_1_192_192 = nn.Conv2d(192, 192, 1)
        self.conv_1_192_class = nn.Conv2d(192, self.n_classes, 1)

        # Batch Normalzation
        self.BN_96 = nn.BatchNorm2d(96)
        self.BN_192 = nn.BatchNorm2d(192)
        self.BN_class= nn.BatchNorm2d(n_classes)
        # Max Pooling
        self.maxP = nn.MaxPool2d(3, stride=2)

        # Other
        self.softMax = nn.Softmax()
        self.flatten = nn.Flatten()
        self.dropOut_2 = nn.Dropout(p=0.2)
        self.dropOut_5 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.avgPooling = nn.AdaptiveAvgPool2d(1)

        if self.baseModel[0]:
            # Model A
            self.conv_5_Input_96_1 = nn.Conv2d(self.inputSize, 96, 5, padding=1)
            self.conv_5_96_192_1 = nn.Conv2d(96, 192, 5, padding=1)

            # Initialize layers
            if self.modifiedModel[1]:
                # Strided A
                self.conv_5_Input_96_2 = nn.Conv2d(self.inputSize, 96, 5, padding=1, stride=2)
                self.conv_5_96_192_2 = nn.Conv2d(96, 192, 5, padding=1, stride=2)
            elif self.modifiedModel[2]:
                # ConvPool
                self.conv_5_96_96_1 = nn.Conv2d(96, 96, 5, padding=1)
                self.conv_5_192_192_1 = nn.Conv2d(192, 192, 5, padding=1)
            elif self.modifiedModel[3]:
                # All A
                self.conv_5_96_96_2 = nn.Conv2d(96, 96, 5, padding=1, stride=2)
                self.conv_5_192_192_2 = nn.Conv2d(192, 192, 5, padding=1, stride=2)

            if self.modifiedModel[0]:
                self.__buildModelA()
            elif self.modifiedModel[1]:
                self.__buildModel_stridedA()
            elif self.modifiedModel[2]:
                self.__buildModel_ConvPoolA()
            else:
                self.__buildModel_AllA()


        elif self.baseModel[1]:

            # Model B
            self.conv_5_Input_96_1 = nn.Conv2d(self.inputSize, 96, 5, padding=1)
            self.conv_5_96_192_1 = nn.Conv2d(96, 192, 5, padding=1)
            self.conv_1_96_96_1 = nn.Conv2d(96, 96, 1, padding=1)
            self.conv_1_192_192_1 = nn.Conv2d(192, 192, 1, padding=1)
            if self.modifiedModel[1] or self.modifiedModel[3]:
                # Strided B amd All B
                self.conv_1_96_96_2 = nn.Conv2d(96, 96, 1, padding=1, stride=2)
                self.conv_1_192_192_2 = nn.Conv2d(192, 192, 1, padding=1, stride=2)

            if self.modifiedModel[0]:
                self.__buildModelB()
            elif self.modifiedModel[1]:
                self.__buildModel_stridedB()
            elif self.modifiedModel[2]:
                self.__buildModel_ConvPoolB()
            else:
                self.__buildModel_AllB()

        else:
            #  Model C
            self.conv_3_Input_96_1 = nn.Conv2d(self.inputSize, 96, 3, padding=1)
            self.conv_3_96_96_1 = nn.Conv2d(96, 96, 3, padding=1)
            self.conv_3_96_192_1 = nn.Conv2d(96, 192, 3, padding=1)

            if self.modifiedModel[1] or self.modifiedModel[3]:
                # Strided C and All C
                self.conv_3_192_192_2 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
                self.conv_3_96_96_2 = nn.Conv2d(96, 96, 3, padding=1, stride=2)

            if self.modifiedModel[0]:
                self.__buildModelC()
            elif self.modifiedModel[1]:
                self.__buildModel_stridedC()
            elif self.modifiedModel[2]:
                self.__buildModel_ConvPoolC()
            else:
                self.__buildModel_AllC()
        # Build top layers
        self.__buildTopLayer()
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)

        return x

    def __buildTopLayer(self):
        '''
                The top layers are identical in each model
                '''
        # Layer 6
        self.layers.append(self.conv_3_192_192_1)
        if self.BN:
            self.layers.append(self.BN_192)
        self.layers.append(self.relu)

        # Layer 7
        self.layers.append(self.conv_1_192_192)
        if self.BN:
            self.layers.append(self.BN_192)
        self.layers.append(self.relu)

        # Layer 8
        self.layers.append(self.conv_1_192_class)
        if self.BN:
            self.layers.append(self.BN_class)
        self.layers.append(self.relu)

        # The Last Layers
        self.layers.append(self.avgPooling)
        self.layers.append(self.flatten)
        #self.layers.append(self.softMax)

    def __buildModelA(self):

        if self.dropOut:
            self.layers.append(self.dropOut_2)

        # Layer 1
        self.layers.append(self.conv_5_Input_96_1)
        if self.BN:
            self.layers.append(self.BN_96)
        self.layers.append(self.relu)

        # Max Pooling
        self.layers.append(self.maxP)
        if self.dropOut:
            self.layers.append(self.dropOut_5)

        # Layer 2
        self.layers.append(self.conv_5_96_192_1)
        if self.BN:
            self.layers.append( self.BN_192)
        self.layers.append(self.relu)

        # Max Pooling
        self.layers.append(self.maxP)
        if self.dropOut:
            self.layers.append(self.dropOut_5)

    def __buildModel_stridedA(self):

        if self.dropOut:
            self.layers.append(self.dropOut_2)

        # Layer 1
        self.layers.append(self.conv_5_Input_96_2)
        if self.BN:
            self.layers.append(self.BN_96)
        self.layers.append( self.relu)
        if self.dropOut:
            self.layers.append( self.dropOut_5)

        # Layer 2dropOut_2
        self.layers.append( self.conv_5_96_192_2)
        if self.BN:
            self.layers.append( self.BN_192)
        self.layers.append( self.relu)
        if self.dropOut:
            self.layers.append( self.dropOut_5)

    def __buildModel_ConvPoolA(self):

        if self.dropOut:
            self.layers.append( self.dropOut_2)

        # Layer 1
        self.layers.append( self.conv_5_Input_96_1)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append( self.relu)

        # Layer 2
        self.layers.append( self.conv_5_96_96_1)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append( self.relu)

        # Max Pooling
        self.layers.append( self.maxP)
        if self.dropOut:
            self.layers.append(self.dropOut_5)

        # Layer 3
        self.layers.append( self.conv_5_96_192_1)
        if self.BN:
            self.layers.append( self.BN_192)
        self.layers.append(self.relu)

        # Layer 4
        self.layers.append( self.conv_5_192_192_1)
        if self.BN:
            self.layers.append( self.BN_192)
        self.layers.append( self.relu)

        # Max Pooling
        self.layers.append(self.maxP)
        if self.dropOut:
            self.layers.append( self.dropOut_5)

    def __buildModel_AllA(self):

        if self.dropOut:
            self.layers.append( self.dropOut_2)

        # Layer 1
        self.layers.append( self.conv_5_Input_96_1)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append( self.relu)

        # Pooling
        self.layers.append( self.conv_5_96_96_2)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append( self.relu)
        if self.dropOut:
            self.layers.append( self.dropOut_5)

        # Layer 2
        self.layers.append( self.conv_5_96_192_1)
        if self.BN:
            self.layers.append(self.BN_192)
        self.layers.append(self.relu)

        # Max Pooling
        self.layers.append( self.conv_5_192_192_2)
        if self.BN:
            self.layers.append( self.BN_192)
        self.layers.append(self.relu)
        if self.dropOut:
            self.layers.append( self.dropOut_5)

    def __buildModelB(self):

        if self.dropOut:
            self.layers.append( self.dropOut_2)

        # Layer 1
        self.layers.append( self.conv_5_Input_96_1)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append(self.relu)

        # Layer 2
        self.layers.append( self.conv_1_96_96_1)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append( self.relu)

        # Max Pooling
        self.layers.append( self.maxP)
        if self.dropOut:
            self.layers.append(self.dropOut_5)

        # Layer 3
        self.layers.append(self.conv_5_96_192_1)
        if self.BN:
            self.layers.append( self.BN_192)
        self.layers.append(self.relu)

        # Layer 4
        self.layers.append( self.conv_1_192_192_1)
        if self.BN:
            self.layers.append( self.BN_192)
        self.layers.append( self.relu)

        # Max Pooling
        self.layers.append(self.maxP)
        if self.dropOut:
            self.layers.append( self.dropOut_5)

    def __buildModel_stridedB(self):

        if self.dropOut:
            self.layers.append(self.dropOut_2)

        # Layer 1
        self.layers.append(self.conv_5_Input_96_1)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append( self.relu)

        # Layer 2
        self.layers.append( self.conv_1_96_96_2)
        if self.BN:
            self.layers.append(self.BN_96)
        self.layers.append(self.relu)
        if self.dropOut:
            self.layers.append(self.dropOut_5)

        # Layer 3
        self.layers.append(self.conv_5_96_192_1)
        if self.BN:
            self.layers.append( self.BN_192)
        self.layers.append( self.relu)

        # Layer 4
        self.layers.append( self.conv_1_192_192_2)
        if self.BN:
            self.layers.append(self.BN_192)
        self.layers.append( self.relu)
        if self.dropOut:
            self.layers.append( self.dropOut_5)

    def __buildModel_ConvPoolB(self):

        if self.dropOut:
            self.layers.append( self.dropOut_2)

        # Layer 1
        self.layers.append( self.conv_5_Input_96_1)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append( self.relu)

        # Layer 2
        self.layers.append( self.conv_1_96_96_1)
        if self.BN:
            self.layers.append(self.BN_96)
        self.layers.append(self.relu)

        # Layer 3
        self.layers.append(self.conv_1_96_96_1)
        if self.BN:
            self.layers.append(self.BN_96)
        self.layers.append( self.relu)

        # Max Pooling
        self.layers.append( self.maxP)
        if self.dropOut:
            self.layers.append(self.dropOut_5)

        # Layer 4
        self.layers.append(self.conv_5_96_192_1)
        if self.BN:
            self.layers.append( self.BN_192)
        self.layers.append( self.relu)

        # Layer 5
        self.layers.append(self.conv_1_192_192_1)
        if self.BN:
            self.layers.append( self.BN_192)
        self.layers.append(self.relu)

        # Layer 6
        self.layers.append(self.conv_1_192_192_1)
        if self.BN:
            self.layers.append(self.BN_192)
        self.layers.append( self.relu)

        # Max Pooling
        self.layers.append( self.maxP)
        if self.dropOut:
            self.layers.append( self.dropOut_5)

    def __buildModel_AllB(self):

        if self.dropOut:
            self.layers.append( self.dropOut_2)

        # Layer 1
        self.layers.append( self.conv_5_Input_96_1)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append( self.relu)

        # Layer 2
        self.layers.append( self.conv_1_96_96_1)
        if self.BN:
            self.layers.append(self.BN_96)
        self.layers.append( self.relu)

        #  Pooling
        self.layers.append( self.conv_1_96_96_2)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append( self.relu)
        if self.dropOut:
            self.layers.append( self.dropOut_5)

        # Layer 3
        self.layers.append( self.conv_5_96_192_1)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append( self.relu)

        # Layer 4
        self.layers.append( self.conv_1_192_192_1)
        if self.BN:
            self.layers.append( self.BN_192)
        self.layers.append( self.relu)

        # Pooling
        self.layers.append( self.conv_1_192_192_2)
        if self.BN:
            self.layers.append(self.BN_192)
        self.layers.append(self.relu)
        if self.dropOut:
            self.layers.append( self.dropOut_5)

    def __buildModelC(self):

        if self.dropOut:
            self.layers.append( self.dropOut_2)

        # Layer 1
        self.layers.append( self.conv_3_Input_96_1)
        if self.BN:
            self.layers.append(self.BN_96)
        self.layers.append(self.relu)

        # Layer 2
        self.layers.append(self.conv_3_96_96_1)
        if self.BN:
            self.layers.append(self.BN_96)
        self.layers.append(self.relu)

        # Max Pooling
        self.layers.append(self.maxP)
        if self.dropOut:
            self.layers.append( self.dropOut_5)

        # Layer 3
        self.layers.append( self.conv_3_96_192_1)
        if self.BN:
            self.layers.append(self.BN_192)
        self.layers.append( self.relu)

        # Layer 4
        self.layers.append(self.conv_3_192_192_1)
        if self.BN:
            self.layers.append( self.BN_192)
        self.layers.append( self.relu)

        # Max Pooling
        self.layers.append( self.maxP)
        if self.dropOut:
            self.layers.append(self.dropOut_5)

    def __buildModel_stridedC(self):

        if self.dropOut:
            self.layers.append( self.dropOut_2)

        # Layer 1
        self.layers.append( self.conv_3_Input_96_1)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append( self.relu)

        # Layer 2
        self.layers.append( self.conv_3_96_96_2)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append(self.relu)
        if self.dropOut:
            self.layers.append(self.dropOut_5)

        # Layer 3
        self.layers.append(self.conv_3_96_192_1)
        if self.BN:
            self.layers.append( self.BN_192)
        self.layers.append( self.relu)

        # Layer 4
        self.layers.append(self.conv_3_192_192_2)
        if self.BN:
            self.layers.append( self.BN_192)
        self.layers.append( self.relu)
        if self.dropOut:
            self.layers.append( self.dropOut_5)

    def __buildModel_ConvPoolC(self):

        if self.dropOut:
            self.layers.append( self.dropOut_2)

        # Layer 1
        self.layers.append(self.conv_3_Input_96_1)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append(self.relu)

        # Layer 2
        self.layers.append(self.conv_3_96_96_1)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append( self.relu)

        # Layer 3
        self.layers.append( self.conv_3_96_96_1)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append( self.relu)

        # Max Pooling
        self.layers.append( self.maxP)
        if self.dropOut:
            self.layers.append(self.dropOut_5)

        # Layer 4
        self.layers.append(self.conv_3_96_192_1)
        if self.BN:
            self.layers.append( self.BN_192)
        self.layers.append( self.relu)

        # Layer 5
        self.layers.append( self.conv_3_192_192_1)
        if self.BN:
            self.layers.append( self.BN_192)
        self.layers.append(self.relu)

        # Layer 6
        self.layers.append( self.conv_3_192_192_1)
        if self.BN:
            self.layers.append(self.BN_192)
        self.layers.append( self.relu)

        # Max Pooling
        self.layers.append( self.maxP)
        if self.dropOut:
            self.layers.append( self.dropOut_5)

    def __buildModel_AllC(self):

        if self.dropOut:
            self.layers.append(self.dropOut_2)

        # Layer 1
        self.layers.append( self.conv_3_Input_96_1)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append( self.relu)

        # Layer 2
        self.layers.append( self.conv_3_96_96_1)
        if self.BN:
            self.layers.append( self.BN_96)
        self.layers.append( self.relu)

        # Layer 3
        self.layers.append( self.conv_3_96_96_2)
        if self.BN:
            self.layers.append(self.BN_96)
        self.layers.append(self.relu)
        if self.dropOut:
            self.layers.append(self.dropOut_5)

        # Layer 4
        self.layers.append( self.conv_3_96_192_1)
        if self.BN:
            self.layers.append(self.BN_192)
        self.layers.append(self.relu)

        # Layer 5
        self.layers.append( self.conv_3_192_192_1)
        if self.BN:
            self.layers.append( self.BN_192)
        self.layers.append( self.relu)

        # Layer 6
        self.layers.append( self.conv_3_192_192_2)
        if self.BN:
            self.layers.append(self.BN_192)
        self.layers.append( self.relu)
        if self.dropOut:
            self.layers.append(self.dropOut_5)