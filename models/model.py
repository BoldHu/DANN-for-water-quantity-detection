import torch.nn as nn
from .functions import ReverseLayerF

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_linear1', nn.Linear(15, 64))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_linear2', nn.Linear(64, 128))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.value_regressor = nn.Sequential()
        self.value_regressor.add_module('v_linear1', nn.Linear(128, 64))
        self.value_regressor.add_module('v_relu1', nn.ReLU(True))
        self.value_regressor.add_module('v_linear2', nn.Linear(64, 1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(128, 64))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(64, 64))
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc3', nn.Linear(64, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.value_regressor(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
