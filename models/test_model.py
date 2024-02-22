import unittest
import torch
import torch.nn as nn
from model import CNNModel

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = CNNModel()

    def test_forward(self):
        input_data = torch.randn(10, 15)
        alpha = 0.5
        class_output, domain_output = self.model.forward(input_data, alpha)

        self.assertEqual(class_output.shape, torch.Size([10, 1]))
        self.assertEqual(domain_output.shape, torch.Size([10, 2]))

    def test_layers(self):
        self.assertIsInstance(self.model.feature, nn.Sequential)
        self.assertIsInstance(self.model.value_regressor, nn.Sequential)
        self.assertIsInstance(self.model.domain_classifier, nn.Sequential)

        self.assertEqual(len(self.model.feature), 4)
        self.assertEqual(len(self.model.value_regressor), 3)
        self.assertEqual(len(self.model.domain_classifier), 6)

if __name__ == '__main__':
    unittest.main()