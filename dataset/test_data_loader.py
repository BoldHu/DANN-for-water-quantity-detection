import unittest
from data_loader import GetLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_root = "D:\python_project\WWTP soft-sensing\Data\X_kla120.mat"
        self.data_label_root = "D:\python_project\WWTP soft-sensing\Data\EQvec_kla120.mat"
        self.transform = None
        self.loader = GetLoader(self.data_root, self.data_label_root, self.transform)
        
    def test_get_item(self):
        item = 0
        d, l = self.loader.__getitem__(item)
        self.assertListEqual(d.tolist(), self.loader.data_df.iloc[item, :].tolist())
        self.assertListEqual(l.tolist(), self.loader.data_label_df.iloc[item, :].tolist())

    def test_len(self):
        self.assertEqual(len(self.loader), len(self.loader.data_df))
        
if __name__ == '__main__':
    unittest.main()