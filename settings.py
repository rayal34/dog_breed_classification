import os

TRAIN_DIR = 'train_torch'
VAL_DIR = 'val_torch'


class ModelDir:

    model_path = 'models/'

    def __init__(self, base_dir):

        self.train = os.path.join(base_dir, TRAIN_DIR)
        self.val = os.path.join(base_dir, VAL_DIR)