import argparse
import os
import pandas as pd
import numpy as np
import utils
import zipfile
from settings import ModelDir


LOGGER = utils.custom_logger('org_files')


class StructureFolder:

    def __init__(self, base_dir):
        md = ModelDir(base_dir)

        self.base_dir = base_dir
        self.raw_train_dir = os.path.join(self.base_dir, 'train')
        self.train_dir = md.train
        self.val_dir = md.val

        self.labels = self.get_labels()

    def get_labels(self):
        '''
        Read the labels csv file and convert it into a dictionary where the keys
        are the filenames and the values are the dog breeds.
        '''
        labels = pd.read_csv(os.path.join(self.base_dir, 'labels.csv'),
                             index_col=[0],
                             squeeze=True).to_dict()

        return labels

    def make_folders(self):
        '''
        Create the train and val folders
        '''

        LOGGER.info('Creating required folders...')
        required_folders = [self.train_dir, self.val_dir]
        for folder in required_folders:
            utils.make_folder(os.path.join(self.base_dir, folder))

    def structure_files(self, val_pct):
        '''
        Organize the training files in the following structure

        train
            corgi
                001.jpg
                002.jpg
            golden_retriever
                003.jpg
                004.jpg
        val
            corgi
                005.jpg
                006.jpg
        '''

        # Get all of the training files
        # For each file, look up the breed and move it to val with val_pct
        # probability, otherwise move it to train.
        files = os.listdir(self.raw_train_dir)
        LOGGER.info('Organizing files...')
        for file in files:
            img_id, _ = file.split('.')
            breed = self.labels[img_id]
            folder_location = self.val_dir if np.random.random() < val_pct else self.train_dir
            breed_location = os.path.join(self.base_dir, folder_location, breed)
            utils.make_folder(breed_location)
            os.rename(os.path.join(self.raw_train_dir, file),
                      os.path.join(breed_location, file))
        LOGGER.info('Finished organizing files')

    def check_missing_classes(self):
        '''
        Check that the train folder contains all classes
        '''

        all_breeds = set(self.labels.values())
        train_breeds = set(os.listdir(os.path.join(self.base_dir, self.train_dir)))
        missing_breeds = all_breeds.difference(train_breeds)

        if missing_breeds:
            LOGGER.info('Training folder is missing the following classes: {}'.format(missing_breeds))
            for breed in missing_breeds:
                train_breed_folder = os.path.join(self.base_dir, self.train_dir, breed)
                val_breed_folder = os.path.join(self.base_dir, self.val_dir, breed)
                val_breed_files = os.listdir(val_breed_folder)
                val_breed_cnt = len(val_breed_files)

                # Convert half of the validation files to training files.
                # The +1 ensures that if there were only 1 val file, it'll be
                # moved to train.
                move_to_train = np.random.choice(val_breed_files,
                                                 val_breed_cnt // 2 + 1)
                for file in move_to_train:
                    utils.make_folder(train_breed_folder)
                    os.rename(os.path.join(val_breed_folder, file),
                              os.path.join(train_breed_folder, file))
        else:
            LOGGER.info('No missing classes in the training folder')


def unzip(base_dir, zip_file):
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall(base_dir)
    zip_ref.close()


def main(base_dir, val_pct):

    # unzips the train and test files
    LOGGER.info('Unzipping files...')
    unzip(base_dir, os.path.join(base_dir, 'train.zip'))
    unzip(base_dir, os.path.join(base_dir, 'test.zip'))
    LOGGER.info('Finished files.')

    sf = StructureFolder(base_dir)
    sf.make_folders()
    sf.structure_files(val_pct)
    sf.check_missing_classes()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', required=True)
    parser.add_argument('-v', '--val_pct', default=.25)
    args = parser.parse_args()
    base_dir = args.base_dir
    val_pct = args.val_pct

    main(base_dir, val_pct)