from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io,transform as sk_transform
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import  utils
from torch.autograd import Variable

import random

class random_augmetaion(object):
    """Applied the required tranformations"""

    def random_augment_image(self, face):
        ## mirror flipping

        # rotate image parmaeter:
        angle = math.radians(random.randrange(-30, +30))

        # translate image parameter
        tr_distance_x = random.randrange(-10, +10)
        tr_distance_y = random.randrange(-10, +10)
        # scaling image parameter
        scaling_val = random.uniform(0.7, 1.3)

        tform = sk_transform.SimilarityTransform(scale=scaling_val, rotation=angle, \
                                                 translation=(tr_distance_x, tr_distance_y))
        # transform probabilty
        tr_probab = random.random()
        if tr_probab <= 0.7:
            face = sk_transform.warp(face, inverse_map=tform)

        return face

    def __call__(self, sample):
        face_1, face_2 = sample['face_1'], sample['face_2']

        face_1 = self.random_augment_image(face_1)
        face_2 = self.random_augment_image(face_2)

        return {'face_1': face_1, 'face_2': face_2, 'label': sample['label']}


class ToTensor(object):
    def __call__(self, sample):
        face_1, face_2 = sample['face_1'], sample['face_2']

        # final resize the image
        face_1 = sk_transform.resize(face_1, (128, 128), mode='constant')
        # transpose because the axis are different in torch.image vs skimage image
        face_1 = face_1.transpose((2, 0, 1))

        # final resize the image
        face_2 = sk_transform.resize(face_2, (128, 128), mode='constant')
        # transpose because the axis are different in torch.image vs skimage image
        face_2 = face_2.transpose((2, 0, 1))

        label = torch.LongTensor(1, 1).zero_()
        label = sample['label']
        return {'face_1': torch.from_numpy(face_1), 'face_2': torch.from_numpy(face_2), 'label': label}

class FacePairsDataset(Dataset):
    """Face Pairs dataset."""

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data = pd.read_csv(txt_file, header=None)
        df = data[0].str[:].str.split(' ', expand=True)
        df.columns = ["face_1", "face_2", "label"]
        self.face_pair_frame = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.face_pair_frame)

    def __getitem__(self, idx):
        img_name_1 = os.path.join(self.root_dir, self.face_pair_frame.loc[idx, 'face_1'])
        img_name_2 = os.path.join(self.root_dir, self.face_pair_frame.loc[idx, 'face_2'])
        face_1 = io.imread(img_name_1)
        face_2 = io.imread(img_name_2)
        label = self.face_pair_frame.loc[idx, 'label']
        label = int(label)
        sample = {'face_1': face_1, 'face_2': face_2, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def show_faces_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    faces_1_batch, faces_2_batch = sample_batched['face_1'], sample_batched['face_2']
    print (faces_1_batch.shape)
    print (faces_2_batch.shape)
    labels_batch=sample_batched['label']
    print(labels_batch.shape)
    batch_size = len(faces_1_batch)
    im_size = faces_1_batch.size(2)
    grid_1 = utils.make_grid(faces_1_batch)
    grid_2 = utils.make_grid(faces_2_batch)
    plt.imshow(grid_1.numpy().transpose((1, 2, 0)))
    plt.imshow(grid_2.numpy().transpose((1, 2, 0)))


def curr_accuracy(test_loader,snet):
    correct = 0
    total = 0
    count = 0
    for data in test_loader:
        count = count + 1
        faces_1_batch, faces_2_batch = data['face_1'], data['face_2']
        labels_batch = data['label']

        if torch.cuda.is_available():
            input1, input2 = Variable(faces_1_batch.float().cuda()), Variable(faces_2_batch.float().cuda())
        else:
            input1, input2 = Variable(faces_1_batch.float()), Variable(faces_2_batch.float())

        outputs = snet(input1, input2)

        outputs.data[outputs.data >= 0.5] = 1
        outputs.data[outputs.data <= 0.5] = 0

        predicted = outputs.data

        total += labels_batch.size(0)
        labels_batch = (labels_batch.view(-1, 1))

        correct += (predicted.long().cpu() == labels_batch.long()).sum()

    ac = 100 * (correct / total)
    print('Accuracy of the network on the %d test images: %d %%' % (total, ac))
    return ac




