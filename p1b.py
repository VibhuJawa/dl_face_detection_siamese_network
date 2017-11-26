import argparse

from time import gmtime, strftime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

import data_loader as dl
import models
import Contraceptive_Loss


def get_args():
    parser = argparse.ArgumentParser(description="Main  Parser")

    parser.add_argument("--data", type=str, required=False, help="The data directory to use for training or testing.",default="lfw")
    parser.add_argument("--augment", type=str, required=False, help="Augmentation, add 'y' or 'n' ",default="y")

    parser.add_argument("--load", type=str, required=False, help="(TEST MODE) Load weights file")
    parser.add_argument("--save", type=str, required=False, help="(TRAIN MODE) Save weights file")

    args = parser.parse_args()

    return args

def p1a():
    args=get_args()

    if not args:
        print("empty inputs")
        return

    if args.save is not None:
        print "--Training Mode--"

        trans_train = transforms.Compose([dl.random_augmetaion(), dl.ToTensor()])
        trans_test = transforms.Compose([dl.ToTensor()])
        N = 20
        threshold=2


        if(args.augment=="N"):
            print("Non data augmentation Mode")
            face_train_dataset = dl.FacePairsDataset(txt_file='lfw/train.txt', root_dir='lfw/', transform=trans_test)
        else:
            print("Augmentation Mode")
            face_train_dataset = dl.FacePairsDataset(txt_file='lfw/train.txt', root_dir='lfw/', transform=trans_train)

        face_test_dataset = dl.FacePairsDataset(txt_file='lfw/test.txt', root_dir='lfw/', transform=trans_test)

        train_loader = DataLoader(dataset=face_train_dataset, batch_size=N, shuffle=True, num_workers=4)
        test_loader = DataLoader(dataset=face_test_dataset, batch_size=N, shuffle=False, num_workers=4)

        print("Loaded Data")


        print("Loss and Loading the Model")
        if torch.cuda.is_available():
            snet = models.Siamese_Net_Contrastive().cuda()
            criterion = Contraceptive_Loss.ContrastiveLoss().cuda()
        else:
            snet = models.Siamese_Net_Contrastive()
            criterion = Contraceptive_Loss.ContrastiveLoss()

        print("Optimizer ")

        optimizer = optim.Adam(snet.parameters())
        a = torch.randn(2, 3, 128, 128)
        b = torch.randn(2, 3, 128, 128)

        print(snet)
        if torch.cuda.is_available():
            o_1, o_2 = snet(Variable(a).cuda(), Variable(b).cuda())
        else:
            o_1, o_2 = snet(Variable(a), Variable(b))

        print("sanity check with 2 random variables",o_1,o_2)
        
        ac_list = []
        print("Started Training")
        for epoch in range(35):  # loop over the dataset multiple times
            for i, sample_batched in enumerate(train_loader):
                # get the inputs
                faces_1_batch, faces_2_batch = sample_batched['face_1'], sample_batched['face_2']
                labels_batch = sample_batched['label']

                # wrap them in Variable
                if torch.cuda.is_available():
                    input1, input2 = Variable(faces_1_batch.float().cuda()), Variable(faces_2_batch.float().cuda())
                    labels_batch = Variable(labels_batch.float().cuda())
                else:
                    input1, input2 = Variable(faces_1_batch.float()), Variable(faces_2_batch.float())
                    labels_batch = Variable(labels_batch.float())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output_1, output_2 = snet(input1, input2)
                loss = criterion(output_1, output_2, labels_batch)
                loss.backward()
                optimizer.step()

            ac_list.append(dl.curr_accuracy_2(test_loader,snet,threshold))
            print("Epoch number ", epoch)

        print('Finished Training')
        save_file_path = args.save
        torch.save(snet.state_dict(), save_file_path)

    else:
        print "--Testing Mode---"
        N=20
	if torch.cuda.is_available():
            snet_load = models.Siamese_Net_Contrastive().cuda()
        else:
            snet_load = models.Siamese_Net_Contrastive()
        snet_load.load_state_dict(torch.load(args.load))
        trans_test = transforms.Compose([dl.ToTensor()])

        face_test_dataset = dl.FacePairsDataset(txt_file='lfw/train.txt', root_dir='lfw/', transform=trans_test)
        test_loader = DataLoader(dataset=face_test_dataset, batch_size=N, shuffle=False, num_workers=4)
        dl.curr_accuracy_2(test_loader,snet_load,2)


if __name__ == '__main__':

    p1a()
