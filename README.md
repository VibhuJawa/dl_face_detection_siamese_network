# Face Detection using Siamese Network

This repo explores face detection using Siamese Network (One Shot Learning) using contrastive loss. 
## Without Data Augmentation
###	Commands to run: 
    python p1a.py --save  "model_p1a_non_augment" --augment "N"
    
##	Hyper Parameters:
    Epoch=4
    Batch Size=20
    
## Reasoning:
Without augmentation, we suffer from severe overfitting, the training accuracy goes to 99 percent if we keep on increasing epochs while testing accuracy decreases back to 50 percent which is the same as random classification. 

So, I chose less epochs and I got an accuracy of:
  *	53 % on testing images
  *	71 % on training images

##	Hyper Parameters with Data Augmentationg:	
    Epoch=10
    Batch Size=20
