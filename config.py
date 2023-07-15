from easydict import EasyDict 
import torch
cfg   = EasyDict()


### model information
cfg.train_images        = '/content/drive/MyDrive/heart_segmentation/train/images/'
cfg.train_masks         = '/content/drive/MyDrive/heart_segmentation/train/masks/'
cfg.test_images         = '/content/drive/MyDrive/heart_segmentation/test/images/'
cfg.test_masks          = '/content/drive/MyDrive/heart_segmentation/test/masks/'
cfg.img_size            = 224
cfg.batch_size          = 16
cfg.data_aug            = True
cfg.lr_init             = 1e-3
cfg.lr_end              = 1e-6
cfg.epochs              = 20
cfg.saved_weights_name  ="/content/best_heart_seg.pt"
cfg.device              = 'cuda'
cfg.encoder             = 'efficientnet-b0'
cfg.weights             = 'imagenet'
cfg.num_classes         = 1
cfg.opt_name            = 'Adam'



