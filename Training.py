from torch.cuda import amp
import torch
from tqdm import tqdm
from segmentation_models_pytorch.losses import DiceLoss,TverskyLoss,SoftBCEWithLogitsLoss
from torch.optim import lr_scheduler
from config import cfg
from Dataset_Generator import data_loader
from Unet_model import *
from evalu import eval_
import gc

class Training:
  def __init__(self,
               train_images = cfg.train_images,
               train_masks  = cfg.train_masks ,
               test_images  = cfg.test_images,
               test_masks   = cfg.test_masks,
               encoder      = cfg.encoder,
               encoder_weights = cfg.weights,
               num_classes     = cfg.num_classes,
               opt_name   = cfg.opt_name,
               device     = cfg.device,
               epochs     = cfg.epochs,
               lr_init    = cfg.lr_init,
               lr_end     = cfg.lr_end,
               img_size   = cfg.img_size,
               batch_size = cfg.batch_size,
               weights_path = cfg.saved_weights_name   ):

    self.device        = device
    self.train_loader  = data_loader(train_images,train_masks,img_size,batch_size,data_aug = True, shuffle = True)
    self.test_loader   = data_loader(test_images,test_masks , img_size,batch_size,data_aug = False, shuffle = False)
    self.seg_model     = segment_model(encoder,encoder_weights,num_classes).to(device)
    self.optimizer     = self.optimizer_(opt_name = opt_name,lr = lr_init )
    self.epochs        = epochs
    self.best_dicecoef = 0
    self.scheduler     = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                            mode='min',
                                             factor=0.1,
                                             patience=7,
                                            verbose  = 1,
                                            min_lr=lr_end)
    self.weights_path = weights_path
  def train_step(self):
    self.seg_model.train()
    self.optimizer.zero_grad()
    print('################ Start  Training ################')
    for epoch in range(self.epochs):
      epoch_loss = 0.0
      scaler = amp.GradScaler()
      print(f'epoch {epoch}/{self.epochs}')
      for images,masks in tqdm(self.train_loader):
        images,masks = images.to(self.device),masks.to(self.device)
        with amp.autocast(enabled=True):
          logits       = self.seg_model(images)
          loss         = self.combine_loss(logits= logits,masks = masks,dice_weight = 1,TverskyLoss_weight = 0 \
                                  ,SoftBCEWithLogitsLoss_weight = 0  )
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        self.optimizer.zero_grad()
        #self.scheduler.step(loss)
        epoch_loss += loss.item()
      print('gpu_mem = ',torch.cuda.memory_reserved()/1E9,'loss = ',epoch_loss)
      iou,Dicecoef = eval_(self.seg_model,self.test_loader)
      self.save_model(Dicecoef,iou,epoch)
      torch.cuda.empty_cache()
      gc.collect()

  def save_model(self,Dicecoef,iou,epoch):
    if Dicecoef>self.best_dicecoef:
      ckpt ={'weights':self.seg_model.state_dict(),
           'optimizer':self.optimizer.state_dict(),
           'val_iou':iou,
           'Dicecoef':Dicecoef,
            'epoch':epoch,
            }
      self.best_dicecoef=Dicecoef
      torch.save(ckpt, self.weights_path)
      del ckpt

  def optimizer_(self,opt_name='RMSProp',lr=0.001,momentum=0.9,weight_decay=1e-5):
      if   opt_name=='SGD':
          optimizer = torch.optim.SGD(self.seg_model.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
      elif opt_name=='Adam':
          optimizer = torch.optim.Adam(self.seg_model.parameters(), lr=lr, betas=(momentum,0.999),weight_decay=weight_decay)
      elif opt_name=='RMSProp':
          optimizer = torch.optim.RMSprop(self.seg_model.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
      else:
          raise NotImplementedError(f'optimizer {opt_name} not implemented')
      return optimizer
  def combine_loss(self,logits,masks,dice_weight=1,TverskyLoss_weight=0,SoftBCEWithLogitsLoss_weight=0):
     return dice_weight*DiceLoss(mode='binary')(logits,masks)\
      + TverskyLoss_weight * TverskyLoss(mode='binary')(logits,masks)\
      + SoftBCEWithLogitsLoss_weight*SoftBCEWithLogitsLoss()(logits,masks)