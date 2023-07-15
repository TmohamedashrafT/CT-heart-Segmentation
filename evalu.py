from utils import diceLoss
import segmentation_models_pytorch as smp
from tqdm import tqdm
import torch
def eval_(model,
          val_loader,
          device='cuda',
          batch_size=16,):
    pred=[]
    mask=[]
    print('test data evaluation ',end="")
    for images,masks in tqdm(val_loader):
      images,masks = images.to(device),masks.to(device)
      with torch.no_grad():
        logits = model(images)
      pred.append(logits)
      mask.append(masks)
    pred,mask=torch.cat(pred,0),torch.cat(mask,0)
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    tp, fp, fn, tn = smp.metrics.get_stats(pred.long(), mask.long(), mode="binary")
    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    dic= diceLoss(pred.long(),mask.long())
    print(" Iou  =  " ,iou.item(),"diceLoss = ",dic.item())
    return iou, dic