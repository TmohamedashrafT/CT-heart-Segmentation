import matplotlib.pyplot as plt
import torch
import albumentations as A
def plot_pred(image,gt_mask,pr_mask):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image.transpose(1, 2, 0))  # convert CHW -> HWC
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask.transpose(1, 2, 0)) # just squeeze classes dim, because we have only one class
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pr_mask.squeeze().detach().cpu().numpy()) # just squeeze classes dim, because we have only one class
    plt.title("Prediction")
    plt.axis("off")
    plt.show()
    
def diceLoss(inputs, targets, smooth=0):
  intersection = (inputs * targets).sum()
  dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
  return  dice

def load_weights(weights_path,model,device,optimizer = None):

  ckpt = torch.load(weights_path, map_location = device)
  model.load_state_dict(ckpt['weights'])
  if optimizer is not None:
    optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, ckpt['Dicecoef'], ckpt['epoch']
  del ckpt
  return model

def predict(model, img_path,mask_path):
  img = cv2.imread(img_path)
  img = (img.transpose((2, 0, 1))[::-1]/255.0).astype(np.float32)
  with torch.no_grad():
    pred = model(img)
  pred = (pred > 0.5).float()
  plot_pred(img,mask,pred)
  
def aug_imgs(image, mask,img_size,train=True):
  if train:
    transform = A.Compose([
    A.Resize(img_size,img_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=45,p=0.5)
    ])
  else:
    transform = A.Compose([
    A.Resize(img_size,img_size),
    ])
  transformed = transform(image=image, mask=mask)
  return transformed['image'],transformed['mask']
