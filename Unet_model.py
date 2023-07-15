from torch import nn
import segmentation_models_pytorch as smp
class segment_model(nn.Module):
  def __init__(self,
               Encoder,
               encoder_weights,
               classes):
    super(segment_model, self).__init__()
    self.model = smp.Unet(encoder_name=Encoder,
                          encoder_weights=encoder_weights,
                          classes=classes)
  def forward(self,image):
    return self.model(image)
