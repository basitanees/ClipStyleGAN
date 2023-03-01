
from losses.clip_loss import CLIPLoss

device = "cuda"
clip_models=["ViT-B/16"]
print("Loading CLIP")
clip_loss_models = {model_name: CLIPLoss(device,clip_model=model_name) for model_name in clip_models}
print("Loaded CLIP")

model = clip_loss_models["ViT-B/16"]
img_encoder = model.model.visual
