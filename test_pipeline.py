from preprocessing import get_rain_loader
from models.pipeline import EnhancementPipeline
import torch

device = 'cpu'
pipeline = EnhancementPipeline(device=device, prune_ratio=0.4)
loader = get_rain_loader('C:/Abhinand/ClearSight/clearsight-image-enhancement/data/rain100h', 'train', batch_size=1)
degraded, gt = next(iter(loader))
with torch.no_grad():
    output = pipeline(degraded)
print(f"Output shape: {output.shape}")  # Expected: [1, 3, 256, 256]
pipeline.visualize_attention(degraded, pipeline.stage1.encoder[0][0], 'docs/attention_real.png')