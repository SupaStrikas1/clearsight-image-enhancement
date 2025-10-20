from preprocessing import get_rain_loader, get_haze_loader, get_lowlight_loader, get_combined_loader
# Test individual
rain_loader = get_rain_loader('data/rain100h', 'train', batch_size=2)
degraded, gt = next(iter(rain_loader))
print(degraded.shape, gt.shape)  # Expected: torch.Size([2, 3, 256, 256]) x2
# Test combined
combined_loader = get_combined_loader('data/rain100h', 'data/reside', 'train', batch_size=2)
degraded, gt = next(iter(combined_loader))
print(degraded.shape, gt.shape)