import torch
import torch.nn as nn
import torch_pruning as tp
from .mulsormer import MulSormer
from .retinexnet import RetinexNet
import time

class EnhancementPipeline(nn.Module):
    def __init__(self, stage1_weights=None, stage2_weights=None, device='cpu', prune_ratio=0.4):
        super(EnhancementPipeline, self).__init__()
        self.stage1 = MulSormer(base_channels=16)
        self.stage2 = RetinexNet()
        self.device = device
        self.to(device)
        
        # Load pretrained weights if available
        if stage1_weights:
            self.stage1.load_state_dict(torch.load(stage1_weights, map_location=device))
        if stage2_weights:
            self.stage2.load_state_dict(torch.load(stage2_weights, map_location=device))
        
        # Apply pruning if requested
        if prune_ratio > 0:
            self.stage1 = self._prune_mulsormer(prune_ratio)
        
        self.eval()

    def _prune_mulsormer(self, prune_ratio):
        model = self.stage1
        model.eval()
        example_inputs = torch.randn(1, 3, 256, 256).to(self.device)

        importance = tp.importance.MagnitudeImportance(p=1)
        ignored_layers = [model.final_conv]

        pruner = tp.pruner.MetaPruner(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            pruning_ratio=prune_ratio,
            ignored_layers=ignored_layers,
        )

        # Perform pruning and return the updated model
        pruner.step()
        model.eval()  # Rebuild pruned graph

        # Force garbage collection to release pruned weights
        import gc
        gc.collect()

        # Log channel reduction
        print(f"✅ Pruned MulSormer: {prune_ratio*100:.1f}% channel sparsity applied")
        return model

    def forward(self, x):
        # Safe autocast for GPU only
        if 'cuda' in self.device:
            context = torch.amp.autocast(device_type='cuda')
        else:
            from contextlib import nullcontext
            context = nullcontext()

        with context:
            deweathered = self.stage1(x)
            enhanced = self.stage2(deweathered)
        return enhanced


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")

    # Initialize model
    pipeline = EnhancementPipeline(device=device, prune_ratio=0.4)

    # Parameter count before pruning
    full_model = MulSormer(base_channels=16)
    params_before = sum(p.numel() for p in full_model.parameters() if p.requires_grad)

    # Parameter count after pruning
    params_after = sum(p.numel() for p in pipeline.stage1.parameters() if p.requires_grad)

    # Dummy input
    x = torch.randn(1, 3, 256, 256).to(device)

    # Measure inference speed
    start = time.time()
    with torch.no_grad():
        y = pipeline(x)
    elapsed = time.time() - start

    print(f"\nOutput shape: {y.shape}")
    print(f"Parameters before pruning: {params_before:,}")
    print(f"Parameters after pruning: {params_after:,} (~{100*(1-params_after/params_before):.1f}% reduction)")
    print(f"Inference time: {elapsed:.2f}s")
