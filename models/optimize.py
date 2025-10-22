import torch
import torch_pruning as tp
from pipeline import EnhancementPipeline

import torch_pruning as tp

def get_improved_mulsormer(model, prune_ratio=0.4):
    sub_model = model.stage1
    sub_model.eval()

    example_inputs = torch.randn(1, 3, 256, 256)
    imp = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MetaPruner(
        model=sub_model,
        example_inputs=example_inputs,
        importance=imp,
        pruning_ratio=prune_ratio,
    )

    pruner.step()
    return model


if __name__ == "__main__":
    pipeline = EnhancementPipeline(device='cpu')
    print(f"Parameters before: {sum(p.numel() for p in pipeline.stage1.parameters())}")
    pruned = get_improved_mulsormer(pipeline)
    print(f"Parameters after: {sum(p.numel() for p in pruned.stage1.parameters())}")