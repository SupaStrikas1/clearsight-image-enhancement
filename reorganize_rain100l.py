import os
import shutil
import random

def reorganize_rain100l(base_dir, input_sub='input', target_sub='target', split_ratios=(0.8, 0.1, 0.1)):
    """
    Reorganizes Rain100L dataset from input/target to train/val/test with rain/norain subfolders.
    Args:
        base_dir: Path to rain100l folder (e.g., 'data/rain100l')
        input_sub: Subfolder with rainy images (default: 'input')
        target_sub: Subfolder with GT images (default: 'target')
        split_ratios: Tuple of (train, val, test) ratios, summing to 1.0
    """
    # Get list of files
    input_files = sorted(os.listdir(os.path.join(base_dir, input_sub)))
    target_files = sorted(os.listdir(os.path.join(base_dir, target_sub)))
    
    # Verify pairing
    assert len(input_files) == len(target_files), "Mismatch in input/target files!"
    for in_file, gt_file in zip(input_files, target_files):
        assert in_file == gt_file, f"Filename mismatch: {in_file} vs {gt_file}"

    # Shuffle indices for random split
    indices = list(range(len(input_files)))
    random.seed(42)  # For reproducibility
    random.shuffle(indices)
    
    num_total = len(indices)
    num_train = int(num_total * split_ratios[0])
    num_val = int(num_total * split_ratios[1])
    num_test = num_total - num_train - num_val

    # Define splits
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    # Move files
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    for split, idx_list in splits.items():
        for idx in idx_list:
            # Copy input (rainy) to rain/
            src_input = os.path.join(base_dir, input_sub, input_files[idx])
            dst_input = os.path.join(base_dir, split, 'rain', input_files[idx])
            shutil.copy(src_input, dst_input)
            # Copy target (GT) to norain/
            src_target = os.path.join(base_dir, target_sub, target_files[idx])
            dst_target = os.path.join(base_dir, split, 'norain', target_files[idx])
            shutil.copy(src_target, dst_target)
    
    print(f"Reorganized {base_dir}: Train={num_train}, Val={num_val}, Test={num_test}")

if __name__ == '__main__':
    base_dir = 'data/rain100l'
    reorganize_rain100l(base_dir)