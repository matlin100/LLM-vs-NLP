import os
import torch
from train import main

if __name__ == "__main__":
    # Set environment variables for RunPod
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Optimize for H100
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable bfloat16 for faster training
        torch.set_float32_matmul_precision('high')
        
        # Set memory allocation strategy for H100
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
        
        # Enable memory efficient attention
        os.environ['USE_MEMORY_EFFICIENT_ATTENTION'] = '1'
        
        print("\nH100 Optimizations Enabled:")
        print("- TF32 Enabled")
        print("- High Precision Matrix Multiplication")
        print("- Memory Efficient Attention")
        print("- Large Memory Chunks (1024MB)")
    else:
        print("No GPU available, using CPU")
    
    # Run training with optimized parameters
    main() 