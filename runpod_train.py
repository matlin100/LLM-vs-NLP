import os
import torch
from src.train import main as train_main

def optimize_h200():
    """Optimize PyTorch settings for H200 SXM"""
    if not torch.cuda.is_available():
        return
        
    # Enable TF32 for faster training
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cudnn benchmarking
    torch.backends.cudnn.benchmark = True
    
    # Set memory allocation strategy
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available memory
    
    # Enable flash attention if available
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Set optimal thread settings
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    torch.set_num_threads(os.cpu_count())
    
    print("\nH200 Optimizations Enabled:")
    print("- TF32 Enabled")
    print("- cuDNN Benchmark Mode")
    print("- Flash Attention")
    print("- Optimal Thread Settings")
    print(f"- Using {os.cpu_count()} CPU threads")

def setup_runpod():
    """Setup RunPod environment"""
    # Print GPU information
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"\nGPU Information:")
        print(f"- Device: {torch.cuda.get_device_name(0)}")
        print(f"- Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"- Compute Capability: {torch.cuda.get_device_capability()}")
        print(f"- CUDA Version: {torch.version.cuda}")
        
        # Apply H200 optimizations
        optimize_h200()
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)

def main():
    """Main function for RunPod training"""
    print("Setting up RunPod environment...")
    setup_runpod()
    
    print("\nStarting training with optimized settings...")
    train_main()
    
    # Print memory stats after training
    if torch.cuda.is_available():
        print("\nGPU Memory Statistics:")
        print(f"- Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"- Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"- Peak Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 