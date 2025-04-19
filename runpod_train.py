import os
import torch
from src.train import main as train_main

def setup_runpod():
    """Setup RunPod environment"""
    # Print GPU information
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)

def main():
    """Main function for RunPod training"""
    print("Setting up RunPod environment...")
    setup_runpod()
    
    print("\nStarting training...")
    train_main()
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 