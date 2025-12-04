"""Simple example of using the prefetcher with PyTorch training."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import sys
from pathlib import Path as PathLib

# Add parent directory to path for imports
sys.path.insert(0, str(PathLib(__file__).parent.parent))

# Enable prefetcher hooks
from collector.dataloader_hook import enable_hook
enable_hook("traces.db")


class SimpleImageDataset(Dataset):
    """Simple dataset that creates dummy files on the fly to test prefetching."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True) # Ensure dir exists
        
        # Try to find image files
        self.image_files = sorted(list(self.data_dir.glob("*.jpg")))[:100]
        
        if len(self.image_files) == 0:
            print(f"No images found in {data_dir}, using dummy file mode")
            self.use_dummy = True
            self.num_samples = 100
        else:
            self.use_dummy = False
            print(f"Found {len(self.image_files)} images")
    
    def __len__(self):
        if self.use_dummy:
            return self.num_samples
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if self.use_dummy:
            # 1. GENERATE A DUMMY FILE PATH
            # We use absolute path so the tracer captures the full location
            dummy_path = self.data_dir / f"sample_{idx}.bin"
            
            # 2. CREATE IT IF MISSING (Simulate the file existing on disk)
            if not dummy_path.exists():
                with open(dummy_path, 'wb') as f:
                    f.write(b'\0' * 1024) # Write 1KB of zeros
            
            # 3. OPEN IT (This triggers the DataLoaderHook!)
            with open(dummy_path, 'rb') as f:
                _ = f.read()
                
            # Return dummy tensor
            return torch.randn(3, 224, 224), 0

        image_path = self.image_files[idx]
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        image = transform(image)
        return image, 0


def main():
    """Run a simple training loop."""
    dataset = SimpleImageDataset("./images")  # Adjust path as needed
    dataloader = DataLoader(dataset, batch_size=8, num_workers=2, shuffle=True)
    
    print("Starting training...")
    print("(File accesses will be traced and stored in traces.db)")
    
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}/3")
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Simulate training step
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}")
        
        print(f"  Epoch {epoch + 1} complete")
    
    print("\nTraining complete!")
    print("Traces saved to: traces.db")
    print("\nNext steps:")
    print("1. Train predictor: python -m predictor.train --db traces.db")
    print("2. Start prefetcher: python -m prefetcher.daemon --config config.yaml")
    print("3. Run training again to see speedup")


if __name__ == "__main__":
    main()

