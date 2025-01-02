import os
import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import math

"""
This script is used to preprocess the raw jpg photographs into preprocessed (cropped, sized and normalised images)
We preprocess in a separate script so we can have all the preprocessed images saved to disk, ready for training in
multiple models later.

Please execute the main method to run the preprocessing. Make sure to update the folder paths. 
"""



    
def create_dataset_batches(dataset, batch_size=1000):
    """
    A generator function that splits the dataset into manageable batches
    This is to prevent python from loading the entire dataset into memory and crashing
    """
    total_size = len(dataset)
    num_batches = math.ceil(total_size / batch_size)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_size)
        yield Subset(dataset, range(start_idx, end_idx))
        
        
def crop_or_pad(img, target_size=1800):
    """
    This function random crops the image to 1800x1800. 
    If the image is smaller, it resizes the image to 1800x1800 using a padding of zeros
    """
    try:
        w, h = img.size
        
        if w >= target_size and h >= target_size:
            return transforms.RandomCrop((target_size, target_size))(img)
            
        # Calculate padding
        ratio = min(target_size / w, target_size / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # We resize the image to make the most of the 1800x1800 target size
        img = transforms.Resize((new_h, new_w))(img)
        
        pad_w = target_size - new_w
        pad_h = target_size - new_h
        padding = (
            pad_w // 2,  # left
            pad_h // 2,  # top
            pad_w - pad_w // 2,  # right
            pad_h - pad_h // 2   # bottom
        )
        
        # Pad out the image to 1800x1800 square shape. We do not distort the aspect ratio in this method
        return transforms.Pad(padding, fill=0)(img)  # fill with zeros makes the image black    
    except Exception as e:
        print(f"Error during padding: {str(e)}")
        return None
    
def collate_fn(batch):
    """
    Custom collate_fn to load our data correctly into dataloader
    """
    # Handles missing images
    batch = [(img, target) for img, target in batch if img is not None]
    if not batch:
        return torch.zeros((0, 3, 1800, 1800)), torch.zeros(0)
    
    images, targets = zip(*batch)
    targets = torch.tensor(targets)  # Convert targets to tensor
    
    # Return PIL images and tensor targets
    return list(images), targets

def process_dataset(in_path, out_path, outer_batch_size=1000, gpu_batch_size=32):
    # Device setup
    if not torch.cuda.is_available():
        raise Exception("GPU not available!")
    
    device = torch.device('cuda')
    torch.cuda.empty_cache()  # Clear the cache from any previous runs which may be taking up memory

    crop_tensor = transforms.Compose([
        transforms.Lambda(crop_or_pad),
        transforms.ToTensor(),
    ])
    
    # Normalisation on GPU
    normalise = transforms.Normalize(
        mean=[0.45579228, 0.4374342, 0.41491082], # Values calculated from the training dataset 
        std=[0.1361281, 0.13010998, 0.12894471]
    )

    # Load the dataset
    full_dataset = ImageFolder(root=in_path, transform=None)
    print(f"Number of images found: {len(full_dataset)}")
    

    processed_count = 0
    skipped_count = 0
    
    print("Beginning batch processing")
    for batch_num, batch_subset in enumerate(create_dataset_batches(full_dataset, outer_batch_size)):
        print(f"Processing outer batch {batch_num}")
        
        batch_loader = DataLoader(
            batch_subset,
            batch_size=gpu_batch_size,
            shuffle=False,
            num_workers=4,  
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        try:

            for inner_batch_idx, (images, labels) in enumerate(tqdm(batch_loader, desc="Processing inner batch")):
                """
                We use a nested batch processing to save memory. 
                This is to prevent the entire dataset from being loaded into memory at once.
                Pytorch native dataloaders was using too much system memory so I intervened and separately load the dataloaders 1000 images at a time
                We process the images in batches of 32 on the GPU and then save them to disk.
                We need to load the images from disk to memory using CPU
                Convert from PIL image to tensors which the GPU can handle
                All of this effort is to speed up the process by taking advantage of the GPU to normalise the images    
                """
                
                with torch.amp.autocast('cuda', dtype=torch.float16): # mixed precision to save memory  
                    # Move data to GPU after transforms, filtering out any None values
                    batch_transformed = [crop_tensor(img) for img in images]
                        
                    batch_transformed = torch.stack(batch_transformed).to(device, non_blocking=True)
                    
                    # Apply normalization after moving to GPU
                    batch_normalised = normalise(batch_transformed)
                    
                    # Process entire batch of images
                    for idx, (transformed_image, label) in enumerate(zip(batch_normalised, labels)):
                        label_dir = os.path.join(out_path, full_dataset.classes[label])
                        os.makedirs(label_dir, exist_ok=True)
                        
                        image_name = f"image_{batch_num * outer_batch_size + inner_batch_idx * gpu_batch_size + idx:06d}.jpg"
                        image_path = os.path.join(label_dir, image_name)
                        
                        try:
                            # Saving preprocessed images to disk
                            img_cpu = transformed_image.cpu()
                            rescaled_image = transforms.ToPILImage()(img_cpu)
                            rescaled_image.save(image_path, quality=95, optimize=True)
                            processed_count += 1
                            if processed_count % 100 == 0:
                                print(f"Successfully processed {processed_count} images")
                            del img_cpu, rescaled_image  # free up memory
                        except Exception as e:
                            # Skip erronous images to prevent crashing  
                            print(f"Error saving image {image_path}: {str(e)}")
                            skipped_count += 1
                    
                    # Clean up GPU memory each iteration
                    del batch_transformed, batch_normalised
                    torch.cuda.empty_cache()
                
        except RuntimeError as e:
            print(f"Error processing batch: {e}")
        

    print(f"Total images processed: {processed_count}")
    print(f"Total images skipped: {skipped_count}")


if __name__ == "__main__":
    """
    Note this script does not work in jupyter notebook format
    Give the input and output folder paths below
    Input data must be in folders labelling the data's class
    """
    input_dataset_path = r'F:\MLDS\UDA Datasets\manual_test_in'
    output_dataset_path = r'F:\MLDS\UDA Datasets\manual_test_in_pp'

    process_dataset(input_dataset_path, output_dataset_path)
    