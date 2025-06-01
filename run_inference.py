import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

# model_e and model_d are two identical instances of your model class
def run_inference_AE(test_data_numpy, test_label_numpy, num_classes,
                  model_e, model_d, gpu_index,
                  batch_size=64, timeout=50, bottleNeckDim = 8192):
    device = torch.device(f"cuda:{gpu_index}")
    print(f"Using device: {device}")
    model_e.to(device)  # Move the model to the GPU
    model_e.eval()  # Set the model to evaluation mode
    model_d.to(device)  # Move the model to the GPU
    model_d.eval()  # Set the model to evaluation mode

    # build test dataloader from the numpy array
    test_data = torch.tensor(test_data_numpy, dtype=torch.float32)
    test_labels = torch.tensor(test_label_numpy, dtype=torch.long)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_latents = []
    criterion = nn.MSELoss(reduction='sum')
    reconstruction_loss = 0
    shape_checked = False
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader, start=1):
            images, labels = images.to(device), labels.to(device)  # Move data to the GPU
            latents = model_e.encode(images)
            
            print("latents shape:", latents.shape)
            # check latents shape not too large
            if not shape_checked:
                latents_orig_shape = latents.shape
                latents = latents.view(latents.shape[0], -1)
                if latents.shape[1] > bottleNeckDim:
                    raise ValueError(f"Latents shape is too large: {latents.shape}. Expected less than {bottleNeckDim}.")    
                latents = latents.view(latents_orig_shape)
                shape_checked = True
            
            outputs = model_d.decode(latents)
            
            # compute reconstruction loss
            loss = criterion(outputs, images)
            reconstruction_loss += loss.item()
            
            all_latents.append(latents.cpu().numpy())
                
        reconstruction_loss = reconstruction_loss / len(test_loader.dataset)
        
        # sample images from the latent space
        # mean and std of all_latents
        all_latents = np.concatenate(all_latents, axis=0)
        mean_latents = np.mean(all_latents, axis=0)
        std_latents = np.std(all_latents, axis=0)
        
        # sample 5 random latents
        random_latents = np.random.normal(mean_latents, std_latents, (all_latents[:5].shape))
        # reconstruct the images from the latents
        random_latents = torch.tensor(random_latents, dtype=torch.float32).to(device)
        sampled_images = model_d.decode(random_latents)
        sampled_images = sampled_images.cpu().numpy()
        # save the reconstructed images, optional
    
    # release gpu memory
    torch.cuda.empty_cache()
