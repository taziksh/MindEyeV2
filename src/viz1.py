import os
from datetime import timedelta

import sys
import torch
from models import SAE, MindEyeModule, RidgeRegression, BrainNetwork, BrainDiffusionPrior, PriorNetwork
import webdataset as wds
import matplotlib.pyplot as plt
import random
import h5py
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import plotly.express as px
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder

torch.backends.cuda.matmul.allow_tf32 = True

from accelerate import Accelerator, InitProcessGroupKwargs
import utils

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
### Multi-GPU config ###
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)  

num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1

# First use "accelerate config" in terminal and setup using deepspeed stage 2 with CPU offloading!

# change depending on your mixed_precision
data_type = torch.bfloat16
accelerator = Accelerator(split_batches=False, mixed_precision="bf16", kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=0.5))])
if utils.is_interactive(): # set batch size here if using interactive notebook instead of submitting job
    global_batch_size = batch_size = 8
else:
    global_batch_size = os.environ["GLOBAL_BATCH_SIZE"]
    batch_size = int(os.environ["GLOBAL_BATCH_SIZE"]) // num_devices

def my_split_by_node(urls): return urls

#Step 1
def load_sae_from_checkpoint(checkpoint_path, device='cpu'):
    import torch
    from models import (
        MindEyeModule,
        SAE,
        RidgeRegression,
        BrainNetwork,
        BrainDiffusionPrior,
        PriorNetwork,
    )
    # Include any other necessary imports from your code

    # Initialize your MindEyeModule and all submodules
    model = MindEyeModule()

    # Set up parameters (ensure these match your training configuration)
    clip_emb_dim = 1664
    hidden_dim = 1024
    n_blocks = 4
    clip_seq_dim = 256  
    use_prior = True  # Set based on your training configuration
    clip_scale = 1.0 

    # Initialize the SAE module
    sae = SAE(input_dim=clip_emb_dim, expansion_factor=64)
    model.sae = sae

    num_voxels_list = []

    #Hyperparameters, same as training run
    data_type = torch.bfloat16
    subj_list = [1]
    num_sessions = 1
    data_path = os.getcwd()
    batch_size = 1

    train_data = {}
    train_dl = {}
    num_voxels = {}
    voxels = {}
    for s in subj_list:
        print(f"WAS     TRAINED with {num_sessions} sessions")
        # if multi_subject:
        #     train_url = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{nsessions_allsubj[s-1]-1}" + "}.tar"
        # else:
        train_url = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{num_sessions-1}" + "}.tar"
        print(train_url)
        
        train_data[f'subj0{s}'] = wds.WebDataset(train_url,resampled=True,nodesplitter=my_split_by_node)\
                            .shuffle(750, initial=1500, rng=random.Random(42))\
                            .decode("torch")\
                            .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                            .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
        train_dl[f'subj0{s}'] = torch.utils.data.DataLoader(train_data[f'subj0{s}'], batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

        f = h5py.File(f'{data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
        betas = f['betas'][:]
        betas = torch.Tensor(betas).to("cpu").to(data_type)
        num_voxels_list.append(betas[0].shape[-1])
        num_voxels[f'subj0{s}'] = betas[0].shape[-1]
        voxels[f'subj0{s}'] = betas
        print(f"num_voxels for subj0{s}: {num_voxels[f'subj0{s}']}")    

    # Initialize the RidgeRegression module
    model.ridge = RidgeRegression(num_voxels_list, out_features=hidden_dim)

    # Initialize the BrainNetwork (backbone)
    model.backbone = BrainNetwork(
        h=hidden_dim,
        in_dim=hidden_dim,
        seq_len=1,
        n_blocks=n_blocks,
        clip_size=clip_emb_dim,
        out_dim=clip_emb_dim * clip_seq_dim,
        blurry_recon=False,  # Adjust if you used blurry_recon during training
        clip_scale=clip_scale,
    )

    # Initialize the diffusion_prior if used during training
    if use_prior:
        out_dim = clip_emb_dim
        depth = 6
        dim_head = 52
        heads = clip_emb_dim // 52
        timesteps = 100

        prior_network = PriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens=clip_seq_dim,
            learned_query_mode="pos_emb",
        )

        model.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        )

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Load the state_dict into the model
    model.load_state_dict(state_dict, strict=True)

    # Extract the SAE model
    sae = model.sae
    sae.to(device)
    sae.eval()

    return sae

# Step: Extract Feature Directions
def extract_feature_directions(sae):
    # Extract the encoder weights (feature directions)
    # If your encoder is defined as nn.Linear(input_dim, hidden_dim),
    # then sae.encoder.weight has shape [hidden_dim, input_dim]
    # We transpose the weights to get them in the correct shape: [input_dim, hidden_dim]
    feature_directions = sae.encoder.weight.data.cpu().numpy().T  # Shape: [input_dim, hidden_dim]
    return feature_directions

# Step 3: Extract SAE Activations
def extract_sae_activations(
    sae, dataloader, images, clip_embedder, device, accelerator, output_dir="activations"
):
    sae.eval()
    clip_embedder.eval()
    
    rank = accelerator.state.process_index
    local_activations = []

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"activations_rank_{rank}.h5")

    # Each rank writes to its own h5 file
    with h5py.File(file_path, "w") as h5f:
        dataset = None  # Will create after seeing first batch
        samples_processed = 0

        with torch.no_grad():
            for behav, _, _, _ in dataloader:
                if samples_processed >= max_samples:
                    break

                image_idx = behav[:, 0, 0].cpu().long().numpy()
                images_batch = torch.tensor(images[image_idx], dtype=torch.float32).to(device)

                with torch.cuda.amp.autocast():
                    clip_target = clip_embedder(images_batch)
                    _, encoded = sae(clip_target)

                local_activations.append(encoded.cpu().numpy())
                samples_processed += len(behav)

                # Create dataset after first batch when we know the shape
                if dataset is None and len(local_activations) > 0:
                    first_batch = local_activations[0]
                    dataset = h5f.create_dataset(
                        "activations",
                        shape=(0, *first_batch.shape[1:]),
                        maxshape=(None, *first_batch.shape[1:]),
                        dtype='float32'
                    )

                # Periodically save to h5
                if len(local_activations) >= 100:
                    batch_data = np.vstack(local_activations)
                    current_size = dataset.shape[0]
                    new_size = current_size + batch_data.shape[0]
                    dataset.resize(new_size, axis=0)
                    dataset[current_size:new_size] = batch_data
                    local_activations = []

            # Save any remaining activations
            if local_activations:
                batch_data = np.vstack(local_activations)
                current_size = dataset.shape[0]
                new_size = current_size + batch_data.shape[0]
                dataset.resize(new_size, axis=0)
                dataset[current_size:new_size] = batch_data

    print(f"Rank {rank}: Processed {samples_processed} samples")
    accelerator.wait_for_everyone()

def merge_numpy_files(input_dir, output_file="merged_activations.h5"):
    """
    Merge all rank-specific h5 files into a single file.
    """
    all_activations = []
    for rank_file in sorted(os.listdir(input_dir)):
        if rank_file.startswith("activations_rank_") and rank_file.endswith(".h5"):
            with h5py.File(os.path.join(input_dir, rank_file), "r") as h5f:
                rank_data = h5f["activations"][:]
                all_activations.append(rank_data)

    # Create the merged file
    with h5py.File(output_file, "w") as h5f:
        merged_data = np.vstack(all_activations)
        h5f.create_dataset("activations", data=merged_data)
    
    print(f"Merged activations saved to {output_file}")       

# Step 4: Perform Spherical K-Means
def spherical_kmeans_with_activation_mass(
    hdf5_path, 
    features,  # Sparse feature directions F
    k_range, 
    tau, 
    batch_size=256, 
    n_init=10
):
    """
    Perform spherical k-means and select the optimal k based on activation mass threshold.
    """
    print(f"Features shape before normalization: {features.shape}")
    
    # Normalize features to unit length
    normalized_features = normalize(features, axis=1, norm="l2")  # Shape: [input_dim, hidden_dim]
    print(f"Normalized features shape: {normalized_features.shape}")

    best_k = None
    all_labels = []  # Store all labels for visualization

    for k in k_range:
        print(f"\nTrying k = {k}")
        kmeans = MiniBatchKMeans(
            n_clusters=k, 
            init="k-means++", 
            n_init=n_init, 
            random_state=42,
            batch_size=batch_size
        )
        
        # First pass: fit the model incrementally
        with h5py.File(hdf5_path, "r") as h5f:
            activations_dataset = h5f["activations"]
            num_samples = activations_dataset.shape[0]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                chunk = activations_dataset[start_idx:end_idx]
                reshaped_chunk = chunk.reshape(-1, chunk.shape[-1])
                kmeans.partial_fit(reshaped_chunk)

        # Second pass: predict labels and compute activation mass incrementally
        cluster_activation_masses = [0] * k  # Track activation mass per cluster
        chunk_labels = []  # Store labels for the current chunk
        with h5py.File(hdf5_path, "r") as h5f:
            activations_dataset = h5f["activations"]
            num_samples = activations_dataset.shape[0]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                chunk = activations_dataset[start_idx:end_idx]  # Shape: (60, 256, 106496)

                # Flatten chunk for clustering
                reshaped_chunk = chunk.reshape(-1, chunk.shape[-1])  # Shape: (60 * 256, 106496)
                print(f"[DEBUG] Predicting Labels: Chunk shape: {reshaped_chunk.shape}")

                labels = kmeans.predict(reshaped_chunk)  # Predict labels for flattened chunk
                chunk_labels.append(labels)  # Collect labels for this chunk

                # Compute activation mass for each cluster
                for cluster_idx in range(k):
                    cluster_mask = labels == cluster_idx  # Boolean mask for the current cluster
                    print(f"[DEBUG] Cluster Mask: Cluster mask sum: {cluster_mask.sum()}, Mask shape: {cluster_mask.shape}")

                    if cluster_mask.sum() > 0:  # Avoid empty clusters
                        cluster_activation_masses[cluster_idx] += reshaped_chunk[cluster_mask].sum()

                print(f"[DEBUG] Completed Chunk: Start idx: {start_idx}, End idx: {end_idx}")

        all_labels.extend(np.concatenate(chunk_labels))
        print(f"Final Cluster Activation Masses: {cluster_activation_masses}")

        # Check if all clusters meet the activation threshold
        if all(am >= tau for am in cluster_activation_masses):
            best_k = k
            break  # Stop if we find a valid k

    return {
        "best_k": best_k,
        "cluster_labels": np.array(all_labels) if best_k else None,  # Return full labels
        "kmeans": kmeans  # Return the model for visualization
    }

# Step 5: Visualization
def visualize_clusters(hdf5_path, clustering_result, output_path=None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(hdf5_path), "spherical_kmeans_visualization.png")

    with h5py.File(hdf5_path, "r") as h5f:
        activations_dataset = h5f["activations"]
        activations = activations_dataset[:]
        print(f"[DEBUG] Loaded Activations Shape: {activations.shape}")

    # Flatten or aggregate activations
    if activations.ndim == 3:
        # Option 1: Flatten
        flattened_activations = activations.reshape(-1, activations.shape[-1])  # (60 * 256, 106496)
        print(f"[DEBUG] Flattened Activations Shape: {flattened_activations.shape}")
        normalized_activations = normalize(flattened_activations, axis=1, norm="l2")

        # If clustering labels need alignment with flattened activations
        cluster_labels = clustering_result["cluster_labels"].reshape(activations.shape[0], activations.shape[1])
        cluster_labels = cluster_labels.flatten()  # Ensure shape matches flattened activations

    elif activations.ndim == 2:
        # If already 2D, normalize directly
        normalized_activations = normalize(activations, axis=1, norm="l2")
        cluster_labels = clustering_result["cluster_labels"]

    else:
        raise ValueError(f"Unexpected activations shape: {activations.shape}")

    # Reduce dimensionality with PCA
    pca = PCA(n_components=2)
    reduced_activations = pca.fit_transform(normalized_activations)
    print(f"[DEBUG] Reduced Activations Shape (PCA): {reduced_activations.shape}")

    # Use the cluster labels from clustering_result
    unique_labels = np.unique(cluster_labels)
    print(f"[DEBUG] Unique Cluster Labels: {unique_labels}")

    # Scatter plot
    plt.figure(figsize=(10, 7))
    for cluster in unique_labels:
        cluster_points = reduced_activations[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")
    plt.title(f"Spherical K-Means Clusters (k={clustering_result['best_k']})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"Cluster visualization saved to {output_path}")


checkpoint_path = '/home/tazik/MindEyeV2/train_logs/final_subj01_pretrained_1sess_24bs/last_20241121-073319.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sae = load_sae_from_checkpoint(checkpoint_path, device=device)

# Load test data
#num_test = 3000
num_test = 10

subj = 1
data_path = os.getcwd()

def print_memory_stats(stage):
    print(f"\n[Memory Stats: {stage}]")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"Max Reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB\n")

def analyze_decoder_weights(sae_model):
    # Get the decoder weights
    decoder_weights = sae_model.decoder.weight.detach()
    
    # Compute L0, L1, and L2 norms
    l0_norms = (decoder_weights != 0).sum(dim=1).float().cpu().numpy()
    l1_norms = torch.norm(decoder_weights, p=1, dim=1).cpu().numpy()
    l2_norms = torch.norm(decoder_weights, p=2, dim=1).cpu().numpy()
    
    return l0_norms, l1_norms, l2_norms

# # Calculate all norms
# l0_norms, l1_norms, l2_norms = analyze_decoder_weights(sae)

# # Create a DataFrame from the norms
# df = pd.DataFrame({
#     'Norm Value': np.concatenate([l0_norms, l1_norms, l2_norms]),
#     'Norm Type': ['L0'] * len(l0_norms) + ['L1'] * len(l1_norms) + ['L2'] * len(l2_norms)
# })

# # Create subplots for all three norms
# fig = px.histogram(
#     data_frame=df,
#     x='Norm Value',
#     color='Norm Type',
#     nbins=50,
#     title="Distribution of Decoder Weight Norms",
#     labels={"Norm Value": "Norm Value", "count": "Count"},
#     template="plotly_dark",
#     opacity=0.7,
#     color_discrete_sequence=["red", "green", "blue"],
#     facet_col='Norm Type'
# )


# # Update layout
# fig.update_layout(
#     showlegend=False,
#     title_x=0.5,
#     height=400,
#     width=1200
# )

# # Update x-axis titles
# for i in range(3):
#     fig.update_xaxes(title_text=f"L{i} Norm", col=i+1)

# # Save the plot as an image
# output_path = "decoder_weight_norms.png"
# fig.write_image(output_path)

# print(f"Plot saved to {output_path}")

test_url = f"{data_path}/wds/subj0{subj}/new_test/" + "0.tar"
test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                    .shuffle(750, initial=1500, rng=random.Random(42))\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, drop_last=True, pin_memory=True)

# Initialize CLIP embedder
clip_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
    only_tokens=True,
).to(device).eval()

# 1. Load Images
f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images']

# 2. Extract Feature Directions
feature_directions = extract_feature_directions(sae)  # Shape: [input_dim, hidden_dim]

# 3. Save Activations Incrementally to HDF5
# hdf5_path = "activations.h5"


total_samples = 0
for behav, _, _, _ in test_dl:
    total_samples += len(behav)
print(f"Total samples in dataloader: {total_samples}")

# Then modify the max_samples in extract_sae_activations to be a percentage
max_samples = total_samples // 100  # Use 10% of the data
print(f"Will process {max_samples} samples")


hdf5_path = os.path.join(os.getcwd(), "sae_activations")

print(f"Attempting to create HDF5 file at: {hdf5_path}")
print(f"Current working directory: {os.getcwd()}")

print("Before extracting SAE activations....")

extract_sae_activations(sae, test_dl, images, clip_embedder, device, accelerator, hdf5_path)
print("After extracting SAE activations....")

# 4. Normalize Feature Directions
normalized_feature_directions = normalize(feature_directions, axis=0, norm="l2")  # Shape: [input_dim, hidden_dim]

# 5. Compute Average Activations and Total Activation

print("Performing spherical k-means...")
merged_file = os.path.join(hdf5_path, "merged_activations.h5")

if accelerator.is_main_process:
    merge_numpy_files(hdf5_path, merged_file)
accelerator.wait_for_everyone()

with h5py.File(merged_file, "r") as h5f:
    activations_dataset = h5f["activations"]
    average_activations = activations_dataset[:].mean(axis=0)
total_activation = average_activations.sum()
tau = 0.1 * total_activation  # Set threshold to 10% of total activation
print(f"Activation mass threshold, tau: {tau}")

# Incremental spherical k-means
# k_range = range(2, 20)  # Cluster numbers to evaluate

k_range = range(2, 20)

batch_size = 256
clustering_result = spherical_kmeans_with_activation_mass(
    hdf5_path=merged_file,
    features=normalized_feature_directions,  # Feature directions F from SAE
    k_range=k_range,
    tau=tau,
    batch_size=batch_size
)

best_k = clustering_result["best_k"]
cluster_labels = clustering_result["cluster_labels"]
print(f"Optimal number of clusters: {best_k}")

# Visualize Clusters
if clustering_result["best_k"] is not None:
    visualize_clusters(merged_file, clustering_result)