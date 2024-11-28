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

def extract_feature_directions(sae):
    # Extract the encoder weights (feature directions)
    # If your encoder is defined as nn.Linear(input_dim, hidden_dim),
    # then sae.encoder.weight has shape [hidden_dim, input_dim]
    # We transpose the weights to get them in the correct shape: [input_dim, hidden_dim]
    feature_directions = sae.encoder.weight.data.cpu().numpy().T  # Shape: [input_dim, hidden_dim]
    return feature_directions

# Step 3: Extract SAE Activations (A)
def extract_sae_activations(
    sae, dataloader, images, clip_embedder, device, accelerator, output_dir="activations"
):
    """
    Each process writes activations to a separate .npy file for simplicity.
    """
    sae.eval()
    clip_embedder.eval()

    rank = accelerator.state.process_index
    local_activations = []

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"activations_rank_{rank}.npy")

    with torch.no_grad():
        for behav, _, _, _ in dataloader:
            # Extract the image indices
            image_idx = behav[:, 0, 0].cpu().long().numpy()
            images_batch = torch.tensor(images[image_idx], dtype=torch.float32).to(device)

            with torch.cuda.amp.autocast():
                clip_target = clip_embedder(images_batch)
                _, encoded = sae(clip_target)

            # Append activations to buffer
            local_activations.append(encoded.cpu().numpy())

            # Write incrementally to avoid memory buildup
            if len(local_activations) >= 10:
                append_to_numpy(file_path, local_activations)
                local_activations = []

    # Write any remaining activations
    if local_activations:
        append_to_numpy(file_path, local_activations)

    print(f"Rank {rank}: Activations saved to {file_path}")
    accelerator.wait_for_everyone()


def append_to_numpy(file_path, data):
    """
    Append data to a .npy file. If file doesn't exist, create it.
    """
    data = np.vstack(data)
    print(f"Appending data of shape {data.shape} to {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Creating new file at {file_path}")
        np.save(file_path, data)
        print(f"Saved initial data of shape {data.shape}")
    else:
        print(f"Loading existing data from {file_path}")
        existing_data = np.load(file_path, mmap_mode="r+")
        
        combined_data = np.vstack([existing_data, data])
        
        np.save(file_path, combined_data)
        print(f"Successfully saved combined data to {file_path}")

def merge_numpy_files(input_dir, output_file="merged_activations.npy"):
    """
    Merge all rank-specific .npy files into a single file.
    """
    all_activations = []
    for rank_file in sorted(os.listdir(input_dir)):
        if rank_file.startswith("activations_rank_"):
            rank_data = np.load(os.path.join(input_dir, rank_file), mmap_mode="r+")
            all_activations.append(rank_data)

    # Concatenate and save to a single file
    all_activations = np.vstack(all_activations)
    np.save(output_file, all_activations)
    print(f"Merged activations saved to {output_file}")        

# Step 4: Perform Spherical K-Means
def spherical_kmeans(features, activations, k_range, tau, n_init=10):
    # features: normalized feature directions, shape [input_dim, hidden_dim]
    # activations: average activations per feature, shape [hidden_dim]
    # We need to transpose features to have shape [hidden_dim, input_dim]
    normalized_features = features.T  # Shape: [hidden_dim, input_dim]

    best_k, best_labels = None, None

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=42)
        cluster_labels = kmeans.fit_predict(normalized_features, sample_weight=activations)

        # Compute activation mass per cluster
        activation_mass = []
        for cluster_idx in range(k):
            cluster_mask = cluster_labels == cluster_idx
            cluster_activation_mass = activations[cluster_mask].sum()
            activation_mass.append(cluster_activation_mass)

        # Debug activation sums
        print(f"K={k}, Activation Masses: {activation_mass}")

        # Ensure all clusters meet the activation threshold
        if all(am >= tau for am in activation_mass):
            best_k, best_labels = k, cluster_labels
        else:
            break  # Stop if any cluster fails the activation mass threshold

    return {
        "best_k": best_k,
        "cluster_labels": best_labels
    }

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np


def incremental_spherical_kmeans_with_mass(
    hdf5_path, 
    features, 
    k_range, 
    tau, 
    batch_size=256, 
    n_init=10
):
    """
    Perform spherical K-Means clustering incrementally with activation mass thresholds.

    Parameters:
        hdf5_path (str): Path to HDF5 file containing activations.
        features (ndarray): Normalized feature directions, shape [input_dim, hidden_dim].
        k_range (range): Range of cluster numbers to evaluate.
        tau (float): Minimum activation mass threshold for each cluster.
        batch_size (int): Number of samples to process per chunk.
        n_init (int): Number of K-Means initializations.

    Returns:
        dict: Containing `best_k` and `cluster_labels` for the optimal clustering.
    """
    # Normalize features for spherical K-Means
    normalized_features = normalize(features.T, axis=1, norm="l2")  # Shape: [hidden_dim, input_dim]

    best_k, best_labels = None, None

    # Iterate over cluster sizes in k_range
    for k in k_range:
        print(f"Trying k = {k}")

        # Initialize KMeans
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=42)

        # Incrementally fit K-Means using HDF5 activations
        with h5py.File(hdf5_path, "r") as h5f:
            activations_dataset = h5f["activations"]
            num_samples = activations_dataset.shape[0]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                chunk = activations_dataset[start_idx:end_idx]

                # Normalize chunk for spherical K-Means
                normalized_chunk = normalize(chunk, axis=1, norm="l2")
                kmeans.partial_fit(normalized_chunk)  # Incremental fitting

        # Predict cluster labels for normalized features
        cluster_labels = kmeans.predict(normalized_features)

        # Compute activation mass per cluster
        activation_mass = []
        for cluster_idx in range(k):
            cluster_mask = cluster_labels == cluster_idx
            cluster_activation_mass = normalized_features[cluster_mask].sum(axis=0)
            activation_mass.append(cluster_activation_mass)

        print(f"K={k}, Activation Masses: {activation_mass}")

        # Check if all clusters meet the activation mass threshold
        if all(am >= tau for am in activation_mass):
            best_k, best_labels = k, cluster_labels
        else:
            break  # Stop if any cluster fails the activation mass threshold

    return {
        "best_k": best_k,
        "cluster_labels": best_labels
    }


# Step 5: Visualization
def visualize_clusters(hdf5_path, kmeans, output_path="spherical_kmeans_visualization.png"):
    from sklearn.decomposition import PCA

    with h5py.File(hdf5_path, "r") as h5f:
        activations_dataset = h5f["activations"]
        activations = activations_dataset[:]

    # Normalize activations before PCA
    normalized_activations = normalize(activations, axis=1, norm="l2")

    # Reduce dimensionality with PCA
    pca = PCA(n_components=2)
    reduced_activations = pca.fit_transform(normalized_activations)

    # Assign cluster labels
    cluster_labels = kmeans.predict(normalized_activations)

    # Scatter plot
    plt.figure(figsize=(10, 7))
    for cluster in np.unique(cluster_labels):
        cluster_points = reduced_activations[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")
    plt.title("Spherical K-Means Cluster Visualization")
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
hdf5_path = "activations.h5"

print("Before extracting SAE activations....")
extract_sae_activations(sae, test_dl, images, clip_embedder, device, accelerator, hdf5_path)

# 4. Normalize Feature Directions
normalized_feature_directions = normalize(feature_directions, axis=0, norm="l2")  # Shape: [input_dim, hidden_dim]

# 5. Compute Average Activations and Total Activation
with h5py.File(hdf5_path, "r") as h5f:
    activations_dataset = h5f["activations"]
    average_activations = activations_dataset[:].mean(axis=0)
total_activation = average_activations.sum()
tau = 0.1 * total_activation  # Set threshold to 10% of total activation

# 6. Perform Incremental Spherical K-Means
k_range = range(2, 20)  # Cluster numbers to evaluate
batch_size = 256
clustering_result = incremental_spherical_kmeans_with_mass(
    hdf5_path=hdf5_path,
    features=normalized_feature_directions,
    k_range=k_range,
    tau=tau,
    batch_size=batch_size
)

best_k = clustering_result["best_k"]
cluster_labels = clustering_result["cluster_labels"]
print(f"Optimal number of clusters: {best_k}")

# 7. Visualize Clusters
visualize_clusters(hdf5_path, clustering_result)
