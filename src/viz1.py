import os
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


# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder

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

# Step 3: Extract Encoded Features (F) and Activations (A)
def extract_sae_activations(sae, dataloader, images, clip_embedder, device):
    sae.eval()
    clip_embedder.eval()
    activations = []
    batch_size = 32  # Process in smaller batches

    with torch.no_grad():
        for behav, _, _, _ in dataloader:
            # Extract the image indices
            image_idx = behav[:, 0, 0].cpu().long().numpy()
            
            # Load images directly without storing in memory
            images_batch = torch.tensor(images[image_idx], dtype=torch.float32).to(device)

            with torch.cuda.amp.autocast():
                clip_target = clip_embedder(images_batch)
                decoded, encoded = sae(clip_target)
                activations.append(encoded.cpu().numpy())
                
            # Explicitly clear cache after each batch
            torch.cuda.empty_cache()

    activations = np.vstack(activations)
    return activations

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


# Step 5: Visualization
def visualize_clusters(features, cluster_labels):
    from sklearn.decomposition import PCA
    # features: normalized feature directions, shape [hidden_dim, input_dim]
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)  # Shape: [hidden_dim, 2]

    # Scatter plot of clusters
    plt.figure(figsize=(10, 7))
    for cluster in np.unique(cluster_labels):
        cluster_points = reduced_features[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")
    plt.title("Feature Direction Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    
    # Save the plot instead of showing it
    output_path = "feature_clusters.png"
    plt.savefig(output_path)
    plt.close()  # Close the figure to free memory
    print(f"Cluster visualization saved to {output_path}")

checkpoint_path = '/home/tazik/MindEyeV2/train_logs/final_subj01_pretrained_1sess_24bs/last_20241121-073319.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sae = load_sae_from_checkpoint(checkpoint_path, device=device)

# Load test data
num_test = 3000
subj = 1
data_path = os.getcwd()

def analyze_decoder_weights(sae_model):
    # Get the decoder weights
    decoder_weights = sae_model.decoder.weight.detach()
    # Compute L2 norm for each weight vector
    l2_norms = torch.norm(decoder_weights, dim=1).cpu().numpy()
    return l2_norms

# Assuming 'sae' is your loaded SAE model
l2_norms = analyze_decoder_weights(sae)

# Create a histogram using Plotly
fig = px.histogram(
    x=l2_norms,
    nbins=50,
    title="L2 Norm of Decoder Weights",
    labels={"x": "L2 Norm", "y": "Count"},
    template="plotly_dark",
    opacity=0.7
)
fig.update_traces(marker_color="blue")
fig.update_layout(xaxis_title="L2 Norm", yaxis_title="Count")

# Save the plot as an image
output_path = "l2_norm_decoder_weights.png"
fig.write_image(output_path)

print(f"Plot saved to {output_path}")

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

f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images']

# Extract feature directions F
feature_directions = extract_feature_directions(sae)  # Shape: [input_dim, hidden_dim]

# Extract activations A
activations = extract_sae_activations(sae, test_dl, images, clip_embedder, device)  # Shape: [num_samples, hidden_dim]

# Normalize feature directions to unit length
from sklearn.preprocessing import normalize

normalized_feature_directions = normalize(feature_directions, axis=0, norm='l2')  # Shape: [input_dim, hidden_dim]


# Since activations are per sample, but feature directions are per feature, compute the average activation for each feature across the dataset:
# Compute average activation per feature across all samples
average_activations = activations.mean(axis=0)  # Shape: [hidden_dim]

k_range = range(2, 20)  # Adjust as needed
total_activation = average_activations.sum()
tau = 0.1 * total_activation  # For example, set tau as 10% of total activation

clustering_result = spherical_kmeans(normalized_feature_directions, average_activations, k_range, tau)

best_k = clustering_result['best_k']
cluster_labels = clustering_result['cluster_labels']

print(f"Selected best_k = {best_k}")

visualize_clusters(normalized_feature_directions.T, cluster_labels) 