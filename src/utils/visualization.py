import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim

def plot_umap_embeddings(features_df, output_path, n_neighbors=15, min_dist=0.1):
    """
    Runs UMAP on the hourly features and plots the result.

    Args:
        features_df: The dataframe returned by create_hourly_features.
                     Must contain 'patient', 'hour', 'week', plus feature columns.
        output_path: Where to save the PNG.
    """
    print("Running UMAP dimensionality reduction...")

    # 1. Select Feature Columns (Duration + Transitions)
    # Exclude metadata like patient_id, hour_ts, week, etc.
    feature_cols = [c for c in features_df.columns if c.startswith('duration_') or c.startswith('t_')]
    X = features_df[feature_cols].copy()

    # 2. Preprocessing (Crucial for Transition Matrices)
    # A. Remove Self-Loops (Optional but recommended to focus on switching)
    #    Identifies cols like 't_Work_to_Work' and drops them
    self_loop_cols = [c for c in X.columns if c.startswith('t_') and c.split('_')[1] == c.split('_')[3]]
    X = X.drop(columns=self_loop_cols)

    # B. Log Transform
    #    Compresses the massive difference between 0 counts and high counts
    X_log = np.log1p(X)

    # 3. Run UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, # larger = preserves more global structure (15-50)
        min_dist=min_dist,       # smaller = tighter clusters (0.01-0.1)
        metric='euclidean',      # or 'cosine' for sparse vector similarity
        random_state=42
    )
    embedding = reducer.fit_transform(X_log)

    # 4. Plotting
    plt.figure(figsize=(14, 10))

    # Create a temporary df for plotting
    plot_df = pd.DataFrame(embedding, columns=['x', 'y'])
    plot_df['Patient'] = features_df['patient'].values
    # You can also color by 'Hour' to see the time progression!
    plot_df['Hour'] = features_df['hour'].values

    # Plot 1: Colored by Patient (Persona)
    sns.scatterplot(
        data=plot_df, x='x', y='y',
        hue='Patient',
        palette='tab10',
        alpha=0.6,
        s=15
    )

    plt.title("UMAP Projection of Hourly Transitions", fontsize=16)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"UMAP plot saved to {output_path}")

# --- NEW: Variational Autoencoder (VAE) + KMeans + t-SNE Pipeline ---

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()

        # Encoder Base
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()

        # Latent Heads (Mean and Log-Variance)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, input_dim)

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.fc_mu(h2), self.fc_logvar(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h4 = self.relu(self.fc4(h3))
        return self.fc5(h4) # Linear output for reconstruction

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# VAE Loss Function (Reconstruction + KL Divergence)
def vae_loss_function(recon_x, x, mu, logvar, beta = 1.0):
    # 1. Reconstruction Loss (MSE)
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # 2. KL Divergence Loss (Regularization)
    #    Forces the distribution to match N(0, 1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + beta * KLD

def analyze_with_autoencoder_kmeans_tsne(features_df, output_path, n_clusters=5, latent_dim=10, epochs=50, seed=42):
    """
    Pipeline:
    1. VAE (Feature Learning) -> 2. KMeans (Clustering) -> 3. t-SNE (Visualization)
    """
    print("\n--- Running VAE -> KMeans -> t-SNE Pipeline ---")

    # 1. Feature Selection & Preprocessing
    feature_cols = [c for c in features_df.columns if c.startswith('duration_') or c.startswith('t_')]
    X = features_df[feature_cols].copy()

    # Drop self-loops (Optional, but often helps signal-to-noise)
    self_loop_cols = [c for c in X.columns if c.startswith('t_') and c.split('_')[1] == c.split('_')[3]]
    X = X.drop(columns=self_loop_cols)

    # Log transform first (handle skew)
    X_log = np.log1p(X)

    # Standard Scale (Crucial for Neural Networks)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    # Convert to PyTorch Tensor
    inputs = torch.tensor(X_scaled, dtype=torch.float32)

    # 2. Train Variational Autoencoder
    input_dim = inputs.shape[1]
    model = VariationalAutoencoder(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training VAE ({input_dim} -> {latent_dim} dim) for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(inputs)
        loss = vae_loss_function(recon_batch, inputs, mu, logvar)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 3. Extract Latent Features (Using Mean 'mu')
    # We use the deterministic mean 'mu' for downstream tasks (Clustering/Vis)
    # to get a stable representation of the input.
    model.eval()
    with torch.no_grad():
        _, latent_features, _ = model(inputs)
    latent_features = latent_features.numpy()

    # 4. K-Means Clustering on Latent Space
    print(f"Running K-Means (K={n_clusters}) on latent features...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_features)

    # 5. t-SNE Visualization of Latent Space
    print("Running t-SNE for visualization...")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(latent_features)

    # 6. Plotting
    plt.figure(figsize=(16, 8))

    plot_df = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
    plot_df['Cluster'] = cluster_labels
    plot_df['Patient'] = features_df['patient'].values

    # Subplot 1: Colored by K-Means Cluster
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        data=plot_df, x='tsne1', y='tsne2',
        hue='Cluster', palette='viridis',
        alpha=0.7, s=15, legend='full'
    )
    plt.title(f"VAE Latent Space (t-SNE) colored by K-Means (K={n_clusters})")

    # Subplot 2: Colored by Original Patient ID (Ground Truth)
    plt.subplot(1, 2, 2)
    sns.scatterplot(
        data=plot_df, x='tsne1', y='tsne2',
        hue='Patient', palette='tab10',
        alpha=0.6, s=15, legend='full'
    )
    plt.title("VAE Latent Space (t-SNE) colored by Patient")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Analysis complete. Plot saved to {output_path}")