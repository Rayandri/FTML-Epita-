import cupy as cp
import cudf
import cuml
from cuml.cluster import KMeans as cuKMeans
from cuml.manifold import UMAP as cuUMAP

print("GPU Test Results:")
print("=" * 30)

print(f"GPU: {cp.cuda.Device().name}")
print(f"VRAM: {cp.cuda.Device().mem_info[1] / 1024**3:.1f}GB")
print(f"Free: {cp.cuda.Device().mem_info[0] / 1024**3:.1f}GB")
print(f"cuML version: {cuml.__version__}")
print(f"cuDF version: {cudf.__version__}")

X = cp.random.rand(1000, 10)
print(f"\nTesting cuKMeans on {X.shape}...")
kmeans = cuKMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
print(f"Success! Labels shape: {labels.shape}")

print("\nTesting cuUMAP...")
umap = cuUMAP(n_components=2, n_neighbors=5, min_dist=0.1, random_state=42)
X_umap = umap.fit_transform(X)
print(f"Success! UMAP shape: {X_umap.shape}")

print("\nAll GPU tests passed!")
