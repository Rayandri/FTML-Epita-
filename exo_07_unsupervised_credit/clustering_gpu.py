import pandas as pd
import numpy as np
import cudf
import cupy as cp
import cuml
from cuml.cluster import KMeans as cuKMeans, DBSCAN as cuDBSCAN
from cuml.manifold import UMAP as cuUMAP
from cuml.preprocessing import StandardScaler as cuStandardScaler
from cuml.decomposition import PCA as cuPCA
from cuml.metrics import silhouette_score as cu_silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    df = pd.read_csv('data/UCI_Credit_Card.csv')
    
    df.columns = [
        'id', 'limit_bal', 'sex', 'education', 'marriage', 'age',
        'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6',
        'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
        'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6',
        'default_payment_next_month'
    ]
    
    df = df.drop('id', axis=1)
    return df

def advanced_feature_engineering(df):
    df_eng = df.copy()
    
    bill_cols = ['bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6']
    pay_cols = ['pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']
    
    df_eng['total_bill'] = df_eng[bill_cols].sum(axis=1)
    df_eng['total_payment'] = df_eng[pay_cols].sum(axis=1)
    df_eng['avg_bill'] = df_eng[bill_cols].mean(axis=1)
    df_eng['avg_payment'] = df_eng[pay_cols].mean(axis=1)
    
    df_eng['credit_utilization'] = np.where(df_eng['limit_bal'] > 0, 
                                           df_eng['total_bill'] / df_eng['limit_bal'], 0)
    df_eng['payment_ratio'] = np.where(df_eng['total_bill'] > 0, 
                                      df_eng['total_payment'] / df_eng['total_bill'], 0)
    
    df_eng['bill_volatility'] = df_eng[bill_cols].std(axis=1)
    df_eng['payment_volatility'] = df_eng[pay_cols].std(axis=1)
    
    df_eng['bill_trend'] = (df_eng['bill_amt1'] - df_eng['bill_amt6']) / 6
    df_eng['payment_trend'] = (df_eng['pay_amt1'] - df_eng['pay_amt6']) / 6
    
    pay_status_cols = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
    df_eng['late_payments'] = (df_eng[pay_status_cols] > 0).sum(axis=1)
    df_eng['severely_late'] = (df_eng[pay_status_cols] >= 2).sum(axis=1)
    
    df_eng['young_high_limit'] = ((df_eng['age'] < 30) & (df_eng['limit_bal'] > 200000)).astype(int)
    df_eng['old_low_limit'] = ((df_eng['age'] > 60) & (df_eng['limit_bal'] < 50000)).astype(int)
    
    df_eng['high_education_high_limit'] = ((df_eng['education'] == 1) & (df_eng['limit_bal'] > 300000)).astype(int)
    df_eng['married_high_util'] = ((df_eng['marriage'] == 1) & (df_eng['credit_utilization'] > 0.8)).astype(int)
    
    df_eng['financial_stress'] = (
        (df_eng['credit_utilization'] > 0.9) & 
        (df_eng['payment_ratio'] < 0.1) & 
        (df_eng['late_payments'] > 3)
    ).astype(int)
    
    df_eng['stable_payer'] = (
        (df_eng['payment_ratio'] > 0.8) & 
        (df_eng['late_payments'] == 0) & 
        (df_eng['payment_volatility'] < df_eng['payment_volatility'].quantile(0.3))
    ).astype(int)
    
    return df_eng

def gpu_clustering_analysis(df):
    target = df['default_payment_next_month']
    features = df.drop('default_payment_next_month', axis=1)
    
    gdf = cudf.from_pandas(features)
    
    scaler = cuStandardScaler()
    X_scaled = scaler.fit_transform(gdf)
    
    print(f"Dataset shape: {X_scaled.shape}")
    print(f"GPU Memory: {cp.cuda.Device().mem_info[0] / 1024**3:.1f}GB free")
    
    umap = cuUMAP(n_components=10, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = umap.fit_transform(X_scaled)
    
    best_score = -1
    best_k = 2
    best_labels = None
    
    print("Optimizing clusters on GPU...")
    for k in range(2, 12):
        kmeans = cuKMeans(n_clusters=k, random_state=42, max_iter=300)
        labels = kmeans.fit_predict(X_umap)
        
        score = cu_silhouette_score(X_umap, labels)
        print(f"K={k}: Silhouette={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    
    print(f"\nBest K={best_k} with Silhouette Score: {best_score:.4f}")
    
    final_kmeans = cuKMeans(n_clusters=best_k, random_state=42, max_iter=300)
    final_labels = final_kmeans.fit_predict(X_umap)
    
    dbscan = cuDBSCAN(eps=0.5, min_samples=50)
    dbscan_labels = dbscan.fit_predict(X_umap)
    
    results = {
        'X_umap': X_umap.to_numpy(),
        'kmeans_labels': final_labels.to_numpy(),
        'dbscan_labels': dbscan_labels.to_numpy(),
        'best_k': best_k,
        'best_score': best_score,
        'target': target.values
    }
    
    return results

def analyze_clusters(results, df):
    X_umap = results['X_umap']
    kmeans_labels = results['kmeans_labels']
    target = results['target']
    
    df_analysis = df.copy()
    df_analysis['cluster'] = kmeans_labels
    
    print(f"\nCluster Analysis (K={results['best_k']}):")
    print("=" * 50)
    
    cluster_profiles = []
    for i in range(results['best_k']):
        cluster_data = df_analysis[df_analysis['cluster'] == i]
        default_rate = cluster_data['default_payment_next_month'].mean()
        size = len(cluster_data)
        
        profile = {
            'cluster': i,
            'size': size,
            'default_rate': default_rate,
            'avg_limit': cluster_data['limit_bal'].mean(),
            'avg_age': cluster_data['age'].mean(),
            'avg_utilization': cluster_data.get('credit_utilization', pd.Series([0])).mean(),
            'avg_late_payments': cluster_data.get('late_payments', pd.Series([0])).mean()
        }
        cluster_profiles.append(profile)
        
        print(f"Cluster {i}: {size:,} customers ({size/len(df)*100:.1f}%)")
        print(f"  Default Rate: {default_rate:.3f}")
        print(f"  Avg Limit: ${profile['avg_limit']:,.0f}")
        print(f"  Avg Age: {profile['avg_age']:.1f}")
        print(f"  Avg Utilization: {profile['avg_utilization']:.3f}")
        print(f"  Avg Late Payments: {profile['avg_late_payments']:.1f}")
        print()
    
    chi2, p_value = stats.chi2_contingency(pd.crosstab(kmeans_labels, target))[:2]
    print(f"Chi-square test: χ²={chi2:.2f}, p-value={p_value:.2e}")
    
    return cluster_profiles

def create_visualizations(results, df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('GPU-Accelerated Credit Card Customer Clustering', fontsize=16, fontweight='bold')
    
    X_umap = results['X_umap']
    kmeans_labels = results['kmeans_labels']
    dbscan_labels = results['dbscan_labels']
    target = results['target']
    
    scatter1 = axes[0,0].scatter(X_umap[:, 0], X_umap[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6, s=1)
    axes[0,0].set_title(f'K-Means Clustering (K={results["best_k"]})')
    axes[0,0].set_xlabel('UMAP Component 1')
    axes[0,0].set_ylabel('UMAP Component 2')
    plt.colorbar(scatter1, ax=axes[0,0])
    
    scatter2 = axes[0,1].scatter(X_umap[:, 0], X_umap[:, 1], c=dbscan_labels, cmap='plasma', alpha=0.6, s=1)
    axes[0,1].set_title('DBSCAN Clustering')
    axes[0,1].set_xlabel('UMAP Component 1')
    axes[0,1].set_ylabel('UMAP Component 2')
    plt.colorbar(scatter2, ax=axes[0,1])
    
    scatter3 = axes[1,0].scatter(X_umap[:, 0], X_umap[:, 1], c=target, cmap='RdYlBu', alpha=0.6, s=1)
    axes[1,0].set_title('Default Payment Distribution')
    axes[1,0].set_xlabel('UMAP Component 1')
    axes[1,0].set_ylabel('UMAP Component 2')
    plt.colorbar(scatter3, ax=axes[1,0])
    
    df_viz = df.copy()
    df_viz['cluster'] = kmeans_labels
    cluster_default_rates = df_viz.groupby('cluster')['default_payment_next_month'].mean().sort_values(ascending=False)
    
    bars = axes[1,1].bar(range(len(cluster_default_rates)), cluster_default_rates.values, 
                        color=plt.cm.viridis(np.linspace(0, 1, len(cluster_default_rates))))
    axes[1,1].set_title('Default Rate by Cluster')
    axes[1,1].set_xlabel('Cluster')
    axes[1,1].set_ylabel('Default Rate')
    axes[1,1].set_xticks(range(len(cluster_default_rates)))
    axes[1,1].set_xticklabels([f'C{i}' for i in cluster_default_rates.index])
    
    for i, (cluster, rate) in enumerate(cluster_default_rates.items()):
        axes[1,1].text(i, rate + 0.01, f'{rate:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('gpu_clustering_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("GPU-Accelerated Credit Card Customer Clustering")
    print("=" * 50)
    
    print(f"GPU: {cp.cuda.Device().name}")
    print(f"VRAM: {cp.cuda.Device().mem_info[1] / 1024**3:.1f}GB")
    print(f"cuML version: {cuml.__version__}")
    print()
    
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    print("Advanced feature engineering...")
    df_engineered = advanced_feature_engineering(df)
    print(f"Features created: {df_engineered.shape[1]}")
    
    print("Running GPU clustering analysis...")
    results = gpu_clustering_analysis(df_engineered)
    
    print("Analyzing clusters...")
    cluster_profiles = analyze_clusters(results, df_engineered)
    
    print("Creating visualizations...")
    create_visualizations(results, df_engineered)
    
    print("\nGPU Clustering Analysis Complete!")
    print(f"Final Silhouette Score: {results['best_score']:.4f}")
    print(f"Optimal Clusters: {results['best_k']}")

if __name__ == "__main__":
    main() 
