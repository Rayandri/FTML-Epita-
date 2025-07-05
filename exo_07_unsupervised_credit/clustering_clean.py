#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from collections import Counter
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import chi2_contingency

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CreditClustering:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        self.results = {}
        self.df_original = None
        self.df_enhanced = None
        self.X_scaled = None
        self.target = None
        
    def load_data(self, file_path='../data/default_of_credit_card_clients.csv'):
        self.df_original = pd.read_csv(file_path)
        print(f"Dataset: {self.df_original.shape[0]:,} observations x {self.df_original.shape[1]} variables")
        
        if 'default payment next month' in self.df_original.columns:
            default_rate = self.df_original['default payment next month'].mean()
            print(f"Default rate: {default_rate:.1%}")
        
        return self.df_original
    
    def feature_engineering(self, df):
        df_fe = df.copy()
        
        if 'default payment next month' in df_fe.columns:
            self.target = df_fe['default payment next month'].copy()
            df_fe = df_fe.drop(['ID', 'default payment next month'], axis=1, errors='ignore')
        else:
            df_fe = df_fe.drop(['ID'], axis=1, errors='ignore')
        
        bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
        pay_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        delay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        
        df_fe['credit_utilization'] = df_fe['BILL_AMT1'] / (df_fe['LIMIT_BAL'] + 1)
        df_fe['max_utilization'] = df_fe[bill_cols].max(axis=1) / (df_fe['LIMIT_BAL'] + 1)
        df_fe['payment_ratio'] = df_fe['PAY_AMT1'] / (df_fe['BILL_AMT1'] + 1)
        df_fe['payment_to_limit'] = df_fe['PAY_AMT1'] / (df_fe['LIMIT_BAL'] + 1)
        
        df_fe['avg_bill'] = df_fe[bill_cols].mean(axis=1)
        df_fe['median_bill'] = df_fe[bill_cols].median(axis=1)
        df_fe['avg_payment'] = df_fe[pay_cols].mean(axis=1)
        df_fe['median_payment'] = df_fe[pay_cols].median(axis=1)
        df_fe['avg_delay'] = df_fe[delay_cols].mean(axis=1)
        df_fe['max_delay'] = df_fe[delay_cols].max(axis=1)
        df_fe['delay_count'] = (df_fe[delay_cols] > 0).sum(axis=1)
        
        df_fe['bill_volatility'] = df_fe[bill_cols].std(axis=1) / (df_fe[bill_cols].mean(axis=1) + 1)
        df_fe['payment_volatility'] = df_fe[pay_cols].std(axis=1) / (df_fe[pay_cols].mean(axis=1) + 1)
        df_fe['payment_consistency'] = 1 - df_fe['payment_volatility']
        df_fe['payment_frequency'] = (df_fe[pay_cols] > 0).sum(axis=1) / 6
        
        recent_bills = df_fe[bill_cols[:3]].mean(axis=1)
        older_bills = df_fe[bill_cols[3:]].mean(axis=1)
        df_fe['bill_trend'] = (recent_bills - older_bills) / (df_fe['avg_bill'] + 1)
        
        recent_payments = df_fe[pay_cols[:3]].mean(axis=1)
        older_payments = df_fe[pay_cols[3:]].mean(axis=1)
        df_fe['payment_trend'] = (recent_payments - older_payments) / (df_fe['avg_payment'] + 1)
        
        df_fe['high_utilization'] = (df_fe['credit_utilization'] > 0.8).astype(int)
        df_fe['frequent_delays'] = (df_fe['delay_count'] >= 3).astype(int)
        df_fe['low_payment_ratio'] = (df_fe['payment_ratio'] < 0.1).astype(int)
        df_fe['high_volatility'] = (df_fe['bill_volatility'] > 1.0).astype(int)
        
        risk_indicators = ['high_utilization', 'frequent_delays', 'low_payment_ratio', 'high_volatility']
        df_fe['risk_score'] = df_fe[risk_indicators].sum(axis=1)
        
        df_fe['age_limit_interaction'] = df_fe['AGE'] * df_fe['LIMIT_BAL'] / 1000
        df_fe['education_utilization'] = df_fe['EDUCATION'] * df_fe['credit_utilization']
        df_fe['payment_bill_ratio'] = df_fe['avg_payment'] / (df_fe['avg_bill'] + 1)
        
        df_fe = df_fe.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"Feature engineering completed: {df_fe.shape[1]} variables")
        
        return df_fe
    
    def preprocessing(self, df_enhanced):
        X = df_enhanced.select_dtypes(include=[np.number]).copy()
        
        scalers = {
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler()
        }
        
        best_scaler = None
        best_score = -1
        
        for name, scaler in scalers.items():
            X_test = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_test)
            score = silhouette_score(X_test, labels)
            
            if score > best_score:
                best_score = score
                best_scaler = scaler
        
        self.X_scaled = best_scaler.fit_transform(X)
        print(f"Best scaler: {type(best_scaler).__name__}")
        
        return self.X_scaled, best_scaler
    
    def optimize_k_parallel(self, X, k_range=(2, 16)):
        def evaluate_k(k):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            return {
                'k': k,
                'silhouette': silhouette_score(X, labels),
                'calinski_harabasz': calinski_harabasz_score(X, labels),
                'davies_bouldin': davies_bouldin_score(X, labels),
                'inertia': kmeans.inertia_
            }
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_k)(k) for k in range(k_range[0], k_range[1])
        )
        
        metrics_df = pd.DataFrame(results)
        
        optimal_k_sil = metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k']
        optimal_k_cal = metrics_df.loc[metrics_df['calinski_harabasz'].idxmax(), 'k']
        optimal_k_dav = metrics_df.loc[metrics_df['davies_bouldin'].idxmin(), 'k']
        
        optimal_k = int(np.median([optimal_k_sil, optimal_k_cal, optimal_k_dav]))
        
        print(f"Optimal k: {optimal_k}")
        
        return optimal_k, metrics_df
    
    def clustering(self, X, optimal_k):
        clustering_results = {}
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
        labels_kmeans = kmeans.fit_predict(X)
        
        clustering_results['K-means'] = {
            'labels': labels_kmeans,
            'silhouette': silhouette_score(X, labels_kmeans),
            'calinski_harabasz': calinski_harabasz_score(X, labels_kmeans),
            'davies_bouldin': davies_bouldin_score(X, labels_kmeans)
        }
        
        gmm = GaussianMixture(n_components=optimal_k, random_state=42)
        labels_gmm = gmm.fit_predict(X)
        
        clustering_results['GMM'] = {
            'labels': labels_gmm,
            'silhouette': silhouette_score(X, labels_gmm),
            'calinski_harabasz': calinski_harabasz_score(X, labels_gmm),
            'davies_bouldin': davies_bouldin_score(X, labels_gmm)
        }
        
        agg = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
        labels_agg = agg.fit_predict(X)
        
        clustering_results['Hierarchical'] = {
            'labels': labels_agg,
            'silhouette': silhouette_score(X, labels_agg),
            'calinski_harabasz': calinski_harabasz_score(X, labels_agg),
            'davies_bouldin': davies_bouldin_score(X, labels_agg)
        }
        
        neighbors = NearestNeighbors(n_neighbors=10)
        distances, _ = neighbors.fit(X).kneighbors(X)
        distances = np.sort(distances[:, 9], axis=0)
        eps = np.percentile(distances, 95)
        
        dbscan = DBSCAN(eps=eps, min_samples=30)
        labels_dbscan = dbscan.fit_predict(X)
        
        n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
        n_outliers = sum(labels_dbscan == -1)
        
        if n_clusters_dbscan > 1:
            mask = labels_dbscan != -1
            clustering_results['DBSCAN'] = {
                'labels': labels_dbscan,
                'silhouette': silhouette_score(X[mask], labels_dbscan[mask]) if sum(mask) > 1 else 0,
                'n_clusters': n_clusters_dbscan,
                'n_outliers': n_outliers
            }
        
        return clustering_results
    
    def dimensionality_reduction(self, X):
        reduction_results = {}
        
        if UMAP_AVAILABLE:
            umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
            X_umap = umap_model.fit_transform(X)
            
            reduction_results['UMAP'] = {
                'data': X_umap,
                'model': umap_model
            }
        
        pca_model = PCA(n_components=2, random_state=42)
        X_pca = pca_model.fit_transform(X)
        variance_explained = float(pca_model.explained_variance_ratio_.sum())
        
        reduction_results['PCA'] = {
            'data': X_pca,
            'model': pca_model,
            'variance_explained': variance_explained
        }
        
        print(f"PCA variance explained: {variance_explained:.1%}")
        
        return reduction_results
    
    def business_analysis(self, df_enhanced, best_labels, clustering_method):
        df_analysis = df_enhanced.copy()
        df_analysis['cluster'] = best_labels
        
        cluster_stats = df_analysis.groupby('cluster').agg({
            'AGE': 'mean',
            'LIMIT_BAL': 'mean',
            'credit_utilization': 'mean',
            'avg_delay': 'mean',
            'avg_bill': 'mean',
            'avg_payment': 'mean',
            'risk_score': 'mean'
        }).round(2)
        
        cluster_sizes = df_analysis['cluster'].value_counts().sort_index()
        
        print(f"\nCluster profiles ({clustering_method}):")
        print("=" * 50)
        
        business_profiles = {}
        
        for cluster in sorted(df_analysis['cluster'].unique()):
            if cluster == -1:
                continue
                
            cluster_data = df_analysis[df_analysis['cluster'] == cluster]
            size = len(cluster_data)
            pct = size / len(df_analysis) * 100
            
            avg_age = cluster_data['AGE'].mean()
            avg_limit = cluster_data['LIMIT_BAL'].mean()
            avg_utilization = cluster_data['credit_utilization'].mean()
            avg_delay = cluster_data['avg_delay'].mean()
            avg_risk = cluster_data['risk_score'].mean()
            
            if avg_utilization > 0.8 and avg_delay > 1:
                profile_type = "HIGH RISK"
            elif avg_limit > 200000 and avg_utilization < 0.5:
                profile_type = "PREMIUM"
            elif avg_utilization > 0.6:
                profile_type = "MODERATE RISK"
            else:
                profile_type = "STANDARD"
            
            business_profiles[cluster] = {
                'type': profile_type,
                'size': size,
                'percentage': pct,
                'avg_age': avg_age,
                'avg_limit': avg_limit,
                'avg_utilization': avg_utilization,
                'avg_delay': avg_delay,
                'avg_risk': avg_risk
            }
            
            print(f"\nCluster {cluster} - {profile_type}")
            print(f"  Size: {size:,} clients ({pct:.1f}%)")
            print(f"  Age: {avg_age:.1f} years")
            print(f"  Credit limit: {avg_limit:,.0f} NT$")
            print(f"  Utilization: {avg_utilization:.1%}")
            print(f"  Delay: {avg_delay:.2f}")
            print(f"  Risk score: {avg_risk:.1f}/4")
        
        return business_profiles, cluster_stats
    
    def validate_with_target(self, best_labels):
        if self.target is None:
            return None
        
        print("\nValidation with target variable:")
        print("=" * 30)
        
        df_validation = pd.DataFrame({
            'cluster': best_labels,
            'default': self.target
        })
        
        default_rates = df_validation.groupby('cluster')['default'].agg(['mean', 'count'])
        default_rates.columns = ['default_rate', 'count']
        default_rates['default_rate_pct'] = default_rates['default_rate'] * 100
        
        for cluster, row in default_rates.iterrows():
            if cluster != -1:
                print(f"  Cluster {cluster}: {row['default_rate_pct']:.1f}% default rate ({row['count']} clients)")
        
        contingency_table = pd.crosstab(df_validation['cluster'], df_validation['default'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\nChi-square test:")
        print(f"  Chi-2 = {chi2:.2f}, p-value = {p_value:.2e}")
        print(f"  Significant difference: {'YES' if p_value < 0.05 else 'NO'}")
        
        return default_rates
    
    def create_visualizations(self, X, clustering_results, reduction_results, best_method):
        fig = plt.figure(figsize=(15, 10))
        
        methods = []
        silhouette_scores = []
        
        for method, result in clustering_results.items():
            if 'silhouette' in result:
                methods.append(method)
                silhouette_scores.append(result['silhouette'])
        
        ax1 = plt.subplot(2, 2, 1)
        bars = ax1.bar(methods, silhouette_scores)
        ax1.set_title('Silhouette Score Comparison')
        ax1.set_ylabel('Silhouette Score')
        plt.xticks(rotation=45)
        
        for bar, score in zip(bars, silhouette_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{score:.3f}', ha='center', va='bottom')
        
        if 'UMAP' in reduction_results:
            X_2d = reduction_results['UMAP']['data']
            title_2d = 'Clusters - UMAP 2D'
        elif 'PCA' in reduction_results:
            X_2d = reduction_results['PCA']['data']
            variance = reduction_results['PCA']['variance_explained']
            title_2d = f'Clusters - PCA 2D (Variance: {variance:.1%})'
        
        ax2 = plt.subplot(2, 2, 2)
        best_labels = clustering_results[best_method]['labels']
        
        scatter = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=best_labels, cmap='viridis', alpha=0.6, s=1)
        ax2.set_title(f'{title_2d} - {best_method}')
        ax2.set_xlabel('Component 1')
        ax2.set_ylabel('Component 2')
        
        ax3 = plt.subplot(2, 2, 3)
        cluster_counts = pd.Series(best_labels).value_counts().sort_index()
        ax3.pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], autopct='%1.1f%%')
        ax3.set_title(f'Cluster Distribution - {best_method}')
        
        ax4 = plt.subplot(2, 2, 4)
        if 'credit_utilization' in self.df_enhanced.columns:
            for cluster in sorted(set(best_labels)):
                if cluster != -1:
                    cluster_data = self.df_enhanced.loc[best_labels == cluster, 'credit_utilization']
                    ax4.hist(cluster_data, alpha=0.6, label=f'Cluster {cluster}', bins=20)
            
            ax4.set_title('Credit Utilization by Cluster')
            ax4.set_xlabel('Credit Utilization')
            ax4.set_ylabel('Frequency')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig('clustering_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved: clustering_results.png")
    
    def run_analysis(self):
        print("Starting complete clustering analysis")
        print("=" * 40)
        
        total_start_time = time.time()
        
        self.load_data()
        
        start_time = time.time()
        self.df_enhanced = self.feature_engineering(self.df_original)
        fe_time = time.time() - start_time
        
        start_time = time.time()
        X_scaled, scaler = self.preprocessing(self.df_enhanced)
        preprocessing_time = time.time() - start_time
        
        start_time = time.time()
        optimal_k, k_metrics = self.optimize_k_parallel(X_scaled)
        k_opt_time = time.time() - start_time
        
        start_time = time.time()
        clustering_results = self.clustering(X_scaled, optimal_k)
        clustering_time = time.time() - start_time
        
        start_time = time.time()
        reduction_results = self.dimensionality_reduction(X_scaled)
        reduction_time = time.time() - start_time
        
        best_method = max(clustering_results.items(), 
                         key=lambda x: x[1]['silhouette'] if 'silhouette' in x[1] else 0)[0]
        best_labels = clustering_results[best_method]['labels']
        
        print(f"\nBest algorithm: {best_method}")
        print(f"Silhouette score: {clustering_results[best_method]['silhouette']:.3f}")
        print(f"Number of clusters: {len(set(best_labels)) - (1 if -1 in best_labels else 0)}")
        
        business_profiles, cluster_stats = self.business_analysis(self.df_enhanced, best_labels, best_method)
        validation_results = self.validate_with_target(best_labels)
        self.create_visualizations(X_scaled, clustering_results, reduction_results, best_method)
        
        total_time = time.time() - total_start_time
        
        print(f"\nAnalysis completed")
        print(f"Total time: {total_time:.1f}s")
        print(f"Best algorithm: {best_method}")
        print(f"Silhouette score: {clustering_results[best_method]['silhouette']:.3f}")
        print(f"Clusters identified: {len(set(best_labels)) - (1 if -1 in best_labels else 0)}")
        
        self.results = {
            'best_method': best_method,
            'best_labels': best_labels,
            'clustering_results': clustering_results,
            'business_profiles': business_profiles,
            'validation_results': validation_results,
            'optimal_k': optimal_k,
            'processing_times': {
                'feature_engineering': fe_time,
                'preprocessing': preprocessing_time,
                'k_optimization': k_opt_time,
                'clustering': clustering_time,
                'reduction': reduction_time,
                'total': total_time
            }
        }
        
        return self.results

def main():
    clustering = CreditClustering(n_jobs=-1)
    results = clustering.run_analysis()
    return results

if __name__ == "__main__":
    results = main() 
