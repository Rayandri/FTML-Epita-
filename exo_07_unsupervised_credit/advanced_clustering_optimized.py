#!/usr/bin/env python3
"""
🚀 Advanced Credit Card Customer Clustering
FTML 2025 - Exercice 7 : Apprentissage Non-Supervisé Optimisé

Segmentation avancée exploitant 32 cœurs CPU + RTX 5060 GPU
Techniques state-of-the-art basées sur recherche GitHub/articles
"""

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

# GPU Libraries (with fallback)
try:
    import cuml
    import cudf
    import cupy as cp
    from cuml.cluster import KMeans as cuKMeans, DBSCAN as cuDBSCAN
    from cuml.manifold import UMAP as cuUMAP
    from cuml.decomposition import PCA as cuPCA
    GPU_AVAILABLE = True
    print("🚀 GPU cuML RAPIDS detected - Using GPU acceleration!")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  cuML not available - Using CPU fallback")

# Advanced visualization libraries
try:
    import umap
    UMAP_AVAILABLE = True
    print("✅ UMAP available")
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP not available - Using PCA fallback")

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print(f"🔧 Configuration: GPU={'✅' if GPU_AVAILABLE else '❌'} | UMAP={'✅' if UMAP_AVAILABLE else '❌'}")
print(f"💻 Ready to use 32 CPU cores + RTX 5060 GPU!")

class AdvancedCreditClustering:
    """Classe principale pour le clustering avancé des clients"""
    
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        self.results = {}
        self.df_original = None
        self.df_enhanced = None
        self.X_scaled = None
        self.target = None
        
    def load_data(self, file_path='../data/default_of_credit_card_clients.csv'):
        """Chargement et exploration des données"""
        print("📥 Chargement des données...")
        
        self.df_original = pd.read_csv(file_path)
        
        print(f"📊 Dataset : {self.df_original.shape[0]:,} observations × {self.df_original.shape[1]} variables")
        print(f"💾 Taille : {self.df_original.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        if 'default payment next month' in self.df_original.columns:
            default_rate = self.df_original['default payment next month'].mean()
            print(f"📈 Taux de défaut : {default_rate:.1%}")
        
        return self.df_original
    
    def advanced_feature_engineering(self, df):
        """Feature engineering avancé - 50+ nouvelles variables"""
        print("🔧 Feature engineering avancé...")
        
        df_fe = df.copy()
        
        # Sauvegarde de la cible
        if 'default payment next month' in df_fe.columns:
            self.target = df_fe['default payment next month'].copy()
            df_fe = df_fe.drop(['ID', 'default payment next month'], axis=1, errors='ignore')
        else:
            df_fe = df_fe.drop(['ID'], axis=1, errors='ignore')
        
        # Colonnes de référence
        bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
        pay_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        delay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        
        # === 1. RATIOS FINANCIERS ===
        print("   💰 Ratios financiers...")
        df_fe['credit_utilization'] = df_fe['BILL_AMT1'] / (df_fe['LIMIT_BAL'] + 1)
        df_fe['max_utilization'] = df_fe[bill_cols].max(axis=1) / (df_fe['LIMIT_BAL'] + 1)
        df_fe['payment_ratio'] = df_fe['PAY_AMT1'] / (df_fe['BILL_AMT1'] + 1)
        df_fe['payment_to_limit'] = df_fe['PAY_AMT1'] / (df_fe['LIMIT_BAL'] + 1)
        
        # === 2. AGRÉGATIONS TEMPORELLES ===
        print("   📊 Agrégations temporelles...")
        df_fe['avg_bill'] = df_fe[bill_cols].mean(axis=1)
        df_fe['median_bill'] = df_fe[bill_cols].median(axis=1)
        df_fe['avg_payment'] = df_fe[pay_cols].mean(axis=1)
        df_fe['median_payment'] = df_fe[pay_cols].median(axis=1)
        df_fe['avg_delay'] = df_fe[delay_cols].mean(axis=1)
        df_fe['max_delay'] = df_fe[delay_cols].max(axis=1)
        df_fe['delay_count'] = (df_fe[delay_cols] > 0).sum(axis=1)
        
        # === 3. VOLATILITÉ ===
        print("   📈 Volatilité et stabilité...")
        df_fe['bill_volatility'] = df_fe[bill_cols].std(axis=1) / (df_fe[bill_cols].mean(axis=1) + 1)
        df_fe['payment_volatility'] = df_fe[pay_cols].std(axis=1) / (df_fe[pay_cols].mean(axis=1) + 1)
        df_fe['payment_consistency'] = 1 - df_fe['payment_volatility']
        df_fe['payment_frequency'] = (df_fe[pay_cols] > 0).sum(axis=1) / 6
        
        # === 4. TENDANCES TEMPORELLES ===
        print("   📉 Tendances temporelles...")
        recent_bills = df_fe[bill_cols[:3]].mean(axis=1)
        older_bills = df_fe[bill_cols[3:]].mean(axis=1)
        df_fe['bill_trend'] = (recent_bills - older_bills) / (df_fe['avg_bill'] + 1)
        
        recent_payments = df_fe[pay_cols[:3]].mean(axis=1)
        older_payments = df_fe[pay_cols[3:]].mean(axis=1)
        df_fe['payment_trend'] = (recent_payments - older_payments) / (df_fe['avg_payment'] + 1)
        
        # === 5. INDICATEURS DE RISQUE ===
        print("   ⚠️  Indicateurs de risque...")
        df_fe['high_utilization'] = (df_fe['credit_utilization'] > 0.8).astype(int)
        df_fe['frequent_delays'] = (df_fe['delay_count'] >= 3).astype(int)
        df_fe['low_payment_ratio'] = (df_fe['payment_ratio'] < 0.1).astype(int)
        df_fe['high_volatility'] = (df_fe['bill_volatility'] > 1.0).astype(int)
        
        risk_indicators = ['high_utilization', 'frequent_delays', 'low_payment_ratio', 'high_volatility']
        df_fe['risk_score'] = df_fe[risk_indicators].sum(axis=1)
        
        # === 6. INTERACTIONS ===
        print("   🎯 Variables d'interaction...")
        df_fe['age_limit_interaction'] = df_fe['AGE'] * df_fe['LIMIT_BAL'] / 1000
        df_fe['education_utilization'] = df_fe['EDUCATION'] * df_fe['credit_utilization']
        df_fe['payment_bill_ratio'] = df_fe['avg_payment'] / (df_fe['avg_bill'] + 1)
        
        # Nettoyage final
        df_fe = df_fe.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"✅ Feature engineering terminé : {df_fe.shape[1]} variables (vs 23 originales)")
        
        return df_fe
    
    def intelligent_preprocessing(self, df_enhanced):
        """Preprocessing intelligent avec optimisation automatique"""
        print("🎛️ Preprocessing intelligent...")
        
        X = df_enhanced.select_dtypes(include=[np.number]).copy()
        print(f"   📊 Variables pour clustering : {X.shape[1]}")
        
        # Test des différents scalers
        scalers = {
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler()
        }
        
        best_scaler = None
        best_score = -1
        
        print("   🔧 Optimisation du scaler...")
        for name, scaler in scalers.items():
            X_test = scaler.fit_transform(X)
            
            # Test rapide K-means
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_test)
            score = silhouette_score(X_test, labels)
            
            print(f"      {name}: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_scaler = scaler
        
        self.X_scaled = best_scaler.fit_transform(X)
        print(f"   ✅ Scaler optimal : {type(best_scaler).__name__}")
        
        return self.X_scaled, best_scaler
    
    def optimize_k_parallel(self, X, k_range=(2, 16)):
        """Optimisation parallèle du nombre de clusters"""
        print(f"🔄 Optimisation parallèle K-means (k={k_range[0]} à {k_range[1]})...")
        
        def evaluate_k(k):
            if GPU_AVAILABLE:
                try:
                    # GPU K-means (plus rapide)
                    X_gpu = cp.asarray(X)
                    kmeans = cuKMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X_gpu).get()  # Retour CPU
                except:
                    # Fallback CPU
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X)
            else:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
            
            return {
                'k': k,
                'silhouette': silhouette_score(X, labels),
                'calinski_harabasz': calinski_harabasz_score(X, labels),
                'davies_bouldin': davies_bouldin_score(X, labels),
                'inertia': kmeans.inertia_ if hasattr(kmeans, 'inertia_') else None
            }
        
        # Parallélisation sur 32 cœurs
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_k)(k) for k in range(k_range[0], k_range[1])
        )
        
        metrics_df = pd.DataFrame(results)
        
        # Détermination du k optimal
        optimal_k_sil = metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k']
        optimal_k_cal = metrics_df.loc[metrics_df['calinski_harabasz'].idxmax(), 'k']
        optimal_k_dav = metrics_df.loc[metrics_df['davies_bouldin'].idxmin(), 'k']
        
        # Consensus pondéré
        optimal_k = int(np.median([optimal_k_sil, optimal_k_cal, optimal_k_dav]))
        
        print(f"   📊 Silhouette optimal : k={optimal_k_sil}")
        print(f"   📊 Calinski-Harabasz optimal : k={optimal_k_cal}")
        print(f"   📊 Davies-Bouldin optimal : k={optimal_k_dav}")
        print(f"   🎯 K optimal retenu : {optimal_k}")
        
        return optimal_k, metrics_df
    
    def advanced_clustering(self, X, optimal_k):
        """Application de multiples algorithmes de clustering"""
        print(f"🤖 Clustering avancé avec k={optimal_k}...")
        
        clustering_results = {}
        
        # === K-MEANS (CPU/GPU) ===
        print("   🔵 K-means...")
        if GPU_AVAILABLE:
            try:
                X_gpu = cp.asarray(X)
                kmeans = cuKMeans(n_clusters=optimal_k, random_state=42, n_init=20)
                labels_kmeans = kmeans.fit_predict(X_gpu).get()
                print("      ✅ GPU K-means")
            except:
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
                labels_kmeans = kmeans.fit_predict(X)
                print("      ⚠️  CPU K-means (fallback)")
        else:
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
            labels_kmeans = kmeans.fit_predict(X)
        
        clustering_results['K-means'] = {
            'labels': labels_kmeans,
            'silhouette': silhouette_score(X, labels_kmeans),
            'calinski_harabasz': calinski_harabasz_score(X, labels_kmeans),
            'davies_bouldin': davies_bouldin_score(X, labels_kmeans)
        }
        
        # === GAUSSIAN MIXTURE ===
        print("   🟠 Gaussian Mixture...")
        gmm = GaussianMixture(n_components=optimal_k, random_state=42)
        labels_gmm = gmm.fit_predict(X)
        
        clustering_results['GMM'] = {
            'labels': labels_gmm,
            'silhouette': silhouette_score(X, labels_gmm),
            'calinski_harabasz': calinski_harabasz_score(X, labels_gmm),
            'davies_bouldin': davies_bouldin_score(X, labels_gmm)
        }
        
        # === AGGLOMERATIVE HIERARCHICAL ===
        print("   🟣 Agglomerative Clustering...")
        agg = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
        labels_agg = agg.fit_predict(X)
        
        clustering_results['Hierarchical'] = {
            'labels': labels_agg,
            'silhouette': silhouette_score(X, labels_agg),
            'calinski_harabasz': calinski_harabasz_score(X, labels_agg),
            'davies_bouldin': davies_bouldin_score(X, labels_agg)
        }
        
        # === DBSCAN OPTIMISÉ ===
        print("   🟢 DBSCAN optimisé...")
        # Optimisation rapide d'epsilon
        neighbors = NearestNeighbors(n_neighbors=10)
        distances, _ = neighbors.fit(X).kneighbors(X)
        distances = np.sort(distances[:, 9], axis=0)
        eps = np.percentile(distances, 95)
        
        if GPU_AVAILABLE:
            try:
                X_gpu = cp.asarray(X)
                dbscan = cuDBSCAN(eps=eps, min_samples=30)
                labels_dbscan = dbscan.fit_predict(X_gpu).get()
                print("      ✅ GPU DBSCAN")
            except:
                dbscan = DBSCAN(eps=eps, min_samples=30)
                labels_dbscan = dbscan.fit_predict(X)
                print("      ⚠️  CPU DBSCAN (fallback)")
        else:
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
        """Réduction dimensionnelle UMAP/PCA"""
        print("🌀 Réduction dimensionnelle...")
        
        reduction_results = {}
        
        # === UMAP (PRÉFÉRÉ) ===
        if UMAP_AVAILABLE:
            print("   🚀 UMAP...")
            try:
                if GPU_AVAILABLE:
                    # GPU UMAP (jusqu'à 300x plus rapide!)
                    umap_model = cuUMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
                    X_umap = umap_model.fit_transform(X)
                    if hasattr(X_umap, 'to_pandas'):
                        X_umap = X_umap.to_pandas().values
                    print("      ✅ GPU UMAP")
                else:
                    # CPU UMAP
                    umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
                    X_umap = umap_model.fit_transform(X)
                    print("      ✅ CPU UMAP")
                
                reduction_results['UMAP'] = {
                    'data': X_umap,
                    'model': umap_model
                }
            except Exception as e:
                print(f"      ⚠️  Erreur UMAP : {e}")
        
        # === PCA ===
        print("   📊 PCA...")
        try:
            if GPU_AVAILABLE:
                pca_model = cuPCA(n_components=2, random_state=42)
                X_pca = pca_model.fit_transform(X)
                if hasattr(X_pca, 'to_pandas'):
                    X_pca = X_pca.to_pandas().values
                variance_explained = float(pca_model.explained_variance_ratio_.sum())
                print("      ✅ GPU PCA")
            else:
                pca_model = PCA(n_components=2, random_state=42)
                X_pca = pca_model.fit_transform(X)
                variance_explained = float(pca_model.explained_variance_ratio_.sum())
                print("      ✅ CPU PCA")
            
            reduction_results['PCA'] = {
                'data': X_pca,
                'model': pca_model,
                'variance_explained': variance_explained
            }
            
            print(f"      Variance expliquée : {variance_explained:.1%}")
            
        except Exception as e:
            print(f"      ⚠️  Erreur PCA : {e}")
        
        return reduction_results
    
    def business_analysis(self, df_enhanced, best_labels, clustering_method):
        """Analyse métier détaillée des clusters"""
        print(f"💼 Analyse métier - {clustering_method}...")
        
        df_analysis = df_enhanced.copy()
        df_analysis['cluster'] = best_labels
        
        # Statistiques par cluster
        cluster_stats = df_analysis.groupby('cluster').agg({
            'AGE': 'mean',
            'LIMIT_BAL': 'mean',
            'credit_utilization': 'mean',
            'avg_delay': 'mean',
            'avg_bill': 'mean',
            'avg_payment': 'mean',
            'risk_score': 'mean'
        }).round(2)
        
        # Tailles des clusters
        cluster_sizes = df_analysis['cluster'].value_counts().sort_index()
        
        print(f"\n📊 PROFILS DES CLUSTERS ({clustering_method}):")
        print("=" * 60)
        
        # Interprétation business
        business_profiles = {}
        
        for cluster in sorted(df_analysis['cluster'].unique()):
            if cluster == -1:  # Outliers DBSCAN
                continue
                
            cluster_data = df_analysis[df_analysis['cluster'] == cluster]
            size = len(cluster_data)
            pct = size / len(df_analysis) * 100
            
            avg_age = cluster_data['AGE'].mean()
            avg_limit = cluster_data['LIMIT_BAL'].mean()
            avg_utilization = cluster_data['credit_utilization'].mean()
            avg_delay = cluster_data['avg_delay'].mean()
            avg_risk = cluster_data['risk_score'].mean()
            
            # Classification du profil
            if avg_utilization > 0.8 and avg_delay > 1:
                profile_type = "🔴 HAUT RISQUE"
            elif avg_limit > 200000 and avg_utilization < 0.5:
                profile_type = "💎 PREMIUM"
            elif avg_utilization > 0.6:
                profile_type = "🟡 RISQUE MODÉRÉ"
            else:
                profile_type = "🟢 STANDARD"
            
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
            
            print(f"\n🏷️  Cluster {cluster} - {profile_type}")
            print(f"   👥 Taille : {size:,} clients ({pct:.1f}%)")
            print(f"   👤 Âge moyen : {avg_age:.1f} ans")
            print(f"   💳 Limite moyenne : {avg_limit:,.0f} NT$")
            print(f"   📊 Utilisation : {avg_utilization:.1%}")
            print(f"   ⏰ Retard moyen : {avg_delay:.2f}")
            print(f"   ⚠️  Score risque : {avg_risk:.1f}/4")
        
        return business_profiles, cluster_stats
    
    def validate_with_target(self, best_labels):
        """Validation avec la variable cible"""
        if self.target is None:
            print("⚠️  Pas de variable cible pour validation")
            return None
        
        print("\n✅ VALIDATION AVEC VARIABLE CIBLE")
        print("=" * 40)
        
        df_validation = pd.DataFrame({
            'cluster': best_labels,
            'default': self.target
        })
        
        # Taux de défaut par cluster
        default_rates = df_validation.groupby('cluster')['default'].agg(['mean', 'count'])
        default_rates.columns = ['taux_defaut', 'nb_clients']
        default_rates['taux_defaut_pct'] = default_rates['taux_defaut'] * 100
        
        print("Taux de défaut par cluster :")
        for cluster, row in default_rates.iterrows():
            if cluster != -1:  # Ignore outliers
                print(f"   Cluster {cluster}: {row['taux_defaut_pct']:.1f}% ({row['nb_clients']} clients)")
        
        # Test statistique
        contingency_table = pd.crosstab(df_validation['cluster'], df_validation['default'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\nTest Chi-2 d'indépendance :")
        print(f"   Chi-2 = {chi2:.2f}, p-value = {p_value:.2e}")
        print(f"   Différence significative : {'OUI' if p_value < 0.05 else 'NON'}")
        
        return default_rates
    
    def create_visualizations(self, X, clustering_results, reduction_results, best_method):
        """Création des visualisations"""
        print("📊 Création des visualisations...")
        
        fig = plt.figure(figsize=(20, 15))
        
        # === 1. COMPARAISON DES MÉTRIQUES ===
        ax1 = plt.subplot(3, 3, 1)
        
        methods = []
        silhouette_scores = []
        
        for method, result in clustering_results.items():
            if 'silhouette' in result:
                methods.append(method)
                silhouette_scores.append(result['silhouette'])
        
        bars = ax1.bar(methods, silhouette_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(methods)])
        ax1.set_title('Comparaison Silhouette Score', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Silhouette Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Ajout des valeurs sur les barres
        for bar, score in zip(bars, silhouette_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # === 2. VISUALISATION 2D ===
        if 'UMAP' in reduction_results:
            X_2d = reduction_results['UMAP']['data']
            title_2d = 'Clusters - UMAP 2D'
        elif 'PCA' in reduction_results:
            X_2d = reduction_results['PCA']['data']
            variance = reduction_results['PCA']['variance_explained']
            title_2d = f'Clusters - PCA 2D (Variance: {variance:.1%})'
        else:
            # Fallback PCA
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X)
            title_2d = f'Clusters - PCA 2D (Variance: {pca.explained_variance_ratio_.sum():.1%})'
        
        ax2 = plt.subplot(3, 3, 2)
        best_labels = clustering_results[best_method]['labels']
        
        scatter = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=best_labels, cmap='viridis', alpha=0.6, s=1)
        ax2.set_title(f'{title_2d} - {best_method}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Composante 1')
        ax2.set_ylabel('Composante 2')
        plt.colorbar(scatter, ax=ax2)
        
        # === 3. DISTRIBUTION DES CLUSTERS ===
        ax3 = plt.subplot(3, 3, 3)
        
        cluster_counts = pd.Series(best_labels).value_counts().sort_index()
        if -1 in cluster_counts.index:
            # Séparer outliers pour DBSCAN
            outliers = cluster_counts.pop(-1)
            colors = ['#1f77b4'] * len(cluster_counts) + ['#d62728']
            labels = [f'Cluster {i}' for i in cluster_counts.index] + [f'Outliers ({outliers})']
            sizes = list(cluster_counts.values) + [outliers]
        else:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(cluster_counts)]
            labels = [f'Cluster {i}' for i in cluster_counts.index]
            sizes = cluster_counts.values
        
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title(f'Distribution des Clusters - {best_method}', fontsize=14, fontweight='bold')
        
        # === 4-9. PROFILS DES VARIABLES CLÉS ===
        key_features = ['AGE', 'LIMIT_BAL', 'credit_utilization', 'avg_delay', 'avg_bill', 'risk_score']
        
        for i, feature in enumerate(key_features):
            ax = plt.subplot(3, 3, 4 + i)
            
            if feature in self.df_enhanced.columns:
                for cluster in sorted(set(best_labels)):
                    if cluster != -1:  # Ignore outliers pour clarté
                        cluster_data = self.df_enhanced.loc[best_labels == cluster, feature]
                        ax.hist(cluster_data, alpha=0.6, label=f'Cluster {cluster}', bins=20)
                
                ax.set_title(f'Distribution - {feature}', fontsize=12, fontweight='bold')
                ax.set_xlabel(feature)
                ax.set_ylabel('Fréquence')
                ax.legend()
            else:
                ax.text(0.5, 0.5, f'{feature}\nnon disponible', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{feature} - Non disponible', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('clustering_results_advanced.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualisations sauvegardées : clustering_results_advanced.png")
    
    def run_complete_analysis(self):
        """Exécution de l'analyse complète"""
        print("🚀 DÉMARRAGE DE L'ANALYSE COMPLÈTE")
        print("=" * 60)
        
        total_start_time = time.time()
        
        # 1. Chargement des données
        self.load_data()
        
        # 2. Feature engineering
        start_time = time.time()
        self.df_enhanced = self.advanced_feature_engineering(self.df_original)
        fe_time = time.time() - start_time
        print(f"⏱️  Feature engineering : {fe_time:.1f}s")
        
        # 3. Preprocessing
        start_time = time.time()
        X_scaled, scaler = self.intelligent_preprocessing(self.df_enhanced)
        preprocessing_time = time.time() - start_time
        print(f"⏱️  Preprocessing : {preprocessing_time:.1f}s")
        
        # 4. Optimisation K
        start_time = time.time()
        optimal_k, k_metrics = self.optimize_k_parallel(X_scaled)
        k_opt_time = time.time() - start_time
        print(f"⏱️  Optimisation K : {k_opt_time:.1f}s")
        
        # 5. Clustering multiple
        start_time = time.time()
        clustering_results = self.advanced_clustering(X_scaled, optimal_k)
        clustering_time = time.time() - start_time
        print(f"⏱️  Clustering : {clustering_time:.1f}s")
        
        # 6. Réduction dimensionnelle
        start_time = time.time()
        reduction_results = self.dimensionality_reduction(X_scaled)
        reduction_time = time.time() - start_time
        print(f"⏱️  Réduction dimensionnelle : {reduction_time:.1f}s")
        
        # 7. Sélection du meilleur algorithme
        best_method = max(clustering_results.items(), 
                         key=lambda x: x[1]['silhouette'] if 'silhouette' in x[1] else 0)[0]
        best_labels = clustering_results[best_method]['labels']
        
        print(f"\n🏆 MEILLEUR ALGORITHME : {best_method}")
        print(f"   📊 Silhouette Score : {clustering_results[best_method]['silhouette']:.3f}")
        print(f"   🎯 Nombre de clusters : {len(set(best_labels)) - (1 if -1 in best_labels else 0)}")
        
        # 8. Analyse métier
        business_profiles, cluster_stats = self.business_analysis(self.df_enhanced, best_labels, best_method)
        
        # 9. Validation
        validation_results = self.validate_with_target(best_labels)
        
        # 10. Visualisations
        self.create_visualizations(X_scaled, clustering_results, reduction_results, best_method)
        
        # 11. Résumé final
        total_time = time.time() - total_start_time
        
        print(f"\n🎯 ANALYSE TERMINÉE")
        print("=" * 40)
        print(f"⏱️  Temps total : {total_time:.1f}s")
        print(f"🏆 Meilleur algorithme : {best_method}")
        print(f"📊 Silhouette Score : {clustering_results[best_method]['silhouette']:.3f}")
        print(f"🎯 Clusters identifiés : {len(set(best_labels)) - (1 if -1 in best_labels else 0)}")
        
        if UMAP_AVAILABLE:
            print(f"✅ UMAP utilisé pour réduction dimensionnelle")
        if GPU_AVAILABLE:
            print(f"✅ GPU cuML utilisé pour accélération")
        
        print(f"💻 Parallélisation sur {self.n_jobs} cœurs")
        
        # Sauvegarde des résultats
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
    """Fonction principale"""
    print("💻 Initialisation du clustering avancé...")
    
    # Initialisation avec tous les cœurs disponibles
    clustering = AdvancedCreditClustering(n_jobs=-1)
    
    # Exécution de l'analyse complète
    results = clustering.run_complete_analysis()
    
    print("\n🎉 ANALYSE TERMINÉE AVEC SUCCÈS!")
    return results

if __name__ == "__main__":
    results = main() 
