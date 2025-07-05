import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Tuple
import logging

class Evaluator:
    """Evaluates synthetic data against real data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compute_univariate_accuracy(self, real_data: pd.Series, synthetic_data: pd.Series) -> float:
        """Compute univariate accuracy using Kolmogorov-Smirnov test for numeric or chi-squared for categorical."""
        if pd.api.types.is_numeric_dtype(real_data):
            stat, _ = ks_2samp(real_data.dropna(), synthetic_data.dropna())
            return 1.0 - stat
        else:
            real_counts = real_data.value_counts(normalize=True)
            synth_counts = synthetic_data.value_counts(normalize=True)
            common_cats = real_counts.index.intersection(synth_counts.index)
            if len(common_cats) == 0:
                self.logger.warning(f"No common categories in {real_data.name}")
                return 0.0
            chi2_stat = sum((real_counts[cat] - synth_counts.get(cat, 0))**2 / real_counts[cat] for cat in common_cats)
            return max(0.0, 1.0 - np.sqrt(chi2_stat))
    
    def compute_cramers_v(self, real_data: pd.Series, synthetic_data: pd.Series, col1: str, col2: str) -> float:
        """Compute Cramér's V for two categorical columns."""
        contingency_table = pd.crosstab(real_data, synthetic_data)
        chi2, _, _, _ = chi2_contingency(contingency_table, correction=False)
        n = contingency_table.sum().sum()
        r, k = contingency_table.shape
        
        # Check for invalid cases
        if n == 0 or min(r, k) <= 1 or np.isnan(chi2) or np.isinf(chi2):
            self.logger.warning(f"Invalid Cramér's V for {col1} vs {col2}: n={n}, r={r}, k={k}, chi2={chi2}")
            return 0.0
        
        denominator = n * (min(r, k) - 1)
        return np.sqrt(chi2 / denominator)
    
    def compute_bivariate_accuracy(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
        """Compute average bivariate accuracy using Cramér's V for categorical pairs and correlation for numeric."""
        accuracies = []
        columns = real_data.columns
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                if pd.api.types.is_numeric_dtype(real_data[col1]) and pd.api.types.is_numeric_dtype(real_data[col2]):
                    real_corr = real_data[[col1, col2]].corr().iloc[0, 1]
                    synth_corr = synthetic_data[[col1, col2]].corr().iloc[0, 1]
                    accuracies.append(1.0 - abs(real_corr - synth_corr))
                else:
                    cramers_v = self.compute_cramers_v(real_data[col1], real_data[col2], col1, col2)
                    accuracies.append(cramers_v)
        return np.mean(accuracies) if accuracies else 0.0
    
    def compute_trivariate_accuracy(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
        """Compute trivariate accuracy using Wasserstein distance on PCA projections."""
        from sklearn.decomposition import PCA
        numeric_cols = real_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 3:
            self.logger.warning("Not enough numeric columns for trivariate accuracy")
            return 0.0
        
        real_numeric = real_data[numeric_cols].dropna()
        synth_numeric = synthetic_data[numeric_cols].dropna()
        if len(real_numeric) == 0 or len(synth_numeric) == 0:
            self.logger.warning("No complete cases for trivariate accuracy")
            return 0.0
        
        pca = PCA(n_components=3)
        real_pca = pca.fit_transform(real_numeric)
        synth_pca = pca.transform(synth_numeric)
        
        distances = [wasserstein_distance(real_pca[:, i], synth_pca[:, i]) for i in range(3)]
        max_distance = max(np.max(real_pca, axis=0) - np.min(real_pca, axis=0))
        return max(0.0, 1.0 - np.mean(distances) / max_distance if max_distance > 0 else 0.0)
    
    def compute_nndr(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
        """Compute Nearest Neighbor Distance Ratio."""
        from sklearn.neighbors import NearestNeighbors
        numeric_cols = real_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            self.logger.warning("No numeric columns for NNDR")
            return 0.0
        
        real_numeric = real_data[numeric_cols].dropna()
        synth_numeric = synthetic_data[numeric_cols].dropna()
        if len(real_numeric) < 2 or len(synth_numeric) < 2:
            self.logger.warning("Not enough data for NNDR")
            return 0.0
        
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(real_numeric)
        distances, _ = nn.kneighbors(synth_numeric)
        return np.mean(distances[:, 1] / (distances[:, 0] + 1e-10))
    
    def compute_dcr(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
        """Compute Discriminator Classification Ratio."""
        from sklearn.linear_model import LogisticRegression
        numeric_cols = real_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            self.logger.warning("No numeric columns for DCR")
            return 0.0
        
        real_numeric = real_data[numeric_cols].dropna()
        synth_numeric = synthetic_data[numeric_cols].dropna()
        if len(real_numeric) == 0 or len(synth_numeric) == 0:
            self.logger.warning("No complete cases for DCR")
            return 0.0
        
        X = pd.concat([real_numeric, synth_numeric])
        y = np.concatenate([np.ones(len(real_numeric)), np.zeros(len(synth_numeric))])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        return clf.score(X, y)
    
    def plot_distributions(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, output_dir: str) -> None:
        """Plot distributions and save to output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        for col in real_data.columns:
            plt.figure(figsize=(10, 6))
            if pd.api.types.is_numeric_dtype(real_data[col]):
                sns.histplot(real_data[col], kde=True, label='Real', stat='density')
                sns.histplot(synthetic_data[col], kde=True, label='Synthetic', stat='density')
            else:
                real_counts = real_data[col].value_counts(normalize=True)
                synth_counts = synthetic_data[col].value_counts(normalize=True)
                pd.DataFrame({'Real': real_counts, 'Synthetic': synth_counts}).plot(kind='bar')
            plt.title(f'Distribution of {col}')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
            plt.close()
    
    def evaluate(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, output_dir: str = "visualizations") -> Dict:
        """Evaluate synthetic data against real data."""
        results = {}
        univariate_accuracies = []
        for col in real_data.columns:
            acc = self.compute_univariate_accuracy(real_data[col], synthetic_data[col])
            univariate_accuracies.append(acc)
            self.logger.info(f"Univariate accuracy for {col}: {acc:.4f}")
        
        results['avg_univariate_accuracy'] = np.mean(univariate_accuracies)
        results['avg_bivariate_accuracy'] = self.compute_bivariate_accuracy(real_data, synthetic_data)
        results['avg_trivariate_accuracy'] = self.compute_trivariate_accuracy(real_data, synthetic_data)
        results['nndr'] = self.compute_nndr(real_data, synthetic_data)
        results['dcr'] = self.compute_dcr(real_data, synthetic_data)
        
        self.plot_distributions(real_data, synthetic_data, output_dir)
        return results