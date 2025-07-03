import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Optional, Dict, Tuple
import itertools
from pathlib import Path

class Evaluator:
    """Evaluates synthetic data quality compared to original data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.label_encoders = {}
    
    def _encode_categorical(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Encode categorical columns for numerical processing."""
        encoded_data = data.copy()
        for col in columns:
            if col in data.columns:
                le = LabelEncoder()
                mask = data[col].notna()
                encoded_data.loc[mask, col] = le.fit_transform(data.loc[mask, col].astype(str))
                self.label_encoders[col] = le
        return encoded_data
    
    def univariate_accuracy(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict:
        """
        Compute univariate similarity for each column.
        
        Args:
            original: Original DataFrame
            synthetic: Synthetic DataFrame
            
        Returns:
            Dictionary with column-wise similarity scores
        """
        scores = {}
        numeric_cols = original.select_dtypes(include=[np.number]).columns
        categorical_cols = original.select_dtypes(exclude=[np.number]).columns
        
        for col in original.columns:
            if col in numeric_cols:
                # Kolmogorov-Smirnov test for numerical columns
                mask_orig = original[col].notna()
                mask_synth = synthetic[col].notna()
                if mask_orig.sum() > 0 and mask_synth.sum() > 0:
                    stat, _ = ks_2samp(original[col][mask_orig], synthetic[col][mask_synth])
                    scores[col] = 1 - stat  # Higher is better
                else:
                    scores[col] = 0.0
            elif col in categorical_cols:
                # Chi-squared test for categorical columns
                mask_orig = original[col].notna()
                mask_synth = synthetic[col].notna()
                if mask_orig.sum() > 0 and mask_synth.sum() > 0:
                    orig_counts = original[col][mask_orig].value_counts()
                    synth_counts = synthetic[col][mask_synth].value_counts()
                    categories = list(set(orig_counts.index).union(synth_counts.index))
                    orig_freq = [orig_counts.get(cat, 0) for cat in categories]
                    synth_freq = [synth_counts.get(cat, 0) for cat in categories]
                    try:
                        chi2_stat, _, _, _ = chi2_contingency([orig_freq, synth_freq])
                        scores[col] = 1 - (chi2_stat / (chi2_stat + 1))  # Normalize
                    except:
                        scores[col] = 0.0
                else:
                    scores[col] = 0.0
        
        return scores
    
    def bivariate_accuracy(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict:
        """
        Compute bivariate similarity for column pairs.
        
        Args:
            original: Original DataFrame
            synthetic: Synthetic DataFrame
            
        Returns:
            Dictionary with pair-wise similarity scores
        """
        scores = {}
        numeric_cols = original.select_dtypes(include=[np.number]).columns
        categorical_cols = original.select_dtypes(exclude=[np.number]).columns
        encoded_orig = self._encode_categorical(original, categorical_cols)
        encoded_synth = self._encode_categorical(synthetic, categorical_cols)
        
        for col1, col2 in itertools.combinations(original.columns, 2):
            key = f"{col1}-{col2}"
            mask_orig = original[[col1, col2]].notna().all(axis=1)
            mask_synth = synthetic[[col1, col2]].notna().all(axis=1)
            
            if mask_orig.sum() == 0 or mask_synth.sum() == 0:
                scores[key] = 0.0
                continue
            
            if col1 in numeric_cols and col2 in numeric_cols:
                # Pearson correlation for numerical-numerical
                orig_corr = original.loc[mask_orig, [col1, col2]].corr().iloc[0, 1]
                synth_corr = synthetic.loc[mask_synth, [col1, col2]].corr().iloc[0, 1]
                scores[key] = 1 - abs(orig_corr - synth_corr)
            elif col1 in numeric_cols and col2 in categorical_cols:
                # Mutual information for numerical-categorical
                orig_mi = mutual_info_regression(
                    encoded_orig.loc[mask_orig, [col1]], encoded_orig.loc[mask_orig, col2]
                )[0]
                synth_mi = mutual_info_regression(
                    encoded_synth.loc[mask_synth, [col1]], encoded_synth.loc[mask_synth, col2]
                )[0]
                scores[key] = 1 - abs(orig_mi - synth_mi) / (max(orig_mi, synth_mi) + 1e-10)
            elif col1 in categorical_cols and col2 in categorical_cols:
                # Cramer's V for categorical-categorical
                def cramers_v(confusion_matrix):
                    chi2 = chi2_contingency(confusion_matrix)[0]
                    n = confusion_matrix.sum()
                    r, k = confusion_matrix.shape
                    return np.sqrt(chi2 / (n * (min(r, k) - 1)))
                
                orig_table = pd.crosstab(encoded_orig.loc[mask_orig, col1], encoded_orig.loc[mask_orig, col2])
                synth_table = pd.crosstab(encoded_synth.loc[mask_synth, col1], encoded_synth.loc[mask_synth, col2])
                try:
                    orig_v = cramers_v(orig_table.values)
                    synth_v = cramers_v(synth_table.values)
                    scores[key] = 1 - abs(orig_v - synth_v)
                except:
                    scores[key] = 0.0
        
        return scores
    
    def trivariate_accuracy(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict:
        """
        Compute trivariate similarity for column triplets.
        
        Args:
            original: Original DataFrame
            synthetic: Synthetic DataFrame
            
        Returns:
            Dictionary with triplet-wise mutual information scores
        """
        scores = {}
        encoded_orig = self._encode_categorical(original, original.select_dtypes(exclude=[np.number]).columns)
        encoded_synth = self._encode_categorical(synthetic, synthetic.select_dtypes(exclude=[np.number]).columns)
        
        for cols in itertools.combinations(original.columns, 3):
            key = "-".join(cols)
            mask_orig = original[list(cols)].notna().all(axis=1)
            mask_synth = synthetic[list(cols)].notna().all(axis=1)
            
            if mask_orig.sum() == 0 or mask_synth.sum() == 0:
                scores[key] = 0.0
                continue
            
            # Compute mutual information for triplets
            orig_data = encoded_orig.loc[mask_orig, list(cols)]
            synth_data = encoded_synth.loc[mask_synth, list(cols)]
            mi_orig = sum(mutual_info_regression(orig_data[[col]], orig_data[cols[0]])[0] 
                         for col in cols[1:])
            mi_synth = sum(mutual_info_regression(synth_data[[col]], synth_data[cols[0]])[0] 
                          for col in cols[1:])
            scores[key] = 1 - abs(mi_orig - mi_synth) / (max(mi_orig, mi_synth) + 1e-10)
        
        return scores
    
    def nndr(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        """
        Compute Nearest Neighbor Distance Ratio.
        
        Args:
            original: Original DataFrame
            synthetic: Synthetic DataFrame
            
        Returns:
            NNDR value
        """
        numeric_cols = original.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 0.0
        
        orig_data = original[numeric_cols].dropna()
        synth_data = synthetic[numeric_cols].dropna()
        if len(orig_data) == 0 or len(synth_data) == 0:
            return 0.0
        
        nn = NearestNeighbors(n_neighbors=2).fit(orig_data)
        distances_orig, _ = nn.kneighbors(orig_data)
        avg_dist_orig = distances_orig[:, 1].mean()
        
        nn = NearestNeighbors(n_neighbors=2).fit(synth_data)
        distances_synth, _ = nn.kneighbors(synth_data)
        avg_dist_synth = distances_synth[:, 1].mean()
        
        return avg_dist_synth / (avg_dist_orig + 1e-10)
    
    def dcr(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        """
        Compute Discrimination Capacity Ratio.
        
        Args:
            original: Original DataFrame
            synthetic: Synthetic DataFrame
            
        Returns:
            DCR value
        """
        encoded_orig = self._encode_categorical(original, original.select_dtypes(exclude=[np.number]).columns)
        encoded_synth = self._encode_categorical(synthetic, synthetic.select_dtypes(exclude=[np.number]).columns)
        
        common_cols = [col for col in original.columns if col in synthetic.columns]
        if not common_cols:
            return 1.0
        
        data = pd.concat([
            encoded_orig[common_cols].assign(label=1),
            encoded_synth[common_cols].assign(label=0)
        ]).dropna()
        
        if len(data) < 2:
            return 1.0
        
        X = data.drop("label", axis=1)
        y = data["label"]
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        return accuracy_score(y, clf.predict(X))
    
    def visualize(self, original: pd.DataFrame, synthetic: pd.DataFrame, output_dir: str = "visualizations") -> None:
        """
        Generate visualizations for univariate and bivariate comparisons.
        
        Args:
            original: Original DataFrame
            synthetic: Synthetic DataFrame
            output_dir: Directory to save plots
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        numeric_cols = original.select_dtypes(include=[np.number]).columns
        categorical_cols = original.select_dtypes(exclude=[np.number]).columns
        
        # Univariate plots
        for col in original.columns:
            plt.figure(figsize=(8, 6))
            if col in numeric_cols:
                sns.histplot(original[col].dropna(), label="Original", kde=True, alpha=0.5)
                sns.histplot(synthetic[col].dropna(), label="Synthetic", kde=True, alpha=0.5)
                plt.title(f"Univariate Distribution: {col}")
            elif col in categorical_cols:
                orig_counts = original[col].value_counts(normalize=True)
                synth_counts = synthetic[col].value_counts(normalize=True)
                pd.DataFrame({"Original": orig_counts, "Synthetic": synth_counts}).plot(kind="bar")
                plt.title(f"Univariate Distribution: {col}")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"univariate_{col}.png"))
            plt.close()
        
        # Bivariate plots
        for col1, col2 in itertools.combinations(original.columns, 2):
            plt.figure(figsize=(8, 6))
            if col1 in numeric_cols and col2 in numeric_cols:
                sns.scatterplot(x=original[col1], y=original[col2], label="Original", alpha=0.5)
                sns.scatterplot(x=synthetic[col1], y=synthetic[col2], label="Synthetic", alpha=0.5)
                plt.title(f"Bivariate Scatter: {col1} vs {col2}")
            elif col1 in numeric_cols and col2 in categorical_cols:
                sns.boxplot(x=original[col2], y=original[col1], alpha=0.5, label="Original")
                sns.boxplot(x=synthetic[col2], y=synthetic[col1], alpha=0.5, label="Synthetic")
                plt.title(f"Bivariate Boxplot: {col1} vs {col2}")
            elif col1 in categorical_cols and col2 in categorical_cols:
                orig_table = pd.crosstab(original[col1], original[col2], normalize=True)
                synth_table = pd.crosstab(synthetic[col1], synthetic[col2], normalize=True)
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                sns.heatmap(orig_table, annot=True, ax=axes[0], cmap="Blues")
                sns.heatmap(synth_table, annot=True, ax=axes[1], cmap="Blues")
                axes[0].set_title("Original")
                axes[1].set_title("Synthetic")
                fig.suptitle(f"Bivariate Heatmap: {col1} vs {col2}")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"bivariate_{col1}_{col2}.png"))
            plt.close()
    
    def evaluate(self, 
                 original: pd.DataFrame, 
                 synthetic: pd.DataFrame, 
                 output_dir: Optional[str] = "visualizations") -> Dict:
        """
        Evaluate synthetic data quality.
        
        Args:
            original: Original DataFrame
            synthetic: Synthetic DataFrame
            output_dir: Directory for visualizations
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            results = {
                "univariate_accuracy": self.univariate_accuracy(original, synthetic),
                "bivariate_accuracy": self.bivariate_accuracy(original, synthetic),
                "trivariate_accuracy": self.trivariate_accuracy(original, synthetic),
                "nndr": self.nndr(original, synthetic),
                "dcr": self.dcr(original, synthetic)
            }
            
            if output_dir:
                self.visualize(original, synthetic, output_dir)
            
            # Compute average scores
            results["avg_univariate_accuracy"] = np.mean(list(results["univariate_accuracy"].values()))
            results["avg_bivariate_accuracy"] = np.mean(list(results["bivariate_accuracy"].values()))
            results["avg_trivariate_accuracy"] = np.mean(list(results["trivariate_accuracy"].values()))
            
            self.logger.info(
                f"Evaluation Results:\n"
                f"Avg Univariate Accuracy: {results['avg_univariate_accuracy']:.4f}\n"
                f"Avg Bivariate Accuracy: {results['avg_bivariate_accuracy']:.4f}\n"
                f"Avg Trivariate Accuracy: {results['avg_trivariate_accuracy']:.4f}\n"
                f"NNDR: {results['nndr']:.4f}\n"
                f"DCR: {results['dcr']:.4f}"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating data: {str(e)}")
            raise