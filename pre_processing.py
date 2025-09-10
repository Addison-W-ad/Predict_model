import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import scvi
import torch
from scipy import sparse
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class XeniumDataProcessor:
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the Xenium data processor
        
        Parameters:
        -----------
        data_dir : str or Path 
            Directory or single file directory containing Xenium data files
        """
        self.data_dir = Path(data_dir)
        self.adata = None
        print(f'CUDA available: {torch.cuda.is_available()}')
        
    def build_counts(self, path: Union[str, Path], sample_id: str, metadata_csv: pd.DataFrame) -> ad.AnnData:
        """
        Build counts matrix from Xenium data
        
        Parameters:
        -----------
        path : str or Path
            Path to the sample directory
        sample_id : str
            Sample identifier
        metadata_csv : pd.DataFrame
            Metadata for the sample
            
        Returns:
        --------
        ad.AnnData
            Annotated data matrix
        """
        # Read in the Counts Matrix
        adata = sc.read_10x_mtx(str(Path(path) / "cell_feature_matrix"), 
                              make_unique=False, gex_only=False)
        
        # Add Patient Metadata
        adata.obs['Sample ID'] = sample_id
        for col, val in metadata_csv.loc[sample_id].items():
            if col != 'Inclusion':
                adata.obs[col] = val
                
        # Add Spatial Information
        cells_df = pd.read_csv(str(Path(path) / "cells.csv"), index_col="cell_id")
        adata = adata[cells_df.index.intersection(adata.obs_names)].copy()
        adata.obs['cell_area'] = cells_df['cell_area']
        adata.obsm["spatial"] = cells_df[["x_centroid", "y_centroid"]].to_numpy()
        
        return adata

    def compute_neg_frac(self, 
                        dataset: ad.AnnData,
                        neg_types: Tuple[str, str] = ("Negative Control Probe", "Negative Control Codeword"),
                        drop_also: Tuple[str, ...] = ("Genomic Control", "Unassigned Codeword", "Deprecated Codeword")
                        ) -> ad.AnnData:
        """Calculate fraction of negative control counts"""
        ft = dataset.var["feature_types"].astype(str)
        neg_mask = ft.isin(neg_types)
        
        if not neg_mask.any():
            dataset.obs["neg_frac"] = 0.0
            drop_mask = ft.isin(set(neg_types) | set(drop_also))
            if drop_mask.any():
                dataset = dataset[:, ~drop_mask].copy()
            return dataset
            
        Xn = dataset[:, neg_mask].X
        neg_counts = (np.asarray(Xn.sum(axis=1)).ravel()
                     if sparse.issparse(Xn) else Xn.sum(axis=1))
        
        Xt = dataset.X
        total = (np.asarray(Xt.sum(axis=1)).ravel()
                if sparse.issparse(Xt) else Xt.sum(axis=1))
        total = np.where(total > 0, total, 1.0)
        
        dataset.obs["neg_frac"] = neg_counts / total
        drop_mask = ft.isin(set(neg_types) | set(drop_also))
        dataset = dataset[:, ~drop_mask].copy()
        
        return dataset

    def compute_shannon_entropy(self, dataset: ad.AnnData) -> ad.AnnData:
        """Calculate Shannon entropy for each cell"""
        X = dataset.X
        ent = np.zeros(dataset.n_obs, dtype=float)
        
        if sparse.issparse(X):
            X = X.tocsr()
            indptr = X.indptr
            data = X.data
            for i in range(dataset.n_obs):
                start, end = indptr[i], indptr[i+1]
                row = data[start:end]
                if row.size:
                    p = row / row.sum()
                    ent[i] = -np.sum(p * np.log(p))
        else:
            row_sums = X.sum(axis=1)
            nz = row_sums > 0
            P = np.zeros_like(X, dtype=float)
            P[nz] = X[nz] / row_sums[nz, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                logP = np.where(P > 0, np.log(P), 0.0)
            ent[nz] = -np.sum(P[nz] * logP[nz], axis=1)
            
        dataset.obs['entropy_shannon'] = ent
        return dataset

    def median_mad_flags(self, s: np.ndarray, k: int = 3, direction: str = 'low') -> np.ndarray:
        """Calculate median absolute deviation flags"""
        med = np.median(s)
        mad = np.median(np.abs(s - med)) or 1e-9
        if direction == 'low':
            return s < (med - k*mad)
        if direction == 'high':
            return s > (med + k*mad)

    def flag_outliers(self, dataset: ad.AnnData, k: int = 3) -> ad.AnnData:
        """Flag outlier cells based on QC metrics"""
        n_counts_t = np.log1p(dataset.obs['total_counts'])
        dataset.obs['qc_low_counts'] = self.median_mad_flags(n_counts_t, k, 'low')
        
        n_features_t = np.log1p(dataset.obs['n_genes_by_counts'])
        dataset.obs['qc_low_features'] = self.median_mad_flags(n_features_t, k, 'low')
        
        entropy_t = dataset.obs['entropy_shannon']
        dataset.obs['qc_low_entropy'] = self.median_mad_flags(entropy_t, k, 'low')
        
        p = (dataset.obs['pct_counts_in_top_1_genes'] / 100.0).astype(float)
        p = p.fillna(0.0).clip(1e-9, 1-1e-9)
        top1_t = np.log(p/(1-p))
        dataset.obs['qc_high_pct1'] = self.median_mad_flags(top1_t, k, 'high')
        
        weights = {'qc_low_counts': 1, 'qc_low_features': 1, 
                  'qc_low_entropy': 2, 'qc_high_pct1': 2}
        score = sum(dataset.obs[col].astype(int) * weight 
                   for col, weight in weights.items())
        dataset.obs['qc_outlier_score'] = score
        
        return dataset[dataset.obs['qc_outlier_score'] < 3]

    def lognormalize(self, adata: ad.AnnData) -> ad.AnnData:
        """Perform log normalization"""
        adata = adata.copy()
        adata.layers['counts'] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.layers['lognorm'] = adata.X.copy()
        adata.X = adata.layers['counts']
        return adata

    def integrate_data(self, adata: ad.AnnData) -> ad.AnnData:
        """Perform scVI integration"""
        scvi.settings.seed = 123456
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="Sample ID")
        torch.set_float32_matmul_precision('medium')
        
        vae = scvi.model.SCVI(adata, n_latent=100, n_layers=2, 
                            n_hidden=128, dropout_rate=0.1, 
                            gene_likelihood="zinb")
        vae.train(max_epochs=1000, validation_size=0.1, early_stopping=True)
        
        adata.obsm['X_scVI'] = vae.get_latent_representation()
        adata.layers['scVI_Normalized'] = vae.get_normalized_expression()
        return adata

    def cluster_data(self, adata: ad.AnnData, target_rep: str = "X_scVI") -> ad.AnnData:
        """Perform clustering and UMAP visualization"""
        sc.pp.neighbors(adata, n_neighbors=100, use_rep=target_rep)
        
        for res in [0.2, 0.5, 1.0, 1.5]:
            sc.tl.leiden(adata, resolution=res, key_added=f'leiden_r{res}')
            sc.tl.umap(adata, min_dist=0.1)
            adata.obsm[f'r{res}_umap'] = adata.obsm['X_umap'].copy()
            
        return adata

    def process_data(self, metadata_path: Union[str, Path]) -> ad.AnnData:
        """
        Process all Xenium datasets
        
        Parameters:
        -----------
        metadata_path : str or Path
            Path to metadata CSV file
        """
        # Load metadata
        metadata_df = pd.read_csv(metadata_path, index_col='Sample ID')
        adatas = []
        
        # Process each sample
        print('Loading and processing datasets...')
        for sample in metadata_df.index:
            if metadata_df.loc[sample, 'Inclusion']:
                print(f'Processing sample: {sample}')
                adata = self.build_counts(
                    self.data_dir / sample,
                    sample,
                    metadata_df
                )
                
                # QC and preprocessing
                adata = self.compute_neg_frac(adata)
                sc.pp.calculate_qc_metrics(adata, percent_top=[1], 
                                         log1p=False, inplace=True)
                adata = self.compute_shannon_entropy(adata)
                adata = self.flag_outliers(adata)
                adata = self.lognormalize(adata)
                adatas.append(adata)
        
        # Combine all samples
        print('Combining samples and performing integration...')
        combined = ad.concat(adatas, join="inner", label="Sample ID",
                           keys=metadata_df.index, index_unique='_')
        
        # Integration and clustering
        combined = self.integrate_data(combined)
        combined = self.cluster_data(combined)
        
        self.adata = combined
        return combined

    def save_data(self, output_path: Union[str, Path]):
        """Save processed data"""
        if self.adata is not None:
            self.adata.write_h5ad(output_path)
        else:
            raise ValueError("No data to save. Run process_data first.")
