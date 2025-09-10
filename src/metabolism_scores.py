import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class MetabolismScoreCalculator:
    def __init__(self):
        """
        Initialize the metabolism score calculator with comprehensive metabolic pathways
        """
        self.metabolism_pathways = {
            'glycolysis': [
                'HK1', 'HK2', 'GPI', 'PFKL', 'ALDOA', 'GAPDH', 
                'PGK1', 'PGAM1', 'ENO1', 'PKM', 'LDHA', 'LDHB'
            ],
            'oxidative_phosphorylation': [
                'NDUFA', 'NDUFB', 'SDHB', 'UQCRC2', 'COX4I1', 
                'ATP5F1A', 'ATP5PB', 'ATP5MC1', 'ATP5PD'
            ],
            'fatty_acid_metabolism': [
                'ACACA', 'FASN', 'CPT1A', 'ACADM', 'HADH',
                'ACSL1', 'ACSL3', 'ACSL4', 'SCD'
            ],
            'glutamine_metabolism': [
                'GLS', 'GLUD1', 'GOT1', 'GOT2', 'GLUL',
                'SLC1A5', 'SLC7A5'
            ],
            'pentose_phosphate': [
                'G6PD', 'PGLS', 'PGD', 'TALDO1', 'TKT',
                'RPIA', 'RPE'
            ],
            'tca_cycle': [
                'CS', 'ACO2', 'IDH2', 'OGDH', 'SUCLA2',
                'SDHA', 'FH', 'MDH2'
            ]
        }
        
        # Pathway interactions and weights
        self.pathway_interactions = {
            ('glycolysis', 'tca_cycle'): 1.2,
            ('oxidative_phosphorylation', 'tca_cycle'): 1.3,
            ('fatty_acid_metabolism', 'tca_cycle'): 1.1,
            ('glutamine_metabolism', 'tca_cycle'): 1.1,
            ('glycolysis', 'pentose_phosphate'): 1.1
        }
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        
    def calculate_pathway_score(self, 
                              expression_data: pd.DataFrame,
                              pathway_genes: List[str],
                              method: str = 'zscore') -> np.ndarray:
        """
        Calculate score for a specific metabolic pathway
        
        Parameters:
        -----------
        expression_data : pd.DataFrame
            Gene expression data
        pathway_genes : List[str]
            List of genes in the pathway
        method : str
            Method to calculate pathway score ('mean', 'zscore', or 'pca')
            
        Returns:
        --------
        np.ndarray
            Pathway scores for each cell
        """
        # Get genes that are present in both pathway and data
        available_genes = [gene for gene in pathway_genes 
                         if gene in expression_data.columns]
        
        if not available_genes:
            raise ValueError("No pathway genes found in expression data")
            
        pathway_expression = expression_data[available_genes]
        
        if method == 'mean':
            return pathway_expression.mean(axis=1).values
        elif method == 'zscore':
            z_scores = stats.zscore(pathway_expression, axis=0)
            return np.mean(z_scores, axis=1)
        elif method == 'pca':
            pca = PCA(n_components=1)
            return pca.fit_transform(pathway_expression).flatten()
        else:
            raise ValueError("Invalid method. Choose 'mean', 'zscore', or 'pca'")
    
    def calculate_pathway_interaction_score(self,
                                         pathway_scores: pd.DataFrame) -> pd.Series:
        """
        Calculate interaction scores between pathways
        
        Parameters:
        -----------
        pathway_scores : pd.DataFrame
            DataFrame containing pathway scores
            
        Returns:
        --------
        pd.Series
            Interaction scores
        """
        interaction_scores = pd.Series(0.0, index=pathway_scores.index)
        
        for (pathway1, pathway2), weight in self.pathway_interactions.items():
            if f"{pathway1}_score" in pathway_scores.columns and \
               f"{pathway2}_score" in pathway_scores.columns:
                interaction = pathway_scores[f"{pathway1}_score"] * \
                            pathway_scores[f"{pathway2}_score"] * weight
                interaction_scores += interaction
                
        return interaction_scores
    
    def calculate_metabolism_score(self,
                                 expression_data: pd.DataFrame,
                                 method: str = 'zscore',
                                 include_interactions: bool = True) -> pd.DataFrame:
        """
        Calculate comprehensive metabolism score
        
        Parameters:
        -----------
        expression_data : pd.DataFrame
            Gene expression data
        method : str
            Method to calculate pathway scores
        include_interactions : bool
            Whether to include pathway interactions
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with pathway scores and overall metabolism score
        """
        # Calculate individual pathway scores
        scores = {}
        for pathway, genes in self.metabolism_pathways.items():
            scores[f"{pathway}_score"] = self.calculate_pathway_score(
                expression_data, genes, method
            )
            
        score_df = pd.DataFrame(scores, index=expression_data.index)
        
        # Add interaction scores if requested
        if include_interactions:
            score_df['interaction_score'] = self.calculate_pathway_interaction_score(score_df)
        
        # Calculate overall metabolism score
        pathway_scores = score_df.values
        self.scaler.fit(pathway_scores)
        scaled_scores = self.scaler.transform(pathway_scores)
        
        # Use PCA for final score computation
        pca_scores = self.pca.fit_transform(scaled_scores)
        score_df['metabolism_score_pc1'] = pca_scores[:, 0]
        score_df['metabolism_score_pc2'] = pca_scores[:, 1]
        score_df['metabolism_score_pc3'] = pca_scores[:, 2]
        
        # Calculate composite score
        weights = np.abs(self.pca.components_[0])
        weighted_scores = scaled_scores @ weights
        score_df['overall_metabolism_score'] = weighted_scores
        
        return score_df
    
    def get_pathway_contributions(self) -> pd.DataFrame:
        """
        Get the contribution of each pathway to the overall score
        
        Returns:
        --------
        pd.DataFrame
            Pathway contributions based on PCA loadings
        """
        if not hasattr(self.pca, 'components_'):
            raise ValueError("PCA has not been fit yet. Calculate metabolism scores first.")
            
        pathways = list(self.metabolism_pathways.keys())
        if hasattr(self, 'pathway_interactions'):
            pathways.append('interactions')
            
        contributions = pd.DataFrame(
            self.pca.components_[:3].T,
            index=pathways,
            columns=['PC1', 'PC2', 'PC3']
        )
        
        # Add explained variance ratio
        contributions.loc['explained_variance_ratio'] = self.pca.explained_variance_ratio_
        
        return contributions
    
    def get_top_contributing_genes(self, 
                                 expression_data: pd.DataFrame,
                                 n_genes: int = 20) -> pd.DataFrame:
        """
        Identify top contributing genes to metabolism score
        
        Parameters:
        -----------
        expression_data : pd.DataFrame
            Gene expression data
        n_genes : int
            Number of top genes to return
            
        Returns:
        --------
        pd.DataFrame
            Top contributing genes and their correlation with metabolism score
        """
        metabolism_scores = self.calculate_metabolism_score(expression_data)
        overall_score = metabolism_scores['overall_metabolism_score']
        
        correlations = []
        for gene in expression_data.columns:
            correlation = stats.spearmanr(expression_data[gene], overall_score)[0]
            correlations.append({
                'gene': gene,
                'correlation': correlation,
                'pathway': self._find_gene_pathway(gene)
            })
            
        correlations_df = pd.DataFrame(correlations)
        return correlations_df.nlargest(n_genes, 'correlation')
    
    def _find_gene_pathway(self, gene: str) -> str:
        """Find which pathway a gene belongs to"""
        for pathway, genes in self.metabolism_pathways.items():
            if gene in genes:
                return pathway
        return 'unknown'
