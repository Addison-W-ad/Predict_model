import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, LogLogisticAFTFitter, WeibullAFTFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from scipy import stats
from sklearn.metrics import concordance_index
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class SurvivalStatistics:
    def __init__(self):
        """
        Initialize survival statistics analyzer
        """
        self.kmf = KaplanMeierFitter()
        self.results = {}
        
    def kaplan_meier_analysis(self,
                            time: np.ndarray,
                            event: np.ndarray,
                            groups: Optional[np.ndarray] = None,
                            group_labels: Optional[List[str]] = None) -> Dict:
        """
        Perform Kaplan-Meier analysis with optional group comparison
        
        Parameters:
        -----------
        time : np.ndarray
            Survival times
        event : np.ndarray
            Event indicators (1 if event occurred, 0 if censored)
        groups : np.ndarray, optional
            Group labels for stratification
        group_labels : List[str], optional
            Labels for groups
            
        Returns:
        --------
        Dict
            Dictionary containing KM estimates and statistics
        """
        if groups is None:
            self.kmf.fit(time, event)
            results = {
                'survival_function': self.kmf.survival_function_,
                'median_survival': self.kmf.median_survival_time_,
                'mean_survival': self.kmf.mean_survival_time_
            }
        else:
            unique_groups = np.unique(groups)
            results = {'groups': {}}
            
            # Fit KM for each group
            for i, group in enumerate(unique_groups):
                mask = groups == group
                group_label = group_labels[i] if group_labels else str(group)
                kmf = KaplanMeierFitter()
                kmf.fit(time[mask], event[mask], label=group_label)
                results['groups'][group_label] = {
                    'survival_function': kmf.survival_function_,
                    'median_survival': kmf.median_survival_time_,
                    'mean_survival': kmf.mean_survival_time_
                }
            
            # Perform log-rank test
            if len(unique_groups) == 2:
                results['logrank_test'] = logrank_test(
                    time[groups == unique_groups[0]],
                    time[groups == unique_groups[1]],
                    event[groups == unique_groups[0]],
                    event[groups == unique_groups[1]]
                )
            else:
                results['logrank_test'] = multivariate_logrank_test(
                    time, groups, event
                )
        
        self.results['kaplan_meier'] = results
        return results
    
    def parametric_survival_analysis(self,
                                   time: np.ndarray,
                                   event: np.ndarray,
                                   features: pd.DataFrame) -> Dict:
        """
        Perform parametric survival analysis using multiple models
        
        Parameters:
        -----------
        time : np.ndarray
            Survival times
        event : np.ndarray
            Event indicators
        features : pd.DataFrame
            Covariate matrix
            
        Returns:
        --------
        Dict
            Dictionary containing model results
        """
        # Prepare data
        df = pd.DataFrame({
            'time': time,
            'event': event
        })
        df = pd.concat([df, features], axis=1)
        
        # Fit different parametric models
        results = {}
        
        # Weibull AFT model
        wf = WeibullAFTFitter()
        wf.fit(df, 'time', 'event')
        results['weibull'] = {
            'model': wf,
            'summary': wf.print_summary(),
            'AIC': wf.AIC_
        }
        
        # Log-logistic AFT model
        llf = LogLogisticAFTFitter()
        llf.fit(df, 'time', 'event')
        results['log_logistic'] = {
            'model': llf,
            'summary': llf.print_summary(),
            'AIC': llf.AIC_
        }
        
        self.results['parametric'] = results
        return results
    
    def concordance_analysis(self,
                           predicted_risks: np.ndarray,
                           time: np.ndarray,
                           event: np.ndarray) -> float:
        """
        Calculate concordance index (C-index) for model evaluation
        
        Parameters:
        -----------
        predicted_risks : np.ndarray
            Predicted risk scores
        time : np.ndarray
            Actual survival times
        event : np.ndarray
            Event indicators
            
        Returns:
        --------
        float
            Concordance index
        """
        c_index = concordance_index(time, -predicted_risks, event)
        self.results['c_index'] = c_index
        return c_index
    
    def feature_association_analysis(self,
                                   features: pd.DataFrame,
                                   time: np.ndarray,
                                   event: np.ndarray) -> pd.DataFrame:
        """
        Analyze associations between features and survival
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix
        time : np.ndarray
            Survival times
        event : np.ndarray
            Event indicators
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing association statistics
        """
        results = []
        
        for column in features.columns:
            # Correlation with survival time
            corr, p_value = stats.spearmanr(features[column], time)
            
            # Univariate Cox analysis
            kmf = KaplanMeierFitter()
            median = np.median(features[column])
            high_risk = features[column] > median
            
            # Log-rank test
            lr_test = logrank_test(
                time[high_risk],
                time[~high_risk],
                event[high_risk],
                event[~high_risk]
            )
            
            results.append({
                'feature': column,
                'correlation': corr,
                'correlation_p_value': p_value,
                'logrank_p_value': lr_test.p_value
            })
        
        results_df = pd.DataFrame(results)
        self.results['feature_associations'] = results_df
        return results_df
    
    def plot_survival_curves(self,
                           output_path: str,
                           groups: Optional[np.ndarray] = None,
                           group_labels: Optional[List[str]] = None):
        """
        Plot survival curves and save to file
        
        Parameters:
        -----------
        output_path : str
            Path to save the plot
        groups : np.ndarray, optional
            Group labels for stratification
        group_labels : List[str], optional
            Labels for groups
        """
        plt.figure(figsize=(10, 6))
        
        if 'kaplan_meier' in self.results:
            if groups is None:
                self.kmf.plot()
            else:
                for group_label, group_data in self.results['kaplan_meier']['groups'].items():
                    plt.step(
                        group_data['survival_function'].index,
                        group_data['survival_function'].values,
                        label=group_label
                    )
            
            plt.xlabel('Time')
            plt.ylabel('Survival Probability')
            plt.title('Kaplan-Meier Survival Curves')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
    def generate_report(self, output_path: str):
        """
        Generate a comprehensive statistical report
        
        Parameters:
        -----------
        output_path : str
            Path to save the report
        """
        report = []
        
        # Add Kaplan-Meier results
        if 'kaplan_meier' in self.results:
            report.append("# Kaplan-Meier Analysis")
            if 'groups' in self.results['kaplan_meier']:
                for group, data in self.results['kaplan_meier']['groups'].items():
                    report.append(f"\n## Group: {group}")
                    report.append(f"Median survival: {data['median_survival']:.2f}")
                    report.append(f"Mean survival: {data['mean_survival']:.2f}")
                
                # Add log-rank test results
                lr_test = self.results['kaplan_meier']['logrank_test']
                report.append("\n## Log-rank Test")
                report.append(f"p-value: {lr_test.p_value:.4f}")
        
        # Add parametric analysis results
        if 'parametric' in self.results:
            report.append("\n# Parametric Survival Analysis")
            for model_name, model_results in self.results['parametric'].items():
                report.append(f"\n## {model_name.capitalize()} Model")
                report.append(f"AIC: {model_results['AIC']:.2f}")
                report.append(model_results['summary'])
        
        # Add concordance index
        if 'c_index' in self.results:
            report.append("\n# Model Performance")
            report.append(f"Concordance Index: {self.results['c_index']:.3f}")
        
        # Add feature associations
        if 'feature_associations' in self.results:
            report.append("\n# Feature Association Analysis")
            report.append(self.results['feature_associations'].to_string())
        
        # Write report to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
