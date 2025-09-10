import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from typing import Tuple, Dict, Any, List
import torch
import torch.nn as nn

class MetabolismAttention(nn.Module):
    def __init__(self, n_pathways: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_pathways, 64),
            nn.ReLU(),
            nn.Linear(64, n_pathways),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class SurvivalNet(nn.Module):
    def __init__(self, metabolism_dim: int, clinical_dim: int, hidden_dims: list = [128, 64, 32]):
        super().__init__()
        self.metabolism_attention = MetabolismAttention(metabolism_dim)
        
        # Separate processing for metabolism and clinical features
        self.metabolism_encoder = nn.Sequential(
            nn.Linear(metabolism_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3)
        )
        
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3)
        )
        
        # Combined processing
        layers = []
        prev_dim = 96  # 64 from metabolism + 32 from clinical
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.3)
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, metabolism_features, clinical_features):
        # Process metabolism features with attention
        metabolism_attended = self.metabolism_attention(metabolism_features)
        metabolism_encoded = self.metabolism_encoder(metabolism_attended)
        
        # Process clinical features
        clinical_encoded = self.clinical_encoder(clinical_features)
        
        # Combine features
        combined = torch.cat([metabolism_encoded, clinical_encoded], dim=1)
        
        # Final processing
        return self.network(combined)

class SurvivalPredictor:
    def __init__(self, random_state: int = 42):
        """
        Initialize the survival predictor
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = {}
        
    def prepare_data(self,
                    metabolism_features: pd.DataFrame,
                    clinical_features: pd.DataFrame,
                    survival_time: np.ndarray,
                    event: np.ndarray,
                    test_size: float = 0.2) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Prepare data for survival analysis
        
        Parameters:
        -----------
        metabolism_features : pd.DataFrame
            Metabolism feature matrix
        clinical_features : pd.DataFrame
            Clinical feature matrix
        survival_time : np.ndarray
            Survival time for each patient
        event : np.ndarray
            Event indicator (1 if event occurred, 0 if censored)
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
            Training and testing data
        """
        # Combine features
        combined_features = pd.concat([metabolism_features, clinical_features], axis=1)
        self.feature_names = combined_features.columns.tolist()
        
        X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
            combined_features, survival_time, event,
            test_size=test_size,
            random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        train_data = {
            'X': X_train_scaled,
            'time': y_time_train,
            'event': y_event_train,
            'feature_names': self.feature_names
        }
        
        test_data = {
            'X': X_test_scaled,
            'time': y_time_test,
            'event': y_event_test,
            'feature_names': self.feature_names
        }
        
        return train_data, test_data
    
    def train_cox_model(self, data: pd.DataFrame) -> CoxPHFitter:
        """
        Train Cox proportional hazards model
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing features and survival information
            
        Returns:
        --------
        CoxPHFitter
            Trained Cox model
        """
        cph = CoxPHFitter()
        cph.fit(data, duration_col='time', event_col='event')
        
        # Store feature importance
        self.feature_importance['cox'] = pd.Series(
            np.abs(cph.params_),
            index=self.feature_names
        ).sort_values(ascending=False)
        
        self.models['cox'] = cph
        return cph
    
    def train_neural_net(self,
                        train_data: Dict[str, np.ndarray],
                        metabolism_features: np.ndarray,
                        clinical_features: np.ndarray,
                        epochs: int = 100,
                        batch_size: int = 32,
                        learning_rate: float = 0.001) -> SurvivalNet:
        """
        Train neural network for survival prediction
        
        Parameters:
        -----------
        train_data : Dict[str, np.ndarray]
            Training data dictionary
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimization
            
        Returns:
        --------
        SurvivalNet
            Trained neural network model
        """
        metabolism_dim = metabolism_features.shape[1]
        clinical_dim = clinical_features.shape[1]
        model = SurvivalNet(metabolism_dim, clinical_dim)
        
        # Convert data to PyTorch tensors
        X_metabolism = torch.FloatTensor(metabolism_features)
        X_clinical = torch.FloatTensor(clinical_features)
        y_time = torch.FloatTensor(train_data['time'])
        y_event = torch.FloatTensor(train_data['event'])
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            risk_scores = model(X_metabolism, X_clinical).squeeze()
            loss = self._negative_log_likelihood(risk_scores, y_time, y_event)
            
            loss.backward()
            optimizer.step()
        
        self.models['neural_net'] = model
        return model
    
    def _negative_log_likelihood(self,
                               risk_scores: torch.Tensor,
                               survival_time: torch.Tensor,
                               event: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative log likelihood for survival data
        """
        hazard_ratio = torch.exp(risk_scores)
        log_risk = torch.log(torch.cumsum(hazard_ratio, 0))
        uncensored_likelihood = risk_scores - log_risk
        censored_likelihood = uncensored_likelihood * event
        return -torch.mean(censored_likelihood)
    
    def predict_risk(self,
                    metabolism_features: np.ndarray,
                    clinical_features: np.ndarray,
                    model_type: str = 'cox') -> np.ndarray:
        """
        Predict risk scores
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix
        model_type : str
            Type of model to use for prediction ('cox' or 'neural_net')
            
        Returns:
        --------
        np.ndarray
            Predicted risk scores
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained")
            
        metabolism_scaled = self.scaler.transform(metabolism_features)
        clinical_scaled = self.scaler.transform(clinical_features)
        
        if model_type == 'cox':
            # Combine features for Cox model
            combined_features = np.concatenate([metabolism_scaled, clinical_scaled], axis=1)
            return self.models['cox'].predict_partial_hazard(pd.DataFrame(combined_features))
        elif model_type == 'neural_net':
            self.models['neural_net'].eval()
            with torch.no_grad():
                return self.models['neural_net'](
                    torch.FloatTensor(metabolism_scaled),
                    torch.FloatTensor(clinical_scaled)
                ).numpy()
    
    def get_feature_importance(self, model_type: str = 'cox') -> pd.Series:
        """
        Get feature importance scores
        
        Parameters:
        -----------
        model_type : str
            Type of model to get feature importance from
            
        Returns:
        --------
        pd.Series
            Feature importance scores
        """
        if model_type not in self.feature_importance:
            raise ValueError(f"Feature importance not available for model {model_type}")
        return self.feature_importance[model_type]
