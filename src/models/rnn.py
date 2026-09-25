import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]
        
        # Return [batch, 1] for SHAP compatibility
        return self.regressor(last_out)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        gru_out, h_n = self.gru(x)
        last_out = gru_out[:, -1, :]
        # Return [batch, 1] for SHAP compatibility
        return self.regressor(last_out)


class AttentionModel(nn.Module):
    """
    Temporal attention model over WL and WM phases.
    
    Architecture:
    1. Project each timestep to 64-dim space with separate MLPs
    2. Apply single-head attention to compute phase importance
    3. Aggregate attended representations
    4. Regression head
    
    Key feature: Attention weights are interpretable - show which phase 
    (WL vs WM) is more important for prediction.
    """
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super(AttentionModel, self).__init__()
        
        # Project each timestep independently to hidden_dim
        self.timestep_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Single-head attention components
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = hidden_dim ** 0.5
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Store attention weights for visualization
        self.attention_weights = None
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len=2, features]
        Returns:
            predictions: [batch]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project each timestep
        # x_encoded: [batch, seq_len, hidden_dim]
        x_encoded = self.timestep_encoder(x.reshape(-1, x.shape[-1])).reshape(batch_size, seq_len, -1)
        
        # Compute Q, K, V
        Q = self.query(x_encoded)  # [batch, seq_len, hidden_dim]
        K = self.key(x_encoded)    # [batch, seq_len, hidden_dim]
        V = self.value(x_encoded)  # [batch, seq_len, hidden_dim]
        
        # Compute attention scores
        # scores: [batch, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply softmax to get attention weights
        attention = torch.softmax(scores, dim=-1)
        
        # Store for visualization (detach to avoid memory issues)
        self.attention_weights = attention.detach()
        
        # Apply attention to values
        # context: [batch, seq_len, hidden_dim]
        context = torch.matmul(attention, V)
        
        # Aggregate: mean pool over sequence dimension
        # aggregated: [batch, hidden_dim]
        aggregated = context.mean(dim=1)
        
        # Regression - Return [batch, 1] for SHAP compatibility
        out = self.regressor(aggregated)
        
        return out
