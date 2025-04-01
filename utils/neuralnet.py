import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau


class NeuralNet:
    def __init__(self, csv_path: str = None, batch_size: int = 32, epochs: int = 20,
                 learning_rate: float = 0.001, architecture: str = "deep") -> None:
        """
        Initialize NeuralNet for continuous feature prediction.

        Args:
            csv_path (str): Path to the CSV file (optional).
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for optimizer.
            architecture (str): Model architecture type ("deep", "wide", "deep-wide").
        """
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = 4  # (Symbol, Year, Month, Day)
        self.num_labels = 4  # (Close, High, Low, Open)

        self.model = self.build_model().to(self.device)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 'min', patience=2, verbose=True)

    def build_model(self) -> nn.Sequential:
        """
        Builds the model to predict continuous features from date and symbol inputs.
        """
        layers = []
        num_layers, hidden_units = self._get_architecture_params()

        # Input layer
        layers.append(nn.Linear(self.input_dim, hidden_units))
        layers.append(self.random_activation())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(self.random_activation())
            layers.append(nn.Dropout(p=random.uniform(0.2, 0.4)))

        # Output layer (5 continuous features)
        layers.append(nn.Linear(hidden_units, self.num_labels))

        return nn.Sequential(*layers)

    def _get_architecture_params(self):
        """
        Returns the architecture parameters based on the selected model type.
        """
        if self.architecture == "deep":
            return random.randint(4, 6), 128
        elif self.architecture == "wide":
            return random.randint(3, 5), 256
        elif self.architecture == "deep-wide":
            return random.randint(5, 7), 192
        else:
            raise ValueError(f"Unsupported architecture '{self.architecture}'")

    def random_activation(self) -> nn.Module:
        """
        Randomly selects an activation function.
        """
        activations = [nn.ReLU(), nn.LeakyReLU(
            0.1), nn.ELU(), nn.Tanh(), nn.Sigmoid()]
        return random.choice(activations)

    def train_model(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Training loop with validation.
        """
        print("🚀 Starting training...")

        # Convert to numpy arrays and split into training and validation sets
        features, labels = np.array(features), np.array(labels)
        train_x, val_x, train_y, val_y = train_test_split(
            features, labels, test_size=0.2, random_state=42)

        # Normalize data
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        val_x = scaler.transform(val_x)

        # Convert to tensors
        train_x, val_x = torch.tensor(train_x, dtype=torch.float32), torch.tensor(
            val_x, dtype=torch.float32)
        train_y, val_y = torch.tensor(train_y, dtype=torch.float32), torch.tensor(
            val_y, dtype=torch.float32)

        # Ensure labels and model output shapes match
        assert train_y.shape[1] == self.num_labels, f"Expected label shape ({train_y.shape[0]}, {self.num_labels}), but got {train_y.shape}"

        # DataLoader for batching
        train_loader = DataLoader(TensorDataset(
            train_x, train_y), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(
            val_x, val_y), batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf')
        no_improve_epochs = 0  # For early stopping

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(
                    self.device), batch_y.to(self.device)

                # Forward pass
                output = self.model(batch_x)
                loss = self.loss_function(output, batch_y)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += np.sqrt(loss.item())

            avg_loss = total_loss / len(train_loader)
            print(f"🔥 Epoch [{epoch + 1}/{self.epochs}] Loss: {avg_loss:.4f}")

            # Validation
            val_loss = self.evaluate(val_loader)

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"✅ New best model saved (epoch {epoch + 1})")
                torch.save(self.model.state_dict(), "best_model.pth")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # Early stopping condition
            if no_improve_epochs >= 3:
                print("⏸️ Early stopping triggered. No improvement for 3 epochs.")
                break

            # Learning rate scheduler step
            self.scheduler.step(val_loss)

        print("🚀 Training complete.")

    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(
                    self.device), batch_y.to(self.device)

                output = self.model(batch_x)
                loss = self.loss_function(output, batch_y)

                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        print(f"✅ Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using the trained model.
        """
        self.model.eval()

        with torch.no_grad():
            input_data = input_data.to(self.device)
            predictions = self.model(input_data)

        return predictions
