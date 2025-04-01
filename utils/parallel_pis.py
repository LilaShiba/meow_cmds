import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
from neuralnet import CustomModel


class DistributedTrainer:
    """
    Handles distributed training across Docker Swarm nodes for CSV data.
    """
    def __init__(self, csv_path, batch_size=32, learning_rate=0.001, architecture="deep-wide", epochs=5):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CSV data
        self.load_csv_data()

        # Initialize model
        self.model = CustomModel(
            input_features=self.input_features,
            num_labels=self.num_labels,
            architecture=self.architecture
        ).to(self.device)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load_csv_data(self):
        """
        Loads and preprocesses CSV data.
        """
        print(f"🔍 Loading CSV dataset: {self.csv_path}")
        
        # Load CSV into pandas DataFrame
        df = pd.read_csv(self.csv_path)
        
        # Extract features and labels
        X = df.drop("label", axis=1).values  # Feature columns
        y = df["label"].values               # Labels

        # Convert to tensors
        self.features = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.labels = torch.tensor(y, dtype=torch.long).to(self.device)

        self.input_features = self.features.shape[1]
        self.num_labels = len(torch.unique(self.labels))

    def train(self):
        """
        Distributed training loop.
        """
        print("🚀 Starting distributed training...")

        # Split the dataset into training and validation
        train_x, val_x, train_y, val_y = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42
        )

        train_dataset = TensorDataset(train_x, train_y)
        val_dataset = TensorDataset(val_x, val_y)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for batch_x, batch_y in tqdm(train_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # Forward pass
                output = self.model(batch_x)
                loss = self.loss_function(output, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"🔥 Epoch [{epoch + 1}/{self.epochs}] Loss: {avg_loss:.4f}")

            # Save model checkpoint
            torch.save(self.model.state_dict(), f"model_epoch_{epoch + 1}.pth")

        self.evaluate(val_loader)

    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                output = self.model(batch_x)
                _, predicted = torch.max(output, 1)

                total_samples += batch_y.size(0)
                total_correct += (predicted == batch_y).sum().item()

        accuracy = 100 * total_correct / total_samples
        print(f"✅ Validation Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed CNN Trainer for CSV data")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--arch", type=str, default="deep-wide", help="Model architecture (deep, wide, deep-wide)")

    args = parser.parse_args()

    trainer = DistributedTrainer(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        architecture=args.arch,
        epochs=args.epochs
    )

    trainer.train()
