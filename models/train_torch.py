import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, lr, wd, epochs, device):
        self.epochs = epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=lr, weight_decay=wd)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = []
        self.train_loss = []

    def train(self, save=False, plot=False, save_path="model.pth"):
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)

            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Calcul des bonnes prédictions
                _, preds = torch.max(outputs, dim=1)
                correct = (preds == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)
                total_loss += loss.item() * labels.size(0)  # perte totale pour la batch

                avg_loss = total_loss / total_samples
                avg_acc = 100.0 * total_correct / total_samples

                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Accuracy': f'{avg_acc:.2f}%'
                })

            self.train_loss.append(avg_loss)
            self.train_acc.append(avg_acc)

        if save:
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        if plot:
            self.plot_training_history()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for inputs, labels in tqdm(self.test_dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            _, preds = torch.max(outputs, dim=1)
            correct = (preds == labels).sum().item()

            total_correct += correct
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples

        print(f"\nTest Accuracy: {accuracy:.2f}%  |  Test Loss: {avg_loss:.4f}")
        return accuracy, avg_loss

    def plot_training_history(self):
        epochs = range(1, len(self.train_loss) + 1)

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color_loss = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color_loss)
        ax1.plot(epochs, self.train_loss, color=color_loss, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color_loss)

        ax2 = ax1.twinx()
        color_acc = 'tab:red'
        ax2.set_ylabel('Accuracy (%)', color=color_acc)
        ax2.plot(epochs, self.train_acc, color=color_acc, label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color_acc)

        plt.title('Training Loss and Accuracy')
        fig.tight_layout()
        plt.show()
