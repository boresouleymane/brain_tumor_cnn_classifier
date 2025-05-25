
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

class TFTrainer:
    def __init__(self, model, train_dataset, test_dataset, lr, epochs):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.train_loss = []
        self.train_acc = []

    def train(self, save=False, plot=False):
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            total_loss = 0.0
            total_batches = 0

            for x_batch_train, y_batch_train in tqdm(self.train_dataset, desc="Training", leave=False):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train, training=True)
                    loss_value = self.loss_fn(y_batch_train, logits)

                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                self.train_acc_metric.update_state(y_batch_train, logits)
                total_loss += loss_value.numpy()
                total_batches += 1

            train_acc = self.train_acc_metric.result().numpy() * 100
            avg_loss = total_loss / total_batches

            print(f"Train Loss: {avg_loss:.4f} | Accuracy: {train_acc:.2f}%")
            self.train_loss.append(avg_loss)
            self.train_acc.append(train_acc)

            self.train_acc_metric.reset_state()

        if save:
            self.model.save("souleymane_bore_model.keras")  # Sauvegarde correcte au format .keras
            print("✅ Modèle sauvegardé au format .keras")
        if plot:
            self.plot_training_history()

    def evaluate(self):
        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        total_loss = 0.0
        total_batches = 0

        for x_batch_test, y_batch_test in tqdm(self.test_dataset, desc="Evaluating", leave=False):
            logits = self.model(x_batch_test, training=False)
            loss_value = self.loss_fn(y_batch_test, logits)

            acc_metric.update_state(y_batch_test, logits)
            total_loss += loss_value.numpy()
            total_batches += 1

        accuracy = acc_metric.result().numpy() * 100
        avg_loss = total_loss / total_batches

        print(f"\nTest Accuracy: {accuracy:.2f}% | Test Loss: {avg_loss:.4f}")
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
