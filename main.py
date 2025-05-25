import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement et évaluation CNN PyTorch/TensorFlow")
    parser.add_argument('--framework', type=str, choices=['torch', 'tf'], required=True, help="Framework: 'torch' ou 'tf'")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help="Mode: entraînement ou évaluation")
    parser.add_argument('--epochs', type=int, default=10, help="Nombre d'époques")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--cuda', action='store_true', help="Utiliser GPU si disponible")
    parser.add_argument('--train_dir', type=str, default='data/training/')
    parser.add_argument('--test_dir', type=str, default='data/testing/')
    parser.add_argument('--save_path', type=str, default=None, help="Chemin pour sauvegarder/charger modèle")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.framework == 'torch':
        import torch
        from utils import prep_torch as prep
        from models.cnn_torch import get_pretrained_model
        from models.train_torch import Trainer

        device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        print(f"[PyTorch] Device: {device}")

        train_loader, test_loader, classes = prep.prepare_data(
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            batch_size=args.batch_size
        )
        print(f"[PyTorch] Classes: {classes}")

        model = get_pretrained_model().to(device)

        save_path = args.save_path if args.save_path else "souleymane_bore_model.torch"

        if args.mode == 'eval':
            print(f"[PyTorch] Chargement modèle depuis {save_path}")
            model.load_state_dict(torch.load(save_path, map_location=device))

        trainer = Trainer(model, train_loader, test_loader, args.lr, args.wd, args.epochs, device)

        if args.mode == 'train':
            print("[PyTorch] Début entraînement...")
            trainer.train(save=True, plot=True, save_path=save_path)

        print("[PyTorch] Évaluation...")
        trainer.evaluate()

    elif args.framework == 'tf':
        import tensorflow as tf
        from utils import prep_tf as prep
        from models.cnn_tf import get_pretrained_model
        from models.train_tf import TFTrainer

        # GPU config for TensorFlow
        physical_devices = tf.config.list_physical_devices('GPU')
        if args.cuda and physical_devices:
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("[TensorFlow] GPU activated")
            except Exception as e:
                print(f"[TensorFlow] Erreur activation GPU: {e}")
        else:
            print("[TensorFlow] GPU non utilisé")

        # Charger données
        train_dataset, test_dataset, classes = prep.prepare_data(
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            batch_size=args.batch_size
        )

        num_classes = len(classes)
        print(f"[TensorFlow] Classes: {classes} ({num_classes})")

        # Créer modèle avec le bon nombre de sorties
        model = get_pretrained_model(num_classes=num_classes)

        save_path = args.save_path if args.save_path else "souleymane_bore_model.tensorflow"

        if args.mode == 'eval':
            print(f"[TensorFlow] Chargement modèle depuis {save_path}")
            model.load_weights(save_path)

        trainer = TFTrainer(model, train_dataset, test_dataset, args.lr, args.epochs)

        if args.mode == 'train':
            print("[TensorFlow] Début entraînement...")
            trainer.train(save=True, plot=True)

        print("[TensorFlow] Évaluation...")
        trainer.evaluate()

    else:
        print("Framework inconnu. Choisissez 'torch' ou 'tf'.")
        sys.exit(1)

if __name__ == "__main__":
    main()




