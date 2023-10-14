import argparse
from data_utils import load_and_process_data
from model_utils import build_and_train_model, save_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Train a flower classification model")
    parser.add_argument("data_dir", type=str, help="Path to the data directory")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="vgg16", help="Model architecture (vgg16, resnet50, etc.)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    args = parser.parse_args()

    dataloaders, class_to_idx = load_and_process_data(args.data_dir)
    model, optimizer, criterion = build_and_train_model(args.arch, args.hidden_units, args.learning_rate, args.gpu, dataloaders, args.epochs)
    save_checkpoint(model, args.save_dir, args.arch, class_to_idx)

if __name__ == "__main__":
    main()
