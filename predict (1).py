import argparse
from model_utils import load_checkpoint, predict

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    parser.add_argument("input", type=str, help="Path to the input image")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint file")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Path to category-to-name mapping file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    args = parser.parse_args()

    model, class_to_idx = load_checkpoint(args.checkpoint)
    predict(args.input, model, args.top_k, args.category_names, args.gpu)

if __name__ == "__main__":
    main()
