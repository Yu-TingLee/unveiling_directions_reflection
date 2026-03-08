import argparse
import os
import shutil
from huggingface_hub import snapshot_download

MODELS = {
    "MyQwen2.5-3B":    "Qwen/Qwen2.5-3B",
    "MyGemma-3-4B-it": "google/gemma-3-4b-it",
}


def setup(model_name, hf_id, models_dir, cache_dir):
    dest = os.path.join(models_dir, model_name)
    print(f"downloading {hf_id} -> {dest}")
    snapshot_download(
        repo_id=hf_id,
        local_dir=dest,
        ignore_patterns=["*.gitattributes"],
        **({"cache_dir": cache_dir} if cache_dir else {}),
    )
    src = os.path.join("mymodel", model_name)
    for fname in os.listdir(src):
        shutil.copy2(os.path.join(src, fname), os.path.join(dest, fname))
        print(f"  copied {fname}")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="mymodels")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help="which models to set up (default: all)")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    for model_name in args.models:
        setup(model_name, MODELS[model_name], args.models_dir, args.cache_dir)


if __name__ == "__main__":
    run()
