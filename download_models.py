import os
import sys

def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub is not installed.")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

    # Ensure models directory exists
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Download Qwen3 GGUF model
    qwen_repo = "ggml-org/Qwen3-1.7B-GGUF"
    qwen_filename = "Qwen3-1.7B-Q4_K_M.gguf"
    qwen_dest = os.path.join(models_dir, qwen_filename)

    print(f"Downloading {qwen_filename} from {qwen_repo}...")
    if os.path.exists(qwen_dest):
        print(f"{qwen_filename} already exists at {qwen_dest}. Skipping.")
    else:
        try:
            hf_hub_download(
                repo_id=qwen_repo,
                filename=qwen_filename,
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded {qwen_filename} to {qwen_dest}")
        except Exception as e:
            print(f"Failed to download Qwen3 model from official repo: {e}")
            print("Trying alternative repository (bartowski/Qwen_Qwen3-1.7B-GGUF)...")
            try:
                hf_hub_download(
                    repo_id="bartowski/Qwen_Qwen3-1.7B-GGUF",
                    filename="Qwen3-1.7B-Q4_K_M.gguf",
                    local_dir=models_dir,
                    local_dir_use_symlinks=False
                )
                print(f"Successfully downloaded {qwen_filename} to {qwen_dest}")
            except Exception as e2:
                print(f"Fallback download also failed: {e2}")

    # 2. Download all-minilm-l6-v2 ONNX model
    onnx_repo = "onnx-community/all-MiniLM-L6-v2-ONNX"
    onnx_filename = "onnx/model.onnx"
    onnx_dest = os.path.join(models_dir, "all-minilm-l6-v2.onnx")

    print(f"\nDownloading all-minilm-l6-v2.onnx (model.onnx from {onnx_repo})...")
    if os.path.exists(onnx_dest):
        print(f"all-minilm-l6-v2.onnx already exists at {onnx_dest}. Skipping.")
    else:
        try:
            downloaded_path = hf_hub_download(
                repo_id=onnx_repo,
                filename=onnx_filename,
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
            # Rename the file to the target name
            temp_path = os.path.join(models_dir, "onnx", "model.onnx")
            if os.path.exists(temp_path):
                os.rename(temp_path, onnx_dest)
                # Clean up empty subdirectory
                try:
                    os.rmdir(os.path.join(models_dir, "onnx"))
                except Exception:
                    pass
                print(f"Successfully downloaded and renamed to {onnx_dest}")
            else:
                print(f"Downloaded model path: {downloaded_path}")
        except Exception as e:
            print(f"Failed to download embedding model: {e}")

    print("\nModel setup complete!")

if __name__ == "__main__":
    main()
