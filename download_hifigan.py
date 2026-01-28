"""
Download helper for HiFi-GAN pretrained checkpoints.
It tries a list of known URLs and extracts a checkpoint to
Real-Time-Voice-Cloning/saved_models/default/hifigan.pt

Run:
    python download_hifigan.py
"""
from pathlib import Path
import urllib.request
import tarfile
import zipfile
import shutil

candidates = [
    # Common release archive (may or may not exist)
    "https://github.com/jik876/hifi-gan/releases/download/v0.1/hifigan_v1.tar.gz",
    "https://github.com/jik876/hifi-gan/releases/download/v0.1/hifigan_v2.tar.gz",
    # Some mirrors or sample checkpoints (may fail if not present)
    "https://github.com/jik876/hifi-gan/raw/master/pretrained/hifigan_v1.pt",
    "https://github.com/jik876/hifi-gan/raw/master/pretrained/hifigan_v2.pt",
]

out_dir = Path("Real-Time-Voice-Cloning") / "saved_models" / "default"
out_dir.mkdir(parents=True, exist_ok=True)

session_dir = Path(".hifi_download")
shutil.rmtree(session_dir, ignore_errors=True)
session_dir.mkdir(exist_ok=True)

success = False
for url in candidates:
    try:
        print(f"Trying {url} ...")
        target = session_dir / Path(url).name
        urllib.request.urlretrieve(url, target)
        print(f"Downloaded {target} ({target.stat().st_size} bytes)")

        # If it's an archive, try to extract and find .pt/.pth
        if tarfile.is_tarfile(target):
            print("Detected tar file, extracting...")
            with tarfile.open(target) as t:
                t.extractall(session_dir)
        elif zipfile.is_zipfile(target):
            print("Detected zip file, extracting...")
            with zipfile.ZipFile(target) as z:
                z.extractall(session_dir)

        # Search for candidate weight files
        found = None
        for p in session_dir.rglob("*.pt"):
            found = p
            break
        if not found:
            for p in session_dir.rglob("*.pth"):
                found = p
                break

        if found:
            dest = out_dir / "hifigan.pt"
            shutil.copy(found, dest)
            print(f"Installed HiFi-GAN weights to {dest}")
            success = True
            break
        else:
            print("No .pt or .pth file found in the download; continuing to next candidate.")

    except Exception as e:
        print(f"Failed to download from {url}: {e}")

if not success:
    print("All candidate downloads failed. You can download a HiFi-GAN checkpoint manually and place it at:")
    print(out_dir / "hifigan.pt")

# Clean up session dir
# shutil.rmtree(session_dir)
print("Done")