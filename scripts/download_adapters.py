"""Download and unpack the fine-tuned LoRA adapters.

    python scripts/download_adapters.py <direct-download-url-of-adapters.zip>

The zip must contain target/ and draft/ folders (adapter_config.json + adapter_model.safetensors each).
For a Google Drive share link, use the form https://drive.google.com/uc?export=download&id=<FILE_ID>.
"""
import os
import sys
import urllib.request
import zipfile

if len(sys.argv) != 2:
    sys.exit(__doc__)
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapters")
os.makedirs(root, exist_ok=True)
zpath = os.path.join(root, "adapters.zip")
print("downloading ...")
urllib.request.urlretrieve(sys.argv[1], zpath)
with zipfile.ZipFile(zpath) as z:
    z.extractall(root)
os.remove(zpath)
for d in ["target", "draft"]:
    ok = os.path.exists(os.path.join(root, d, "adapter_model.safetensors"))
    print(d, "OK" if ok else "MISSING")
