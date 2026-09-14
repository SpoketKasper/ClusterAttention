# - download div8k with:
#     hf download Iceclear/DIV8K_TrainingSet --repo-type dataset --local-dir div8k_raw
# - move to paper/DINOv2/data, then run this file
# - finally upload to modal with:
#     modal volume delete vit-bench-data
#     modal volume create vit-bench-data
#     modal volume put vit-bench-data paper/vit/data/DIV8K/train_HR /div8k

import zipfile
import struct
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

reference_resolution = 5306

base = Path(__file__).parent
out_dir = base / "data"

kept = []
with zipfile.ZipFile(base / "data/div8k_raw/DIV8K.zip") as zf:
    names = [n for n in zf.namelist() if n.startswith("DIV8K/train_HR/") and n.endswith(".png")]
    for name in sorted(names):
        with zf.open(name) as f:
            header = f.read(33)
        width, height = struct.unpack(">II", header[16:24])
        if min(width, height) >= reference_resolution:
            zf.extract(name, out_dir)
            kept.append(out_dir / name)
            print(f"kept {name}: {width}x{height}")

rng = random.Random(42)
rng.shuffle(kept)

thumb = 512
cols = 3  # 4
rows = 4  # 5

collage = Image.new("RGB", (cols * thumb, rows * thumb))

for i, path in enumerate(kept):
    img = Image.open(path)
    # square center crop
    s = min(img.size)
    left = (img.width - s) // 2
    top = (img.height - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((thumb, thumb))

    x = (i % cols) * thumb
    y = (i // cols) * thumb
    collage.paste(img, (x, y))

collage.save("collage.png")
plt.imshow(collage)
plt.axis("off")
plt.tight_layout()
plt.show()
