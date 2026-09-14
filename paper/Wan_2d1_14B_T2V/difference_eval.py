import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import glob, os, json
import av

reference = "dense"  # "dense", "ca"
modified_name = "svoo"  # "ca", "svoo"
nr = 0

def load_video(path):
    container = av.open(str(path))
    frames = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24")
        frames.append(torch.from_numpy(arr))
    container.close()
    return torch.stack(frames).permute(0, 3, 1, 2).float() / 255.0

def compute_fidelity(dense_path, modified_path):
    # load videos
    dense, modified = load_video(dense_path), load_video(modified_path)

    # check for bugs (zip would truncate the longer)
    assert len(dense)==len(modified)

    # psnr and ssim (per frame, then average)
    psnrs, ssims = [], []
    for d, s in zip(dense, modified):
        d_np = d.permute(1, 2, 0).numpy()
        s_np = s.permute(1, 2, 0).numpy()
        psnrs.append(peak_signal_noise_ratio(d_np, s_np, data_range=1.0))
        ssims.append(structural_similarity(d_np, s_np, channel_axis=2, data_range=1.0))

    # lpips (per frame, then average)
    with torch.no_grad():
        lpips_scores = [
            # converting to [-1, 1]
            loss_fn(d.unsqueeze(0) * 2 - 1, s.unsqueeze(0) * 2 - 1).item()
            for d, s in zip(dense, modified)
        ]

    # return
    return {
        "psnr": float(sum(psnrs)/len(psnrs)),
        "ssim": float(sum(ssims)/len(ssims)),
        "lpips": float(sum(lpips_scores)/len(lpips_scores)),
    }

results = {}
loss_fn = lpips.LPIPS(net="alex")
folder = "paper/Wan_2d1_14B_T2V/generated_videos/"
dense_paths = sorted(glob.glob(f"{folder}wan_{reference}_*.mp4"))
for dense_path in dense_paths:
    if nr is not None:
        if dense_path != f"{folder}wan_{reference}_{nr}.mp4":
            continue
    # generate modified path, this will break with dates so must be removed
    idx = dense_path.split("_")[-1].replace(".mp4", "")
    modified_path = f"paper/Wan_2d1_14B_T2V/generated_videos/wan_{modified_name}_{idx}.mp4"
    assert os.path.exists(modified_path)

    # do the run
    results[idx] = compute_fidelity(dense_path, modified_path)

    # print the results
    print(f"Video {idx}: {results[idx]}")

# save the results
with open("paper/Wan_2d1_14B_T2V/results/lpips_results.json", "w") as f:
    json.dump(results, f, indent=2)