import json
import glob

version = "v1"
folder = f"paper/Wan_2d1_14B_T2V/results/{version}"
modified_backends = [
    "clusterattention_topk_finecomp", "svoo"
]

# load all results
all_entries = []
for path in sorted(glob.glob(f"{folder}/pareto_*.json")):
    with open(path) as f:
        all_entries.extend(json.load(f))

grouped_by_prompt = {}
for e in all_entries:
    prompt = e["setup"]["prompt_idx"]
    if prompt not in grouped_by_prompt:
        grouped_by_prompt[prompt] = {}
    grouped_by_prompt[prompt][e["name"]] = e

for prompt in sorted(grouped_by_prompt):
    entry_by_name = grouped_by_prompt[prompt]
    dense_lat = entry_by_name["dense"]["latency"]
    print(f"prompt: {prompt}")
    for mb in modified_backends:
        try:
            speedup = dense_lat / entry_by_name[mb]["latency"]
            print(f"  {mb} speedup: {speedup:.2f}")
        except:
            print(f"  no run for {mb}")