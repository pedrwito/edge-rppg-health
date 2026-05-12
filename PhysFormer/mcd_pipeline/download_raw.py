"""Download raw MCD-rPPG videos + ground-truth signals to disk, no preprocessing.

Stores .avi files under data/mcd_videos/ (flat) and signals under data/mcd_raw/
mirroring the HF repo layout. Resumable: skips files already present locally.
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

REPO_ID = "kyegorov/mcd_rppg"


def pick_subjects(all_files: list[str], n_subjects: int, state: str) -> list[str]:
    vids = [f for f in all_files if f.startswith("video/") and f.endswith(".avi")]
    if state != "all":
        vids = [f for f in vids if f.endswith(f"_{state}.avi")]
    subjects = sorted({os.path.basename(f).split("_")[0] for f in vids})[:n_subjects]
    return subjects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_subjects", type=int, default=250)
    ap.add_argument("--state", default="all", choices=["before", "after", "all"])
    ap.add_argument("--out_root", type=Path, default=Path("data"))
    args = ap.parse_args()

    videos_root = args.out_root / "mcd_videos"
    signals_root = args.out_root / "mcd_raw"
    videos_root.mkdir(parents=True, exist_ok=True)
    signals_root.mkdir(parents=True, exist_ok=True)

    print(f"[list]     fetching repo file list...")
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    subjects = pick_subjects(all_files, args.n_subjects, args.state)
    print(f"[subjects] {len(subjects)} selected: {subjects[0]}..{subjects[-1]}")

    state_suffix = "" if args.state == "all" else f"_{args.state}"
    videos = [f for f in all_files
              if f.startswith("video/") and f.endswith(".avi")
              and os.path.basename(f).split("_")[0] in set(subjects)
              and (args.state == "all" or f.endswith(f"_{args.state}.avi"))]
    print(f"[videos]   {len(videos)} files to fetch (~{len(videos)*36/1024:.1f} GB)")

    # Pull metadata + GT signals (small) via snapshot_download in one call
    meta_patterns = ["db.csv", "README.md", "readme.txt"]
    for sid in subjects:
        meta_patterns.append(f"meta/{sid}_*")
        meta_patterns.append(f"ppg/{sid}_*")
        meta_patterns.append(f"ppg_sync/{sid}_*")
        meta_patterns.append(f"ecg/{sid}_*")
    print(f"[gt]       snapshot_download for {len(subjects)} subjects' signals + meta")
    snapshot_download(REPO_ID, repo_type="dataset", local_dir=str(signals_root),
                      allow_patterns=meta_patterns)

    # Pull videos one-by-one into flat data/mcd_videos/ dir; skip existing
    done = 0
    skipped = 0
    for vrepo in videos:
        basename = os.path.basename(vrepo)
        target = videos_root / basename
        if target.exists() and target.stat().st_size > 0:
            skipped += 1
            done += 1
            if done % 50 == 0:
                print(f"[progress] {done}/{len(videos)} ({skipped} already present)")
            continue
        print(f"[download] {basename}")
        local = hf_hub_download(REPO_ID, vrepo, repo_type="dataset",
                                local_dir=str(args.out_root / "_hf_cache"))
        # Move into flat dir
        os.replace(local, target)
        done += 1
        if done % 50 == 0:
            print(f"[progress] {done}/{len(videos)}")

    # Clean up the staging cache dir
    cache_dir = args.out_root / "_hf_cache"
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)

    print(f"[done]     {done} videos in {videos_root} ({skipped} were already present)")


if __name__ == "__main__":
    main()
