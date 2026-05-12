"""Stage 0 (UBFC variant): MediaPipe FaceMesh on UBFC's subject-folder layout.

UBFC2 ships videos as <UBFC_root>/subjectN/vid.avi. We reuse `process_one`
from detect_landmarks.py and just feed it the right (video_path, out_dir)
pairs, naming each output by the subject folder (since `vid.avi` collides).

Output: data/face_landmarks_ubfc/<subject>.npz
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from mcd_pipeline.detect_landmarks import process_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_root", type=Path, required=True,
                    help="UBFC dataset root (contains subject1/, subject2/, ...)")
    ap.add_argument("--out_root", type=Path, default=Path("data/face_landmarks_ubfc"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    # Each subject has exactly one vid.avi
    videos = sorted(args.videos_root.glob("*/vid.avi"))
    if args.limit:
        videos = videos[:args.limit]
    print(f"[scan]   {len(videos)} UBFC videos, workers={args.workers}")

    # Symlink-or-rename trick: process_one names the output by video stem
    # (which would be 'vid' for every subject). Instead, build pairs with a
    # per-subject out_dir trick: pass out_dir but rely on caller naming.
    # Simplest: temporarily copy logic — call process_one then rename.
    work = []
    for v in videos:
        subject = v.parent.name
        # process_one writes <out_dir>/<v.stem>.npz == <out_dir>/vid.npz, so use
        # per-subject out_dir then move. Easier: use a sentinel temp dir per video.
        tmp_dir = args.out_root / f"_tmp_{subject}"
        tmp_dir.mkdir(exist_ok=True)
        work.append((v, tmp_dir, subject))

    done = skipped = errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fut_map = {ex.submit(process_one, (v, tmp)): (v, tmp, subj)
                   for v, tmp, subj in work}
        for fut in as_completed(fut_map):
            v, tmp, subject = fut_map[fut]
            name, T, status = fut.result()
            final = args.out_root / f"{subject}.npz"
            src = tmp / "vid.npz"
            if status == "skip" or src.exists():
                if src.exists() and not final.exists():
                    src.rename(final)
                if status == "skip":
                    skipped += 1
                else:
                    done += 1
            elif status is not None:
                errors += 1
                print(f"[err]    {subject}: {status} (T={T})")
            # cleanup tmp
            try:
                tmp.rmdir()
            except OSError:
                pass
    print(f"[done]   {done} new, {skipped} skipped, {errors} errors -> {args.out_root}")


if __name__ == "__main__":
    main()
