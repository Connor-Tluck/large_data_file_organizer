import os
import re
import shutil
import hashlib
from pathlib import Path

# Configuration - Update these paths as needed
CAD_DIR = Path(os.getenv("CAD_DIR", r"C:\path\to\your\cad\files"))
QUAR_DIR = CAD_DIR / "_Duplicates_Quarantine"
CAD_EXTS = {".dwg", ".dgn"}

suffix_re = re.compile(r"^(?P<base>.+)_(\d+)$")  # matches name_1.dwg, name_2.dgn

def md5sum(p, chunk=1024 * 1024):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def base_name_no_suffix(stem):
    m = suffix_re.match(stem)
    return m.group("base") if m else stem

def find_cad_files(folder):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in CAD_EXTS]

def clean_dupes(cad_dir=None):
    """
    Clean duplicate CAD files by moving exact duplicates to quarantine folder.
    
    Args:
        cad_dir (str, optional): Path to CAD directory. If None, uses CAD_DIR from config.
    """
    if cad_dir:
        cad_path = Path(cad_dir)
    else:
        cad_path = CAD_DIR
    
    if not cad_path.exists():
        print(f"Folder not found: {cad_path}")
        print("Please set the CAD_DIR environment variable or pass cad_dir parameter")
        return

    quar_path = cad_path / "_Duplicates_Quarantine"
    quar_path.mkdir(exist_ok=True)

    files = find_cad_files(cad_path)

    # map base name without numeric suffix to list of paths
    groups = {}
    for p in files:
        base = base_name_no_suffix(p.stem) + p.suffix.lower()
        groups.setdefault(base, []).append(p)

    moved = 0
    checked = 0

    for base_key, paths in groups.items():
        if len(paths) < 2:
            continue

        # prefer the version without numeric suffix as canonical if present
        canon = None
        for p in paths:
            if p.stem == Path(base_key).stem:
                canon = p
                break
        if canon is None:
            # choose the earliest modified as canonical
            canon = min(paths, key=lambda x: x.stat().st_mtime)

        # compute canonical hash once
        try:
            canon_hash = md5sum(canon)
        except Exception as e:
            print(f"Could not hash {canon}, {e}")
            continue

        for p in paths:
            if p == canon:
                continue
            checked += 1
            try:
                # quick size check first
                if p.stat().st_size != canon.stat().st_size:
                    continue
                # confirm by hash
                if md5sum(p) == canon_hash:
                    target = quar_path / p.name
                    # avoid collision in quarantine
                    i = 1
                    while target.exists():
                        target = quar_path / f"{p.stem}_{i}{p.suffix}"
                        i += 1
                    shutil.move(str(p), str(target))
                    moved += 1
                    print(f"Duplicate moved: {p.name} -> canonical is {canon.name}")
            except Exception as e:
                print(f"Error processing {p}, {e}")

    print(f"\nDone, checked {checked} potential duplicates, moved {moved} exact duplicates to {quar_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        clean_dupes(sys.argv[1])
    else:
        clean_dupes()
