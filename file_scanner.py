import os
import shutil
from collections import defaultdict, Counter
from datetime import datetime

# Configuration - Update these paths as needed
TARGET_EXTS = (".dwg", ".dgn", ".las", ".e57", ".laz")
CAD_EXTS = (".dwg", ".dgn")
DEST_DIR = os.getenv("DEST_DIR", r"C:\temp\CAD_Files")

def human_size(num):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} PB"

def find_files(root_dir, extensions):
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(extensions):
                full = os.path.join(dirpath, name)
                try:
                    stat = os.stat(full)
                    size = stat.st_size
                    mtime = stat.st_mtime
                except OSError:
                    size = 0
                    mtime = 0
                matches.append((full, name, os.path.splitext(name)[1].lower(), dirpath, size, mtime))
    return matches

def print_listing(matches):
    print(f"Found {len(matches)} matching files\n")
    for full, _, _, _, _, _ in matches:
        print(full)
    print("\nSummary report follows\n")

def report(matches):
    if not matches:
        print("No matching files were found")
        return

    counts_by_ext = Counter(ext for _, _, ext, _, _, _ in matches)
    size_by_ext = defaultdict(int)
    for _, _, ext, _, size, _ in matches:
        size_by_ext[ext] += size

    las_like = [m for m in matches if m[2] in (".las", ".laz")]
    las_counts_by_dir = Counter(m[3] for m in las_like)
    las_sizes_by_dir = defaultdict(int)
    for _, _, _, d, size, _ in las_like:
        las_sizes_by_dir[d] += size

    top_dirs = las_counts_by_dir.most_common(10)
    cad_count = counts_by_ext.get(".dwg", 0) + counts_by_ext.get(".dgn", 0)

    total_size = sum(s for _, _, _, _, s, _ in matches)
    mtimes = [m for *_, m in matches if m]
    first_seen = datetime.fromtimestamp(min(mtimes)).strftime("%Y-%m-%d %H:%M:%S") if mtimes else "unknown"
    last_seen = datetime.fromtimestamp(max(mtimes)).strftime("%Y-%m-%d %H:%M:%S") if mtimes else "unknown"

    print("Totals by file type")
    for ext in sorted(counts_by_ext):
        cnt = counts_by_ext[ext]
        sz = human_size(size_by_ext[ext])
        print(f"  {ext}: {cnt} files, {sz}")

    print("\nCAD drawing count")
    print(f"  Total DWG and DGN files: {cad_count}")

    print("\nLAS and LAZ files by directory")
    for d, cnt in sorted(las_counts_by_dir.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {d}  ,  {cnt} files  ,  {human_size(las_sizes_by_dir[d])}")

    print("\nTop directories by LAS and LAZ count")
    for d, cnt in top_dirs:
        print(f"  {d}  ,  {cnt} files  ,  {human_size(las_sizes_by_dir[d])}")

    print("\nAdditional metrics")
    print(f"  Total matched size: {human_size(total_size)}")
    print(f"  Unique directories containing matches: {len(set(m[3] for m in matches))}")
    print(f"  Earliest modified among matches: {first_seen}")
    print(f"  Latest modified among matches: {last_seen}")

def copy_cad(matches, dest_dir=None):
    if dest_dir is None:
        dest_dir = DEST_DIR
    
    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    for full, name, ext, _, _, _ in matches:
        if ext in CAD_EXTS:
            dest_path = os.path.join(dest_dir, name)
            try:
                shutil.copy2(full, dest_path)
                copied += 1
            except Exception as e:
                print(f"Failed to copy {full}: {e}")
    print(f"\nCopied {copied} CAD files (.dwg, .dgn) to {dest_dir}")

def main():
    import sys
    
    # Get root directory from command line argument or use default
    if len(sys.argv) > 1:
        root_directory = sys.argv[1]
    else:
        root_directory = os.getenv("SCAN_ROOT", r"D:\\")
    
    print(f"Scanning directory: {root_directory}")
    matches = find_files(root_directory, TARGET_EXTS)

    print_listing(matches)
    report(matches)
    
    # Ask user if they want to copy CAD files
    if any(m[2] in CAD_EXTS for m in matches):
        response = input("\nDo you want to copy CAD files to destination directory? (y/n): ")
        if response.lower() in ['y', 'yes']:
            copy_cad(matches)

if __name__ == "__main__":
    main()
