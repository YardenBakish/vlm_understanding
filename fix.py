import os
import re
from collections import defaultdict

base_dir = "dataset/tempcomp/videos"   # directory with subdirs
work_dir = "check_res2"   # directory with txt/pt files

# collect subdir names from base_dir
subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# prepare grouping
files_by_subdir = defaultdict(list)



# find matching files
for fname in os.listdir(work_dir):
    for subdir in subdirs:
        if fname.startswith(subdir):
            files_by_subdir[subdir].append(fname)
            break



# rename files
for subdir, files in files_by_subdir.items():
    files.sort()  # ensure consistent ordering
    for i, fname in enumerate(files, start=1):
        old_path = os.path.join(work_dir, fname)

        # separate extension
        root, ext = os.path.splitext(fname)

        # remove existing trailing numbers or underscores
        new_root = re.sub(r"(_\d+)*$", "", root)



        new_name = f"{subdir}_{i}{ext}"
        new_path = os.path.join(work_dir, new_name)

        print(f"Renaming {old_path} -> {new_path}")
        exit(1)
        os.rename(old_path, new_path)