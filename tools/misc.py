import os

def mkdirpath(path):
    """make all folders in a path"""
    parts = split_path(path)
    full_path = ''
    for part in parts:
        full_path = os.path.join(full_path, part)
        if not os.path.isdir(full_path):
            os.mkdir(full_path)

def split_path(path):
    """fully split a path"""
    parts = []
    path, end = os.path.split(path)
    while end:
        parts.append(end)
        path, end = os.path.split(path)

    if path:
        parts.append(path)
    parts.reverse()
    return parts

def get_all_files_walk(folder):
    """Get all files from an os.walk"""
    files = []
    for root, dirs, filenames in os.walk(folder):
        files.extend(os.path.join(root, f) for f in filenames)
    return files
