import os

def convert_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024

files = []

for root, dirs, filenames in os.walk("."):
    for f in filenames:
        path = os.path.join(root, f)
        size = os.path.getsize(path)
        files.append((path, size))

files.sort(key=lambda x: x[1], reverse=True)

for file, size in files[:20]:
    print(file, "-", convert_size(size))