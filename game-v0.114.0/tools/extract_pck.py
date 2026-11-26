"""从 Godot 4 的 .pck（格式 v2/v3、目录未加密）中提取指定前缀的文件。

用法：python3 -I tools/extract_pck.py <pck 路径> <输出目录> [前缀 ...]
前缀形如 localization/，不写前缀时只列出目录。
"""
import os
import struct
import sys


def read_directory(f):
    magic, fmt, major, minor, patch, flags = struct.unpack('<4s5I', f.read(24))
    if magic != b'GDPC':
        raise ValueError('不是 pck 文件')
    file_base = struct.unpack('<Q', f.read(8))[0]
    if fmt >= 3:
        dir_offset = struct.unpack('<Q', f.read(8))[0]
    else:
        f.read(16 * 4)
        dir_offset = f.tell()
    if flags & 1:
        raise ValueError('目录已加密，无法提取')
    if flags & 2:
        rel = file_base
    else:
        rel = 0
    f.seek(dir_offset)
    count = struct.unpack('<I', f.read(4))[0]
    entries = []
    for _ in range(count):
        n = struct.unpack('<I', f.read(4))[0]
        path = f.read(n).rstrip(b'\0').decode('utf-8')
        offset, size = struct.unpack('<QQ', f.read(16))
        f.read(16)
        eflags = struct.unpack('<I', f.read(4))[0]
        entries.append((path, offset + rel, size, eflags))
    return (major, minor, patch), entries


def main():
    pck, out = sys.argv[1], sys.argv[2]
    prefixes = sys.argv[3:]
    with open(pck, 'rb') as f:
        version, entries = read_directory(f)
        print(f'Godot {version[0]}.{version[1]}.{version[2]}，共 {len(entries)} 个文件')
        if not prefixes:
            for path, _, size, _ in entries:
                print(size, path)
            return
        n = 0
        for path, offset, size, eflags in entries:
            rel_path = path.removeprefix('res://')
            if not any(rel_path.startswith(p) for p in prefixes):
                continue
            if eflags & 1:
                raise ValueError(f'{path} 已加密')
            f.seek(offset)
            dest = os.path.join(out, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, 'wb') as g:
                g.write(f.read(size))
            n += 1
        print(f'提取 {n} 个文件到 {out}')


if __name__ == '__main__':
    main()
