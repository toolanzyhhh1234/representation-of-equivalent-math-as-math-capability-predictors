"""Build checksum-indexed experiment snapshots; validate all paths before restoring."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

from .config import ROOT


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build(destination, paths, name):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = {}
    for arg in paths:
        base = (ROOT/arg).resolve()
        if not base.is_relative_to(ROOT) or not base.exists():
            raise ValueError(f"Missing/outside repository: {arg}")
        if destination.is_relative_to(base): raise ValueError("Snapshot destination is inside an input")
        for path in sorted(base.rglob("*") if base.is_dir() else [base]):
            if path.is_symlink(): raise ValueError(f"Refusing symlink: {path}")
            if not path.is_file(): continue
            if any(p in (".cache", "__pycache__", ".git", ".venv") for p in path.parts): continue
            if path.suffix in (".log", ".tmp", ".pyc"): continue
            rel = path.relative_to(ROOT).as_posix()
            files[rel] = {"size": path.stat().st_size, "sha256": sha(path)}
    if not files: raise ValueError("Empty snapshot")
    data_tar = destination/"data.tar.gz"
    with tarfile.open(data_tar, "w:gz") as tar:
        for rel in sorted(files): tar.add(ROOT/rel, arcname=rel, recursive=False)
    code_tar = destination/"code.tar.gz"
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(code_tar), commit], cwd=ROOT, check=True)
    manifest = {"schema": 1, "name": name, "git_commit": commit, "files": files,
                "archives": {p.name: {"size": p.stat().st_size, "sha256": sha(p)} for p in (data_tar, code_tar)}}
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"name": name, "git_commit": commit, "files": len(files),
                      "uncompressed_bytes": sum(f["size"] for f in files.values()),
                      "archives": manifest["archives"]}, indent=2))


def verify_restore(snapshot, target=None):
    snapshot = Path(snapshot)
    manifest = json.loads((snapshot/"manifest.json").read_text())
    if set(manifest["archives"]) != {"data.tar.gz", "code.tar.gz"}:
        raise ValueError("Unexpected archive names")
    for name, info in manifest["archives"].items():
        path = snapshot/name
        if path.stat().st_size != info["size"] or sha(path) != info["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {name}")
    # Validate every member and every conflict before writing anything.
    with tarfile.open(snapshot/"data.tar.gz", "r:gz") as tar:
        members = tar.getmembers()
        if len(members) != len(manifest["files"]): raise ValueError("Member count mismatch")
        seen = set()
        for member in members:
            rel = PurePosixPath(member.name)
            if not member.isfile() or rel.is_absolute() or ".." in rel.parts or member.name in seen:
                raise ValueError(f"Unsafe/duplicate member: {member.name}")
            seen.add(member.name)
            expected = manifest["files"][member.name]
            stream = tar.extractfile(member)
            if member.size != expected["size"] or hashlib.file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
                raise ValueError(f"Data checksum mismatch: {member.name}")
            if target is not None:
                root = Path(target).resolve()
                if (root/member.name).is_symlink(): raise ValueError("Refusing existing symlink")
                dest = (root/member.name).resolve()
                if not dest.is_relative_to(root): raise ValueError("Restore traverses a symlink outside target")
                if dest.exists() and (not dest.is_file() or sha(dest) != expected["sha256"]):
                    raise ValueError(f"Refusing to replace different data: {dest}")
        if set(seen) != set(manifest["files"]): raise ValueError("Manifest member mismatch")
        if target is not None:
            for member in members:
                dest = Path(target)/member.name
                if dest.exists(): continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, dest.open("wb") as out:
                    while chunk := source.read(1024*1024): out.write(chunk)
    print(f"Verified {len(members)} data files and both archive checksums" + ("; restored." if target else "."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build"); b.add_argument("destination"); b.add_argument("--name", required=True); b.add_argument("paths", nargs="+")
    v = sub.add_parser("verify"); v.add_argument("snapshot"); v.add_argument("--restore-to")
    args = parser.parse_args()
    if args.command == "build": build(args.destination, args.paths, args.name)
    else: verify_restore(args.snapshot, args.restore_to)


if __name__ == "__main__": main()
