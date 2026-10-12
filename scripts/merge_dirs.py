import shutil
import sys
from pathlib import Path
from typing import Annotated

import tyro
from tyro.conf import arg


def iter_files(root: Path):
    """Recursively yield relative paths of all files under root."""
    for path in root.rglob("*"):
        if path.is_file():
            yield path.relative_to(root)


def main(
    dir_a: Annotated[Path, arg(aliases=["-a"])],
    dir_b: Annotated[Path, arg(aliases=["-b"])],
    dir_out: Annotated[Path, arg(aliases=["-o"])],
):
    """Merge two folders with identical subfolder structure into a new folder.

    Source folders are not modified. All copied files retain their original
    names prefixed with their source root directory name (e.g. dir_a_image.jpg)
    to avoid collisions and maintain traceability.

    Args:
        dir_a: First source folder.
        dir_b: Second source folder.
        dir_out: Destination folder (must not exist yet).
    """
    # check source folders
    if not dir_a.is_dir():
        print(f"Error: Source folder '{dir_a}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not dir_b.is_dir():
        print(f"Error: Source folder '{dir_b}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if dir_out.exists():
        print(f"Error: Destination '{dir_out}' already exists.", file=sys.stderr)
        sys.exit(1)

    # create destination
    dir_out.mkdir(parents=True)

    # collect all files retaining their source folder origin
    all_files: list[tuple[Path, Path]] = [
        (dir_a, rel) for rel in iter_files(dir_a)
    ] + [
        (dir_b, rel) for rel in iter_files(dir_b)
    ]

    # sort for reproducibility
    all_files.sort(key=lambda x: (x[1], str(x[0])))

    for base_dir, rel in all_files:
        src = base_dir / rel
        if not src.is_file():
            continue

        out_subdir = dir_out / rel.parent
        out_subdir.mkdir(parents=True, exist_ok=True)

        # prefix with source directory name
        new_name = f"{base_dir.name}_{rel.name}"
        dst = out_subdir / new_name

        # safety: avoid accidental overwrites
        counter = 1
        stem = f"{base_dir.name}_{rel.stem}"
        ext = rel.suffix
        while dst.exists():
            dst = out_subdir / f"{stem}_{counter}{ext}"
            counter += 1

        shutil.copy2(src, dst)
        print(f"{src}  ->  {dst}")

    print(f"\n>>> Success! Merged dataset saved to: {dir_out.resolve()}")


if __name__ == "__main__":
    tyro.cli(main)