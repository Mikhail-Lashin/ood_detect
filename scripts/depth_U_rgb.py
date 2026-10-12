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
    depth_dir: Annotated[Path, arg(aliases=["-d"])],
    dir_out: Annotated[Path, arg(aliases=["-o"])],
    rgb_a: Annotated[Path, arg(aliases=["-a"], help="Source RGB root directory for dataset A.")],
    rgb_b: Annotated[Path, arg(aliases=["-b"], help="Source RGB root directory for dataset B.")],
    prefix_a: str = "a",
    prefix_b: str = "b",
    rgb_subfolder: str = "rgb_seg",
    symlink: bool = False,
):
    """Pair merged depth dataset with corresponding RGB images into a unified structure.

    Inspects the existing depth dataset splits (train/eval/test/normal|anomalous),
    matches each depth map to its source RGB image from rgb_a or rgb_b (under
    <class>/<rgb_subfolder>/), and constructs depth and rgb datasets inside dir_out.

    Args:
        depth_dir: Path to the already merged depth dataset.
        dir_out: Target parent directory where named depth/rgb datasets will be created.
        rgb_a: Root directory containing 'normal' and 'anomalous' folders for dataset A.
        rgb_b: Root directory containing 'normal' and 'anomalous' folders for dataset B.
        prefix_a: Prefix used for dataset A files in merged depth (without trailing underscore).
        prefix_b: Prefix used for dataset B files in merged depth (without trailing underscore).
        rgb_subfolder: Name of the RGB subfolder inside normal/anomalous (default: 'rgb_seg').
        symlink: Create symbolic links instead of copying files.
    """
    # map datasets to their specific rgb subdirectories
    rgb_dirs = {
        (prefix_a, "normal"): rgb_a / "normal" / rgb_subfolder,
        (prefix_a, "anomalous"): rgb_a / "anomalous" / rgb_subfolder,
        (prefix_b, "normal"): rgb_b / "normal" / rgb_subfolder,
        (prefix_b, "anomalous"): rgb_b / "anomalous" / rgb_subfolder,
    }

    # define target dataset roots once
    out_depth_root = dir_out / f"{dir_out.name}_depth"
    out_rgb_root = dir_out / f"{dir_out.name}_rgb"

    # validate directories
    if not depth_dir.is_dir():
        print(f"Error: Depth directory '{depth_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    for (pfx, label), path in rgb_dirs.items():
        if not path.is_dir():
            print(f"Error: Required RGB folder '{path}' does not exist.", file=sys.stderr)
            sys.exit(1)

    if out_depth_root.exists():
        print(f"Error: Destination folder '{out_depth_root}' already exists.", file=sys.stderr)
        sys.exit(1)
    if out_rgb_root.exists():
        print(f"Error: Destination folder '{out_rgb_root}' already exists.", file=sys.stderr)
        sys.exit(1)

    # index rgb files flatly per group for fast lookup: filename -> absolute path
    rgb_lookup: dict[tuple[str, str], dict[str, Path]] = {}
    for key, folder in rgb_dirs.items():
        rgb_lookup[key] = {p.name: p for p in folder.iterdir() if p.is_file()}

    # collect pairs and validate before copying
    depth_files = list(iter_files(depth_dir))
    if not depth_files:
        print(f"Error: No files found in depth directory '{depth_dir}'.", file=sys.stderr)
        sys.exit(1)

    tasks: list[tuple[Path, Path, Path, Path]] = []  # (src_depth, dst_depth, src_rgb, dst_rgb)
    missing_files: list[str] = []

    prefix_a_tag = f"{prefix_a}_"
    prefix_b_tag = f"{prefix_b}_"

    for rel in depth_files:
        filename = rel.name
        class_label = rel.parent.name  # 'normal' or 'anomalous'

        # determine source prefix and extract original name
        if filename.startswith(prefix_a_tag):
            pfx = prefix_a
            orig_name = filename[len(prefix_a_tag):]
        elif filename.startswith(prefix_b_tag):
            pfx = prefix_b
            orig_name = filename[len(prefix_b_tag):]
        else:
            missing_files.append(f"Cannot determine prefix for file: {rel}")
            continue

        if class_label not in ("normal", "anomalous"):
            missing_files.append(f"Unexpected class directory '{class_label}' for file: {rel}")
            continue

        # lookup corresponding rgb
        target_group = rgb_lookup.get((pfx, class_label), {})
        rgb_src = target_group.get(orig_name)

        if rgb_src is None:
            missing_files.append(
                f"RGB counterpart not found: '{orig_name}' in group ({pfx}, {class_label}/{rgb_subfolder})"
            )
            continue

        depth_src = depth_dir / rel
        depth_dst = out_depth_root / rel
        rgb_dst = out_rgb_root / rel

        tasks.append((depth_src, depth_dst, rgb_src, rgb_dst))

    # fail early if any file pairs are missing
    if missing_files:
        print(f"Error: Dataset validation failed with {len(missing_files)} issue(s):", file=sys.stderr)
        for issue in missing_files[:20]:
            print(f"  - {issue}", file=sys.stderr)
        if len(missing_files) > 20:
            print(f"  ... and {len(missing_files) - 20} more.", file=sys.stderr)
        sys.exit(1)

    # transfer files
    transfer_func = (lambda s, d: d.symlink_to(s.resolve())) if symlink else shutil.copy2
    action_label = "Linked" if symlink else "Copied"

    for depth_src, depth_dst, rgb_src, rgb_dst in tasks:
        depth_dst.parent.mkdir(parents=True, exist_ok=True)
        rgb_dst.parent.mkdir(parents=True, exist_ok=True)

        transfer_func(depth_src, depth_dst)
        transfer_func(rgb_src, rgb_dst)

    print(f"\n>>> Success! {action_label} {len(tasks)} image pairs.")
    print(f"    Depth dataset: {out_depth_root.resolve()}")
    print(f"    RGB dataset:   {out_rgb_root.resolve()}")


if __name__ == "__main__":
    tyro.cli(main)