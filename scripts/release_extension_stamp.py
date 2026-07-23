#!/usr/bin/env python3
"""Write or verify the content-bound marker for a frozen full CK extension."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
from pathlib import Path


CHUNK_SIZE = 8 * 1024 * 1024
MODULE_NAME = "flash_attn_2_cuda"


def content_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            hasher.update(chunk)
    return base64.b64encode(hasher.digest()).decode("ascii")


def expected_marker(
    extension: Path,
    source_manifest: Path,
    gpu_arch: str,
) -> dict[str, object]:
    return {
        "extension": extension.name,
        "gpu_arch": gpu_arch,
        "profile": "release",
        "minimal_debug": False,
        "content_digest": content_digest(extension),
        "generated_source_manifest": source_manifest.name,
        "generated_source_manifest_digest": content_digest(source_manifest),
    }


def validate_input_files(extension: Path, source_manifest: Path) -> tuple[Path, Path]:
    if extension.is_symlink():
        raise RuntimeError("the local extension must not be a symbolic link")
    extension = extension.resolve()
    source_manifest = source_manifest.resolve()
    if not extension.is_file():
        raise RuntimeError(f"extension does not exist: {extension}")
    if not source_manifest.is_file():
        raise RuntimeError(f"generated-source manifest does not exist: {source_manifest}")
    return extension, source_manifest


def write_marker(
    extension: Path,
    stamp: Path,
    source_manifest: Path,
    gpu_arch: str,
) -> None:
    extension, source_manifest = validate_input_files(extension, source_manifest)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(
        json.dumps(
            expected_marker(extension, source_manifest, gpu_arch),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Recorded frozen full extension marker: {stamp}")


def check_marker(
    extension: Path,
    stamp: Path,
    source_manifest: Path,
    gpu_arch: str,
) -> None:
    extension, source_manifest = validate_input_files(extension, source_manifest)
    if stamp.is_symlink():
        raise RuntimeError("the full-build marker must not be a symbolic link")
    if not stamp.is_file():
        raise RuntimeError("no successful frozen full-build marker; run make build-full")

    recorded = json.loads(stamp.read_text(encoding="utf-8"))
    expected = expected_marker(extension, source_manifest, gpu_arch)
    for field in (
        "extension",
        "gpu_arch",
        "profile",
        "minimal_debug",
        "generated_source_manifest",
    ):
        if recorded.get(field) != expected[field]:
            raise RuntimeError(f"full-build marker has the wrong {field}")
    if recorded.get("content_digest") != expected["content_digest"]:
        raise RuntimeError("full-build marker does not match extension content")
    if (
        recorded.get("generated_source_manifest_digest")
        != expected["generated_source_manifest_digest"]
    ):
        raise RuntimeError("full-build marker does not match generated-source manifest")

    importlib.invalidate_caches()
    module = importlib.import_module(MODULE_NAME)
    loaded_path = Path(module.__file__).resolve()
    if loaded_path != extension:
        raise RuntimeError(f"loaded extension is not local: {loaded_path}")
    print(f"Local frozen full extension: {loaded_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("write", "check"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--extension", type=Path, required=True)
        subparser.add_argument("--stamp", type=Path, required=True)
        subparser.add_argument("--source-manifest", type=Path, required=True)
        subparser.add_argument("--gpu-arch", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "write":
        write_marker(
            args.extension,
            args.stamp,
            args.source_manifest,
            args.gpu_arch,
        )
    else:
        check_marker(
            args.extension,
            args.stamp,
            args.source_manifest,
            args.gpu_arch,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
