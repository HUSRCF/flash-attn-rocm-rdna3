# Copyright (c) 2023, Tri Dao.

import sys
import functools
import warnings
import os
import re
import ast
import glob
import shutil
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    ROCM_HOME,
    IS_HIP_EXTENSION,
)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_ROCM = True
    else:
        IS_ROCM = False
else:
    if BUILD_TARGET == "cuda":
        IS_ROCM = False
    elif BUILD_TARGET == "rocm":
        IS_ROCM = True

PACKAGE_NAME = "flash_attn"

BASE_WHEEL_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/{tag_name}/{wheel_name}"
)

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FLASH_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"
USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
SKIP_CK_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CK_BUILD", "TRUE") == "TRUE" if USE_TRITON_ROCM else False
NVCC_THREADS = os.getenv("NVCC_THREADS") or "16"


def _env_flag(name: str, default: str = "FALSE") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

# CK 极简调试模式：仅编译极小实例集合，快速定位 kernel 问题
CK_MINIMAL_DEBUG = _env_flag("FLASH_ATTN_CK_MINIMAL_DEBUG", "FALSE")
CK_MINIMAL_FAMILY = os.getenv("FLASH_ATTN_CK_MINIMAL_FAMILY", "standard").strip().lower()
CK_MINIMAL_OPTDIM = os.getenv("FLASH_ATTN_CK_MINIMAL_OPTDIM", "64")
CK_MINIMAL_DTYPE = os.getenv("FLASH_ATTN_CK_MINIMAL_DTYPE", "fp16").lower()
CK_MINIMAL_NONCAUSAL_ONLY = _env_flag("FLASH_ATTN_CK_MINIMAL_NONCAUSAL_ONLY", "TRUE")
CK_MINIMAL_BATCH_ONLY = _env_flag("FLASH_ATTN_CK_MINIMAL_BATCH_ONLY", "TRUE")
CK_FWD_WMMA_NATIVE_O_EPILOGUE = os.getenv("CK_TILE_FMHA_FWD_WMMA_NATIVE_O_EPILOGUE")
CK_BWD_WMMA_PT_LDS_REMAP = os.getenv("CK_TILE_FMHA_BWD_WMMA_PT_LDS_REMAP")
CK_BWD_WMMA_SGRADT_LDS_REMAP = os.getenv("CK_TILE_FMHA_BWD_WMMA_SGRADT_LDS_REMAP")
CK_BWD_WMMA_PT_REGISTER_REMAP = os.getenv("CK_TILE_FMHA_BWD_WMMA_PT_REGISTER_REMAP")
CK_BWD_WMMA_SGRADT_REGISTER_REMAP = os.getenv(
    "CK_TILE_FMHA_BWD_WMMA_SGRADT_REGISTER_REMAP"
)
CK_BWD_WMMA_PT_REGISTER_REMAP_D64_ONLY = os.getenv(
    "CK_TILE_FMHA_BWD_WMMA_PT_REGISTER_REMAP_D64_ONLY"
)
CK_BWD_WMMA_SGRADT_REGISTER_REMAP_D64_ONLY = os.getenv(
    "CK_TILE_FMHA_BWD_WMMA_SGRADT_REGISTER_REMAP_D64_ONLY"
)
CK_BWD_LAYOUT_DIAG = os.getenv("CK_TILE_FMHA_BWD_LAYOUT_DIAG")
CK_BWD_SPLIT_DISPATCH = os.getenv("CK_TILE_FMHA_BWD_SPLIT_DISPATCH")
CK_DEBUG_BWD_FORCE_GEMM34_BLOCK_SYNC = os.getenv("CK_TILE_DEBUG_BWD_FORCE_GEMM34_BLOCK_SYNC")
CK_DEBUG_BWD_FORCE_LDS_WAITCNT = os.getenv("CK_TILE_DEBUG_BWD_FORCE_LDS_WAITCNT")
CK_DEBUG_BWD_GEMM3_SCHED_BARRIER = os.getenv("CK_TILE_DEBUG_BWD_GEMM3_SCHED_BARRIER")
CK_DEBUG_BREG_VMCNT_BEFORE_WMMA = os.getenv("CK_TILE_DEBUG_BREG_VMCNT_BEFORE_WMMA")
CK_DEBUG_DRAIN_VMCNT_BEFORE_2D_EPILOGUE_STORE = os.getenv(
    "CK_TILE_DEBUG_DRAIN_VMCNT_BEFORE_2D_EPILOGUE_STORE"
)
CK_DEBUG_BWD_DRAIN_VMCNT_BEFORE_SGRADT_LDS_WRITE = os.getenv(
    "CK_TILE_DEBUG_BWD_DRAIN_VMCNT_BEFORE_SGRADT_LDS_WRITE"
)
CK_DEBUG_BWD_FORCE_FUSED_DQDKDV = os.getenv("CK_TILE_DEBUG_BWD_FORCE_FUSED_DQDKDV")
CK_DEBUG_BWD_DK_ONLY_USE_DKDV_PIPELINE = os.getenv(
    "CK_TILE_DEBUG_BWD_DK_ONLY_USE_DKDV_PIPELINE"
)
CK_DEBUG_BWD_DK_USE_DQDKDV_PIPELINE = os.getenv("CK_TILE_DEBUG_BWD_DK_USE_DQDKDV_PIPELINE")
CK_DEBUG_BWD_SPLIT_USE_FUSED_DKDV = os.getenv("CK_TILE_DEBUG_BWD_SPLIT_USE_FUSED_DKDV")
CK_DEBUG_BWD_SPLIT_PARALLEL_STREAMS = os.getenv("CK_TILE_DEBUG_BWD_SPLIT_PARALLEL_STREAMS")
CK_DEBUG_BWD_SPLIT_CONVERT_ON_DQ_STREAM = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_CONVERT_ON_DQ_STREAM"
)
CK_DEBUG_BWD_SPLIT_DISABLE_D256_PARALLEL_STREAMS = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DISABLE_D256_PARALLEL_STREAMS"
)
CK_ENABLE_D256_HYBRID_FAST = os.getenv("CK_TILE_ENABLE_D256_HYBRID_FAST")
CK_DEBUG_BWD_SPLIT_DQ_FORCE_DETERMINISTIC = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_FORCE_DETERMINISTIC"
)
CK_DEBUG_BWD_SPLIT_DQ_QMAJOR = os.getenv("CK_TILE_DEBUG_BWD_SPLIT_DQ_QMAJOR")
CK_DEBUG_BWD_SPLIT_DQ_QMAJOR_ALLOW_UNSTABLE = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_QMAJOR_ALLOW_UNSTABLE"
)
CK_DEBUG_BWD_SPLIT_DQ_QMAJOR_D256 = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_QMAJOR_D256"
)
CK_DEBUG_BWD_SPLIT_DQ_QMAJOR_MAX_SEQLEN = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_QMAJOR_MAX_SEQLEN"
)
CK_DEBUG_BWD_SPLIT_DQ_LATE_DO_D_PREFETCH = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_LATE_DO_D_PREFETCH"
)
CK_DEBUG_BWD_SPLIT_DQ_LATE_REG_LOAD = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_LATE_REG_LOAD"
)
CK_DEBUG_BWD_SPLIT_DQ_INPLACE_DS = os.getenv("CK_TILE_DEBUG_BWD_SPLIT_DQ_INPLACE_DS")
CK_DEBUG_BWD_SPLIT_DQ_D256_NO_INPLACE_DS = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_D256_NO_INPLACE_DS"
)
CK_DEBUG_BWD_SPLIT_DQ_D256_ALLOW_LATE_LOAD = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_D256_ALLOW_LATE_LOAD"
)
CK_DEBUG_BWD_SPLIT_DQ_D256_ALLOW_LATE_LOAD_NONMASK = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_D256_ALLOW_LATE_LOAD_NONMASK"
)
CK_DEBUG_BWD_SPLIT_DQ_D256_LATE_Q_ONLY_NONMASK = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_D256_LATE_Q_ONLY_NONMASK"
)
CK_DEBUG_BWD_SPLIT_DQ_D256_NO_DS_PREFETCH_NEXT = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_D256_NO_DS_PREFETCH_NEXT"
)
CK_DEBUG_BWD_SPLIT_DQ_SCOPE_DS_GEMM_STORE = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_SCOPE_DS_GEMM_STORE"
)
CK_DEBUG_BWD_SPLIT_DQ_STATIC_DQ_WINDOW = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_STATIC_DQ_WINDOW"
)
CK_DEBUG_BWD_SPLIT_DQ_STATIC_DQ_WINDOW_RAW = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_STATIC_DQ_WINDOW_RAW"
)
CK_DEBUG_BWD_SPLIT_DQ_D256_SINGLE_K4_FAST = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_D256_SINGLE_K4_FAST"
)
CK_DEBUG_BWD_SPLIT_DQ_D256_TAIL_DIRECT_UPDATE = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_D256_TAIL_DIRECT_UPDATE"
)
CK_DEBUG_BWD_SPLIT_DQ_D256_SKIP_NONMASK_TAIL_REMAP = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_D256_SKIP_NONMASK_TAIL_REMAP"
)
CK_DEBUG_BWD_SPLIT_DQ_TAIL_DIRECT_UPDATE = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_TAIL_DIRECT_UPDATE"
)
CK_DEBUG_BWD_SPLIT_DQ_D256_BN64 = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_D256_BN64"
)
CK_DEBUG_BWD_SPLIT_DQ_D256_HEAD_SPLIT2 = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_D256_HEAD_SPLIT2"
)
CK_DEBUG_BWD_SPLIT_DQ_D256_HEAD_SPLIT4 = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_D256_HEAD_SPLIT4"
)
CK_DEBUG_BWD_SPLIT_DQ_D256_GEMM4_ASMEM = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DQ_D256_GEMM4_ASMEM"
)
CK_DEBUG_BWD_D256_NONMASK_NO_EDGE_CHECK = os.getenv(
    "CK_TILE_DEBUG_BWD_D256_NONMASK_NO_EDGE_CHECK"
)
CK_DEBUG_BWD_DQ_D256_NONMASK_NO_EDGE_CHECK = os.getenv(
    "CK_TILE_DEBUG_BWD_DQ_D256_NONMASK_NO_EDGE_CHECK"
)
CK_DEBUG_BWD_DKDV_D256_NONMASK_NO_EDGE_CHECK = os.getenv(
    "CK_TILE_DEBUG_BWD_DKDV_D256_NONMASK_NO_EDGE_CHECK"
)
CK_DEBUG_BWD_DKDV_NONMASK_NO_EDGE_CHECK = os.getenv(
    "CK_TILE_DEBUG_BWD_DKDV_NONMASK_NO_EDGE_CHECK"
)
CK_DEBUG_BWD_SPLIT_DV_D256_NONMASK_NO_EDGE_CHECK = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DV_D256_NONMASK_NO_EDGE_CHECK"
)
CK_DEBUG_BWD_SPLIT_LAUNCH_DQ_FIRST = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_LAUNCH_DQ_FIRST"
)
CK_DEBUG_BWD_SPLIT_LAUNCH_DQ_FIRST_AUTO = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_LAUNCH_DQ_FIRST_AUTO"
)
CK_DEBUG_BWD_SPLIT_LAUNCH_DK_FIRST = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_LAUNCH_DK_FIRST"
)
CK_DEBUG_BWD_SPLIT_DKDV_PRECAST_SGRAD = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DKDV_PRECAST_SGRAD"
)
CK_DEBUG_BWD_SPLIT_DKDV_FRESH_DS = os.getenv("CK_TILE_DEBUG_BWD_SPLIT_DKDV_FRESH_DS")
CK_DEBUG_BWD_SPLIT_DKDV_FRESH_P = os.getenv("CK_TILE_DEBUG_BWD_SPLIT_DKDV_FRESH_P")
CK_DEBUG_BWD_SPLIT_DKDV_D256_INLINE_DS_REMAP = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DKDV_D256_INLINE_DS_REMAP"
)
CK_DEBUG_BWD_SPLIT_DKDV_D256_DS_LDS_MATERIALIZE = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DKDV_D256_DS_LDS_MATERIALIZE"
)
CK_DEBUG_BWD_SPLIT_DKDV_POST_GEMM3_DS_LDS_BOUNDARY = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DKDV_POST_GEMM3_DS_LDS_BOUNDARY"
)
CK_DEBUG_BWD_SPLIT_DKDV_USE_GENERIC_LDS_LAYOUT = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DKDV_USE_GENERIC_LDS_LAYOUT"
)
CK_DEBUG_BWD_SPLIT_DKDV_LATE_QDO_STORE_LAMBDA = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DKDV_LATE_QDO_STORE_LAMBDA"
)
CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DV_GEMM1 = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DV_GEMM1"
)
CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4 = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4"
)
CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_KT_ONLY = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_KT_ONLY"
)
CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_DS_LDS_ONLY = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_DS_LDS_ONLY"
)
CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_ONE_SLICE = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_ONE_SLICE"
)
CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_SINK_MODE = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_SINK_MODE"
)
CK_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS"
)
CK_DEBUG_BWD_SPLIT_DV_EARLY_Q_LSE_STORE = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DV_EARLY_Q_LSE_STORE"
)
CK_DEBUG_BWD_SPLIT_DV_GEMM1_BSMEM = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DV_GEMM1_BSMEM"
)
CK_DEBUG_BWD_SPLIT_DV_GEMM1_BSMEM_D256 = os.getenv(
    "CK_TILE_DEBUG_BWD_SPLIT_DV_GEMM1_BSMEM_D256"
)
CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT_ONLY = os.getenv(
    "CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT_ONLY"
)


def filter_ck_minimal_generated_sources(paths):
    """筛选 CK codegen 产物，仅保留调试所需最小集合。"""
    dtype_items = [item.strip() for item in CK_MINIMAL_DTYPE.split(",") if item.strip()]
    dim_tokens = [
        f"d{dim.strip()}_{dtype}"
        for dim in CK_MINIMAL_OPTDIM.split(",")
        if dim.strip()
        for dtype in dtype_items
    ]

    def has_dim_dtype(name: str) -> bool:
        return any(token in name for token in dim_tokens)

    kept = []
    for path in paths:
        name = os.path.basename(path)

        # API 文件按 family 保留，避免把其他方向的静态分派引用带入链接。
        if name.endswith("_api.cpp"):
            if CK_MINIMAL_FAMILY == "kvcache" and name not in {
                "fmha_fwd_appendkv_api.cpp",
                "fmha_fwd_splitkv_api.cpp",
            }:
                continue
            if CK_MINIMAL_FAMILY == "varlen" and name not in {
                "fmha_fwd_api.cpp",
                "fmha_fwd_splitkv_api.cpp",
            }:
                continue
            if CK_MINIMAL_FAMILY == "standard" and (
                "appendkv" in name or "splitkv" in name or "varlen" in name
            ):
                continue
            kept.append(path)
            continue

        # 仅处理 fmha kernel 实例文件
        if not name.startswith("fmha_"):
            kept.append(path)
            continue

        if not has_dim_dtype(name):
            continue

        if CK_MINIMAL_FAMILY == "kvcache":
            if name.startswith("fmha_fwd_appendkv_"):
                kept.append(path)
                continue
            if name.startswith("fmha_fwd_splitkv_"):
                if CK_MINIMAL_BATCH_ONLY and "_batch_" not in name:
                    continue
                kept.append(path)
            continue

        if CK_MINIMAL_FAMILY == "varlen":
            # mha_varlen_fwd has a statically linked paged-KV branch, so the
            # group-mode SplitKV main/combine kernels are part of its link closure.
            if (
                name.startswith("fmha_fwd_")
                and "appendkv" not in name
                and "_group_" in name
            ):
                kept.append(path)
            continue

        if CK_MINIMAL_BATCH_ONLY and "_batch_" not in name:
            continue

        # 不保留不相关路径（KV cache / splitkv / varlen）
        if "appendkv" in name or "splitkv" in name or "varlen" in name:
            continue

        # 前向：保留该最小域内所有变体，避免与 *_api.cpp 的静态引用不一致
        if name.startswith("fmha_fwd_"):
            kept.append(path)
            continue

        # 反向辅助 kernel（dot_do_o / convert_dq）
        if name.startswith("fmha_bwd_dot_do_o_"):
            kept.append(path)
            continue
        if name.startswith("fmha_bwd_convert_dq_"):
            kept.append(path)
            continue

        # 反向主 kernel
        if name.startswith("fmha_bwd_"):
            kept.append(path)
            continue

    return kept

@functools.lru_cache(maxsize=None)
def cuda_archs() -> str:
    return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;110;120").split(";")


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f'linux_{platform.uname().machine}'
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def add_cuda_gencodes(cc_flag, archs, bare_metal_version):
    """
    Adds -gencode flags based on nvcc capabilities:
      - sm_80/90 (regular)
      - sm_100/120 on CUDA >= 12.8
      - Use 100f on CUDA >= 12.9 (Blackwell family-specific)
      - Map requested 110 -> 101 if CUDA < 13.0 (Thor rename)
      - Embed PTX for newest arch for forward compatibility
    """
    # Always-regular 80
    if "80" in archs:
        cc_flag += ["-gencode", "arch=compute_80,code=sm_80"]

    # Hopper 9.0 needs >= 11.8
    if bare_metal_version >= Version("11.8") and "90" in archs:
        cc_flag += ["-gencode", "arch=compute_90,code=sm_90"]

    # Blackwell 10.x requires >= 12.8
    if bare_metal_version >= Version("12.8"):
        if "100" in archs:
            # CUDA 12.9 introduced "family-specific" for Blackwell (100f)
            if bare_metal_version >= Version("12.9"):
                cc_flag += ["-gencode", "arch=compute_100f,code=sm_100"]
            else:
                cc_flag += ["-gencode", "arch=compute_100,code=sm_100"]

        if "120" in archs:
            # sm_120 is supported in CUDA 12.8/12.9+ toolkits
            if bare_metal_version >= Version("12.9"):
                cc_flag += ["-gencode", "arch=compute_120f,code=sm_120"]
            else:
                cc_flag += ["-gencode", "arch=compute_120,code=sm_120"]


        # Thor rename: 12.9 uses sm_101; 13.0+ uses sm_110
        if "110" in archs:
            if bare_metal_version >= Version("13.0"):
                cc_flag += ["-gencode", "arch=compute_110f,code=sm_110"]
            else:
                # Provide Thor support for CUDA 12.9 via sm_101
                if bare_metal_version >= Version("12.8"):
                    cc_flag += ["-gencode", "arch=compute_101,code=sm_101"]
                # else: no Thor support in older toolkits

    # PTX for newest requested arch (forward-compat)
    numeric = [a for a in archs if a.isdigit()]
    if numeric:
        newest = max(numeric, key=int)
        cc_flag += ["-gencode", f"arch=compute_{newest},code=compute_{newest}"]

    return cc_flag


def get_hip_version():
    return parse(torch.version.hip.split()[-1].rstrip('-').replace('-', '+'))


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def check_if_rocm_home_none(global_option: str) -> None:
    if ROCM_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so hipcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but hipcc was not found."
    )


def detect_hipify_v2():
    try:
        from torch.utils.hipify import __version__
        from packaging.version import Version
        if Version(__version__) >= Version("2.0.0"):
            return True
    except Exception as e:
        print("failed to detect pytorch hipify version, defaulting to version 1.0.0 behavior")
        print(e)
    return False


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", NVCC_THREADS]


def rename_cpp_to_cu(cpp_files):
    for entry in cpp_files:
        shutil.copy(entry, os.path.splitext(entry)[0] + ".cu")


def validate_and_update_archs(archs):
    # List of allowed architectures
    allowed_archs = ["native", "gfx90a", "gfx950", "gfx942",
                     "gfx1100", "gfx1101", "gfx1102",  # RDNA3
                     "gfx1200", "gfx1201"]              # RDNA3.5

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported by Flash-Attention"


cmdclass = {}
ext_modules = []

# We want this even if SKIP_CUDA_BUILD because when we run python setup.py sdist we want the .hpp
# files included in the source distribution, in case the user compiles from source.
if os.path.isdir(".git"):
    if not SKIP_CK_BUILD:
        # RDNA3 Fix: Commented out to prevent overwriting manual patches
        # subprocess.run(["git", "submodule", "update", "--init", "csrc/composable_kernel"], check=True)
        # subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], check=True)
        pass
else:
    if IS_ROCM:
        if not SKIP_CK_BUILD:
            assert (
                os.path.exists("csrc/composable_kernel/example/ck_tile/01_fmha/generate.py")
            ), "csrc/composable_kernel is missing, please use source distribution or git clone"
    else:
        assert (
            os.path.exists("csrc/cutlass/include/cutlass/cutlass.h")
        ), "csrc/cutlass is missing, please use source distribution or git clone"

if not SKIP_CUDA_BUILD and not IS_ROCM:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    check_if_cuda_home_none("flash_attn")
    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version < Version("11.7"):
            raise RuntimeError(
                "FlashAttention is only supported on CUDA 11.7 and above.  "
                "Note: make sure nvcc has a supported version by running nvcc -V."
            )
        # Build -gencode (regular + PTX + family-specific 'f' when available)
        add_cuda_gencodes(cc_flag, set(cuda_archs()), bare_metal_version)
    else:
        # No nvcc present; warnings already emitted above
        pass

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    # "--ptxas-options=-v",
    # "--ptxas-options=-O2",
    # "-lineinfo",
    # "-DFLASHATTENTION_DISABLE_BACKWARD",
    # "-DFLASHATTENTION_DISABLE_DROPOUT",
    # "-DFLASHATTENTION_DISABLE_ALIBI",
    # "-DFLASHATTENTION_DISABLE_SOFTCAP",
    # "-DFLASHATTENTION_DISABLE_UNEVEN_K",
    # "-DFLASHATTENTION_DISABLE_LOCAL",
    ]

    compiler_c17_flag=["-O3", "-std=c++17"]
    # Add Windows-specific flags
    if sys.platform == "win32" and os.getenv('DISTUTILS_USE_SDK') == '1':
        nvcc_flags.extend(["-Xcompiler", "/Zc:__cplusplus"])
        compiler_c17_flag=["-O2", "/std:c++17", "/Zc:__cplusplus"]

    ext_modules.append(
        CUDAExtension(
            name="flash_attn_2_cuda",
            sources=[
                "csrc/flash_attn/flash_api.cpp",
                "csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_causal_sm80.cu",
            ],
            extra_compile_args={
                "cxx": compiler_c17_flag,
                "nvcc": append_nvcc_threads(nvcc_flags + cc_flag),
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "flash_attn",
                Path(this_dir) / "csrc" / "flash_attn" / "src",
                Path(this_dir) / "csrc" / "cutlass" / "include",
            ],
        )
    )
elif not SKIP_CUDA_BUILD and IS_ROCM:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    # Skips CK C++ extension compilation if using Triton Backend
    if not SKIP_CK_BUILD:
        ck_dir = "csrc/composable_kernel"

        #use codegen get code dispatch
        if not os.path.exists("./build"):
            os.makedirs("build")
        else:
            # Codegen filenames include tile sizes. When tuning tile shapes, stale
            # generated FMHA files can remain and be picked up by glob() below.
            for stale_path in glob.glob("build/fmha_*"):
                if os.path.isfile(stale_path):
                    os.remove(stale_path)

        # Detect architecture family for CK code generation
        archs_for_detection = os.getenv("GPU_ARCHS", "native").split(";")
        is_rdna3 = any(a.startswith("gfx11") for a in archs_for_detection if a != "native")
        is_rdna4 = any(a.startswith("gfx12") for a in archs_for_detection if a != "native")
        if archs_for_detection == ["native"]:
            device_arch = torch.cuda.get_device_properties("cuda").gcnArchName.split(":")[0]
            is_rdna3 = device_arch.startswith("gfx11")
            is_rdna4 = device_arch.startswith("gfx12")

        # Set CK code generation targets based on architecture
        # gfx11 = RDNA3 (gfx1100/1101/1102), gfx12 = RDNA4 (gfx1200/1201)
        if is_rdna3:
            ck_targets = "gfx11"
        elif is_rdna4:
            ck_targets = "gfx12"
        else:
            ck_targets = "gfx9,gfx950"

        print(
            f"[CK_MINIMAL_DEBUG] enabled={CK_MINIMAL_DEBUG}, "
            f"family={CK_MINIMAL_FAMILY}, "
            f"dtype={CK_MINIMAL_DTYPE}, optdim={CK_MINIMAL_OPTDIM}, "
            f"noncausal_only={CK_MINIMAL_NONCAUSAL_ONLY}, batch_only={CK_MINIMAL_BATCH_ONLY}"
        )

        if CK_MINIMAL_DEBUG and CK_MINIMAL_FAMILY not in {
            "standard",
            "kvcache",
            "varlen",
        }:
            raise ValueError(
                "FLASH_ATTN_CK_MINIMAL_FAMILY must be one of: "
                "standard, kvcache, varlen"
            )

        optdim = CK_MINIMAL_OPTDIM if CK_MINIMAL_DEBUG else os.getenv("OPT_DIM", "32,64,128,256")
        if CK_MINIMAL_DEBUG:
            if CK_MINIMAL_FAMILY == "kvcache":
                gen_dirs = ["fwd_appendkv", "fwd_splitkv"]
            elif CK_MINIMAL_FAMILY == "varlen":
                gen_dirs = ["fwd", "fwd_splitkv"]
            else:
                gen_dirs = ["fwd", "bwd"]
        else:
            gen_dirs = ["fwd", "fwd_appendkv", "fwd_splitkv", "bwd"]
        optdim_items = [item.strip() for item in CK_MINIMAL_OPTDIM.split(",") if item.strip()]
        dtype_items = [item.strip() for item in CK_MINIMAL_DTYPE.split(",") if item.strip()]
        if CK_MINIMAL_DEBUG and (not optdim_items or not dtype_items):
            raise ValueError(
                "FLASH_ATTN_CK_MINIMAL_OPTDIM and FLASH_ATTN_CK_MINIMAL_DTYPE "
                "must each contain at least one value"
            )
        if CK_MINIMAL_DEBUG and CK_MINIMAL_FAMILY == "kvcache":
            if len(optdim_items) != 1 or len(dtype_items) != 1:
                raise ValueError(
                    "kvcache minimal builds require exactly one optdim and one dtype; "
                    "build each coverage point separately"
                )
            if not CK_MINIMAL_BATCH_ONLY:
                raise ValueError("kvcache minimal builds require batch_only=TRUE")
        if CK_MINIMAL_DEBUG and CK_MINIMAL_FAMILY == "varlen":
            if len(optdim_items) != 1 or len(dtype_items) != 1:
                raise ValueError(
                    "varlen minimal builds require exactly one optdim and one dtype; "
                    "build each coverage point separately"
                )
            if CK_MINIMAL_BATCH_ONLY:
                raise ValueError("varlen minimal builds require batch_only=FALSE")
        # The codegen --filter path accepts one glob expression, not a comma-separated
        # dtype list. For multi-dtype minimal builds, generate by optdim first and let
        # filter_ck_minimal_generated_sources() do the exact dim/dtype pruning below.
        can_apply_codegen_filter = (
            CK_MINIMAL_DEBUG
            and len(optdim_items) == 1
            and len(dtype_items) == 1
            and (
                CK_MINIMAL_BATCH_ONLY
                or CK_MINIMAL_FAMILY == "varlen"
            )
        )
        minimal_codegen_pattern = ""
        minimal_codegen_bwd_filter = ""
        minimal_codegen_family_filters = {}
        if can_apply_codegen_filter:
            dim = optdim_items[0]
            dtype = dtype_items[0]
            # 保持100+规模：仅限定 d/dtype/batch，不再按 bias/mask/dropout 细切
            codegen_mode = "group" if CK_MINIMAL_FAMILY == "varlen" else "batch"
            minimal_codegen_pattern = f"*d{dim}_{dtype}_{codegen_mode}*"
            minimal_codegen_bwd_filter = (
                f"*fmha_bwd_dot_do_o_d{dim}_{dtype}*batch*"
                f"@*fmha_bwd_convert_dq_d{dim}_{dtype}*batch*"
                f"@*fmha_bwd_d{dim}_{dtype}_batch*"
            )
            if CK_MINIMAL_FAMILY == "kvcache":
                minimal_codegen_family_filters = {
                    "fwd_appendkv": f"*d{dim}_{dtype}*",
                    "fwd_splitkv": (
                        f"*d{dim}_{dtype}_batch*@*d{dim}_{dtype}_batch*"
                    ),
                }
            elif CK_MINIMAL_FAMILY == "varlen":
                minimal_codegen_family_filters = {
                    "fwd_splitkv": (
                        f"*d{dim}_{dtype}_group*@*d{dim}_{dtype}_group*"
                    ),
                }
            print(f"[CK_MINIMAL_DEBUG] codegen filter pattern={minimal_codegen_pattern}")
            print(f"[CK_MINIMAL_DEBUG] codegen bwd filter={minimal_codegen_bwd_filter}")
            if minimal_codegen_family_filters:
                print(
                    "[CK_MINIMAL_DEBUG] codegen family filters="
                    f"{minimal_codegen_family_filters}"
                )
        for direction in gen_dirs:
            command = [
                sys.executable,
                f"{ck_dir}/example/ck_tile/01_fmha/generate.py",
                "-d",
                direction,
                "--targets",
                ck_targets,
                "--output_dir",
                "build",
                "--receipt",
                "2",
                "--optdim",
                optdim,
            ]

            if can_apply_codegen_filter:
                if direction == "bwd":
                    command.extend(["--filter", minimal_codegen_bwd_filter])
                elif direction in minimal_codegen_family_filters:
                    command.extend(
                        ["--filter", minimal_codegen_family_filters[direction]]
                    )
                else:
                    command.extend(["--filter", minimal_codegen_pattern])

            subprocess.run(command, check=True)

        # Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
        # See https://github.com/pytorch/pytorch/pull/70650
        generator_flag = []
        torch_dir = torch.__path__[0]
        if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
            generator_flag = ["-DOLD_GENERATOR_PATH"]

        check_if_rocm_home_none("flash_attn")
        archs = os.getenv("GPU_ARCHS", "native").split(";")
        validate_and_update_archs(archs)

        if archs != ['native']:
            cc_flag = [f"--offload-arch={arch}" for arch in archs]
        else:
            arch = torch.cuda.get_device_properties("cuda").gcnArchName.split(":")[0]
            cc_flag = [f"--offload-arch={arch}"]

        # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
        # torch._C._GLIBCXX_USE_CXX11_ABI
        # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
        if FORCE_CXX11_ABI:
            torch._C._GLIBCXX_USE_CXX11_ABI = True

        if CK_MINIMAL_DEBUG:
            if CK_MINIMAL_FAMILY == "kvcache":
                base_cpp_sources = [
                    "csrc/flash_attn_ck/flash_api.cpp",
                    "csrc/flash_attn_ck/flash_common.cpp",
                    "csrc/flash_attn_ck/mha_fwd_kvcache.cpp",
                ]
            elif CK_MINIMAL_FAMILY == "varlen":
                base_cpp_sources = [
                    "csrc/flash_attn_ck/flash_api.cpp",
                    "csrc/flash_attn_ck/flash_common.cpp",
                    "csrc/flash_attn_ck/mha_varlen_fwd.cpp",
                ]
            else:
                base_cpp_sources = [
                    "csrc/flash_attn_ck/flash_api.cpp",
                    "csrc/flash_attn_ck/flash_common.cpp",
                    "csrc/flash_attn_ck/mha_bwd.cpp",
                    "csrc/flash_attn_ck/mha_fwd.cpp",
                ]
        else:
            base_cpp_sources = [
                "csrc/flash_attn_ck/flash_api.cpp",
                "csrc/flash_attn_ck/flash_common.cpp",
                "csrc/flash_attn_ck/mha_bwd.cpp",
                "csrc/flash_attn_ck/mha_fwd_kvcache.cpp",
                "csrc/flash_attn_ck/mha_fwd.cpp",
                "csrc/flash_attn_ck/mha_varlen_bwd.cpp",
                "csrc/flash_attn_ck/mha_varlen_fwd.cpp",
            ]
        generated_cpp_sources = glob.glob("build/fmha_*wd*.cpp")
        if CK_MINIMAL_DEBUG:
            before = len(generated_cpp_sources)
            generated_cpp_sources = filter_ck_minimal_generated_sources(generated_cpp_sources)
            after = len(generated_cpp_sources)
            print(
                f"[CK_MINIMAL_DEBUG] keeping generated kernels: {after}/{before} "
                f"(dtype={CK_MINIMAL_DTYPE}, optdim={CK_MINIMAL_OPTDIM}, "
                f"noncausal_only={CK_MINIMAL_NONCAUSAL_ONLY}, batch_only={CK_MINIMAL_BATCH_ONLY})"
            )
            kernel_count = sum(
                not os.path.basename(path).endswith("_api.cpp")
                for path in generated_cpp_sources
            )
            if kernel_count == 0:
                raise RuntimeError("[CK_MINIMAL_DEBUG] 过滤后无可用 CK kernel，请检查最小化条件")

        sources = base_cpp_sources + generated_cpp_sources

        # Check if torch is using hipify v2. Until CK is updated with HIPIFY_V2 macro,
        # we must replace the incorrect APIs.
        maybe_hipify_v2_flag = []
        if detect_hipify_v2():
            maybe_hipify_v2_flag = ["-DHIPIFY_V2"]

        rename_cpp_to_cu(sources)

        if CK_MINIMAL_DEBUG:
            if CK_MINIMAL_FAMILY == "kvcache":
                base_cu_sources = [
                    "csrc/flash_attn_ck/flash_api.cu",
                    "csrc/flash_attn_ck/flash_common.cu",
                    "csrc/flash_attn_ck/mha_fwd_kvcache.cu",
                ]
            elif CK_MINIMAL_FAMILY == "varlen":
                base_cu_sources = [
                    "csrc/flash_attn_ck/flash_api.cu",
                    "csrc/flash_attn_ck/flash_common.cu",
                    "csrc/flash_attn_ck/mha_varlen_fwd.cu",
                ]
            else:
                base_cu_sources = [
                    "csrc/flash_attn_ck/flash_api.cu",
                    "csrc/flash_attn_ck/flash_common.cu",
                    "csrc/flash_attn_ck/mha_bwd.cu",
                    "csrc/flash_attn_ck/mha_fwd.cu",
                ]
        else:
            base_cu_sources = [
                "csrc/flash_attn_ck/flash_api.cu",
                "csrc/flash_attn_ck/flash_common.cu",
                "csrc/flash_attn_ck/mha_bwd.cu",
                "csrc/flash_attn_ck/mha_fwd_kvcache.cu",
                "csrc/flash_attn_ck/mha_fwd.cu",
                "csrc/flash_attn_ck/mha_varlen_bwd.cu",
                "csrc/flash_attn_ck/mha_varlen_fwd.cu",
            ]
        generated_cu_sources = [source[:-4] + ".cu" for source in generated_cpp_sources]
        renamed_sources = base_cu_sources + generated_cu_sources

        if CK_MINIMAL_DEBUG:
            minimal_api_flag = {
                "kvcache": "-DFLASH_ATTN_CK_MINIMAL_KVCACHE_API=1",
                "varlen": "-DFLASH_ATTN_CK_MINIMAL_VARLEN_API=1",
                "standard": "-DFLASH_ATTN_CK_MINIMAL_API=1",
            }[CK_MINIMAL_FAMILY]
            cc_flag += [minimal_api_flag]

        cc_flag += ["-O3","-std=c++20",
                    "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
                    "-fgpu-flush-denormals-to-zero",
                    "-DCK_ENABLE_BF16",
                    "-DCK_ENABLE_BF8",
                    "-DCK_ENABLE_FP16",
                    "-DCK_ENABLE_FP32",
                    "-DCK_ENABLE_FP64",
                    "-DCK_ENABLE_FP8",
                    "-DCK_ENABLE_INT8",
                    "-DUSE_PROF_API=1",
                    # "-DFLASHATTENTION_DISABLE_BACKWARD",
                    "-D__HIP_PLATFORM_HCC__=1"]

        # RDNA3/RDNA4 use WMMA, CDNA uses XDL - only add XDL flag for CDNA
        if not (is_rdna3 or is_rdna4):
            cc_flag += ["-DCK_USE_XDL"]
        else:
            # Suppress occupancy warning for RDNA3/4 - compiler bug reports wrong desired occupancy
            cc_flag += ["-Wno-pass-failed"]

        cc_flag += [f"-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT={os.environ.get('CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT', 3)}"]

        ck_tile_dump_a_tmp = os.environ.get("CK_TILE_DEBUG_DUMP_A_TMP")
        if ck_tile_dump_a_tmp is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_DUMP_A_TMP={ck_tile_dump_a_tmp}"]

        ck_tile_no_raw_store = os.environ.get("CK_TILE_DEBUG_NO_RAW_STORE")
        if ck_tile_no_raw_store is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_NO_RAW_STORE={ck_tile_no_raw_store}"]

        ck_tile_force_store_path = os.environ.get("CK_TILE_DEBUG_FORCE_STORE_PATH")
        if ck_tile_force_store_path is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_FORCE_STORE_PATH={ck_tile_force_store_path}"]

        if CK_BWD_WMMA_PT_LDS_REMAP is not None:
            cc_flag += [f"-DCK_TILE_FMHA_BWD_WMMA_PT_LDS_REMAP={CK_BWD_WMMA_PT_LDS_REMAP}"]

        if CK_FWD_WMMA_NATIVE_O_EPILOGUE is not None:
            cc_flag += [
                "-DCK_TILE_FMHA_FWD_WMMA_NATIVE_O_EPILOGUE="
                f"{CK_FWD_WMMA_NATIVE_O_EPILOGUE}"
            ]

        if CK_BWD_WMMA_SGRADT_LDS_REMAP is not None:
            cc_flag += [
                f"-DCK_TILE_FMHA_BWD_WMMA_SGRADT_LDS_REMAP={CK_BWD_WMMA_SGRADT_LDS_REMAP}"
            ]

        if CK_BWD_WMMA_PT_REGISTER_REMAP is not None:
            cc_flag += [
                f"-DCK_TILE_FMHA_BWD_WMMA_PT_REGISTER_REMAP={CK_BWD_WMMA_PT_REGISTER_REMAP}"
            ]

        if CK_BWD_WMMA_SGRADT_REGISTER_REMAP is not None:
            cc_flag += [
                "-DCK_TILE_FMHA_BWD_WMMA_SGRADT_REGISTER_REMAP="
                f"{CK_BWD_WMMA_SGRADT_REGISTER_REMAP}"
            ]

        if CK_BWD_WMMA_PT_REGISTER_REMAP_D64_ONLY is not None:
            cc_flag += [
                "-DCK_TILE_FMHA_BWD_WMMA_PT_REGISTER_REMAP_D64_ONLY="
                f"{CK_BWD_WMMA_PT_REGISTER_REMAP_D64_ONLY}"
            ]

        if CK_BWD_WMMA_SGRADT_REGISTER_REMAP_D64_ONLY is not None:
            cc_flag += [
                "-DCK_TILE_FMHA_BWD_WMMA_SGRADT_REGISTER_REMAP_D64_ONLY="
                f"{CK_BWD_WMMA_SGRADT_REGISTER_REMAP_D64_ONLY}"
            ]

        if CK_BWD_LAYOUT_DIAG is not None:
            cc_flag += [f"-DCK_TILE_FMHA_BWD_LAYOUT_DIAG={CK_BWD_LAYOUT_DIAG}"]

        if CK_BWD_SPLIT_DISPATCH is not None:
            cc_flag += [f"-DCK_TILE_FMHA_BWD_SPLIT_DISPATCH={CK_BWD_SPLIT_DISPATCH}"]

        if CK_DEBUG_BWD_FORCE_GEMM34_BLOCK_SYNC is not None:
            cc_flag += [
                f"-DCK_TILE_DEBUG_BWD_FORCE_GEMM34_BLOCK_SYNC={CK_DEBUG_BWD_FORCE_GEMM34_BLOCK_SYNC}"
            ]

        if CK_DEBUG_BWD_FORCE_LDS_WAITCNT is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_BWD_FORCE_LDS_WAITCNT={CK_DEBUG_BWD_FORCE_LDS_WAITCNT}"]

        if CK_DEBUG_BWD_GEMM3_SCHED_BARRIER is not None:
            cc_flag += ["-DCK_TILE_DEBUG_BWD_GEMM3_SCHED_BARRIER"]

        if CK_DEBUG_BREG_VMCNT_BEFORE_WMMA is not None:
            cc_flag += ["-DCK_TILE_DEBUG_BREG_VMCNT_BEFORE_WMMA"]

        if CK_DEBUG_DRAIN_VMCNT_BEFORE_2D_EPILOGUE_STORE is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_DRAIN_VMCNT_BEFORE_2D_EPILOGUE_STORE="
                f"{CK_DEBUG_DRAIN_VMCNT_BEFORE_2D_EPILOGUE_STORE}"
            ]

        if CK_DEBUG_BWD_DRAIN_VMCNT_BEFORE_SGRADT_LDS_WRITE is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_DRAIN_VMCNT_BEFORE_SGRADT_LDS_WRITE="
                f"{CK_DEBUG_BWD_DRAIN_VMCNT_BEFORE_SGRADT_LDS_WRITE}"
            ]

        if CK_DEBUG_BWD_FORCE_FUSED_DQDKDV is not None:
            cc_flag += [
                f"-DCK_TILE_DEBUG_BWD_FORCE_FUSED_DQDKDV={CK_DEBUG_BWD_FORCE_FUSED_DQDKDV}"
            ]

        if CK_DEBUG_BWD_DK_ONLY_USE_DKDV_PIPELINE is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_DK_ONLY_USE_DKDV_PIPELINE="
                f"{CK_DEBUG_BWD_DK_ONLY_USE_DKDV_PIPELINE}"
            ]

        if CK_DEBUG_BWD_DK_USE_DQDKDV_PIPELINE is not None:
            cc_flag += [
                f"-DCK_TILE_DEBUG_BWD_DK_USE_DQDKDV_PIPELINE={CK_DEBUG_BWD_DK_USE_DQDKDV_PIPELINE}"
            ]

        if CK_DEBUG_BWD_SPLIT_USE_FUSED_DKDV is not None:
            cc_flag += [
                f"-DCK_TILE_DEBUG_BWD_SPLIT_USE_FUSED_DKDV={CK_DEBUG_BWD_SPLIT_USE_FUSED_DKDV}"
            ]

        if CK_DEBUG_BWD_SPLIT_PARALLEL_STREAMS is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_PARALLEL_STREAMS="
                f"{CK_DEBUG_BWD_SPLIT_PARALLEL_STREAMS}"
            ]

        if CK_DEBUG_BWD_SPLIT_CONVERT_ON_DQ_STREAM is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_CONVERT_ON_DQ_STREAM="
                f"{CK_DEBUG_BWD_SPLIT_CONVERT_ON_DQ_STREAM}"
            ]

        if CK_DEBUG_BWD_SPLIT_DISABLE_D256_PARALLEL_STREAMS is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DISABLE_D256_PARALLEL_STREAMS="
                f"{CK_DEBUG_BWD_SPLIT_DISABLE_D256_PARALLEL_STREAMS}"
            ]

        if CK_ENABLE_D256_HYBRID_FAST is not None:
            cc_flag += [
                f"-DCK_TILE_ENABLE_D256_HYBRID_FAST={CK_ENABLE_D256_HYBRID_FAST}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_FORCE_DETERMINISTIC is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_FORCE_DETERMINISTIC="
                f"{CK_DEBUG_BWD_SPLIT_DQ_FORCE_DETERMINISTIC}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_QMAJOR is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_QMAJOR="
                f"{CK_DEBUG_BWD_SPLIT_DQ_QMAJOR}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_QMAJOR_ALLOW_UNSTABLE is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_QMAJOR_ALLOW_UNSTABLE="
                f"{CK_DEBUG_BWD_SPLIT_DQ_QMAJOR_ALLOW_UNSTABLE}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_QMAJOR_D256 is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_QMAJOR_D256="
                f"{CK_DEBUG_BWD_SPLIT_DQ_QMAJOR_D256}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_QMAJOR_MAX_SEQLEN is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_QMAJOR_MAX_SEQLEN="
                f"{CK_DEBUG_BWD_SPLIT_DQ_QMAJOR_MAX_SEQLEN}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_LATE_DO_D_PREFETCH is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_LATE_DO_D_PREFETCH="
                f"{CK_DEBUG_BWD_SPLIT_DQ_LATE_DO_D_PREFETCH}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_LATE_REG_LOAD is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_LATE_REG_LOAD="
                f"{CK_DEBUG_BWD_SPLIT_DQ_LATE_REG_LOAD}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_INPLACE_DS is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_INPLACE_DS="
                f"{CK_DEBUG_BWD_SPLIT_DQ_INPLACE_DS}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_D256_NO_INPLACE_DS is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_D256_NO_INPLACE_DS="
                f"{CK_DEBUG_BWD_SPLIT_DQ_D256_NO_INPLACE_DS}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_D256_ALLOW_LATE_LOAD is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_D256_ALLOW_LATE_LOAD="
                f"{CK_DEBUG_BWD_SPLIT_DQ_D256_ALLOW_LATE_LOAD}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_D256_ALLOW_LATE_LOAD_NONMASK is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_D256_ALLOW_LATE_LOAD_NONMASK="
                f"{CK_DEBUG_BWD_SPLIT_DQ_D256_ALLOW_LATE_LOAD_NONMASK}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_D256_LATE_Q_ONLY_NONMASK is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_D256_LATE_Q_ONLY_NONMASK="
                f"{CK_DEBUG_BWD_SPLIT_DQ_D256_LATE_Q_ONLY_NONMASK}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_D256_NO_DS_PREFETCH_NEXT is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_D256_NO_DS_PREFETCH_NEXT="
                f"{CK_DEBUG_BWD_SPLIT_DQ_D256_NO_DS_PREFETCH_NEXT}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_SCOPE_DS_GEMM_STORE is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_SCOPE_DS_GEMM_STORE="
                f"{CK_DEBUG_BWD_SPLIT_DQ_SCOPE_DS_GEMM_STORE}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_STATIC_DQ_WINDOW is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_STATIC_DQ_WINDOW="
                f"{CK_DEBUG_BWD_SPLIT_DQ_STATIC_DQ_WINDOW}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_STATIC_DQ_WINDOW_RAW is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_STATIC_DQ_WINDOW_RAW="
                f"{CK_DEBUG_BWD_SPLIT_DQ_STATIC_DQ_WINDOW_RAW}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_D256_SINGLE_K4_FAST is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_D256_SINGLE_K4_FAST="
                f"{CK_DEBUG_BWD_SPLIT_DQ_D256_SINGLE_K4_FAST}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_D256_TAIL_DIRECT_UPDATE is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_D256_TAIL_DIRECT_UPDATE="
                f"{CK_DEBUG_BWD_SPLIT_DQ_D256_TAIL_DIRECT_UPDATE}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_D256_SKIP_NONMASK_TAIL_REMAP is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_D256_SKIP_NONMASK_TAIL_REMAP="
                f"{CK_DEBUG_BWD_SPLIT_DQ_D256_SKIP_NONMASK_TAIL_REMAP}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_TAIL_DIRECT_UPDATE is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_TAIL_DIRECT_UPDATE="
                f"{CK_DEBUG_BWD_SPLIT_DQ_TAIL_DIRECT_UPDATE}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_D256_BN64 is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_D256_BN64="
                f"{CK_DEBUG_BWD_SPLIT_DQ_D256_BN64}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_D256_HEAD_SPLIT2 is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_D256_HEAD_SPLIT2="
                f"{CK_DEBUG_BWD_SPLIT_DQ_D256_HEAD_SPLIT2}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_D256_HEAD_SPLIT4 is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_D256_HEAD_SPLIT4="
                f"{CK_DEBUG_BWD_SPLIT_DQ_D256_HEAD_SPLIT4}"
            ]

        if CK_DEBUG_BWD_SPLIT_DQ_D256_GEMM4_ASMEM is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DQ_D256_GEMM4_ASMEM="
                f"{CK_DEBUG_BWD_SPLIT_DQ_D256_GEMM4_ASMEM}"
            ]

        if CK_DEBUG_BWD_D256_NONMASK_NO_EDGE_CHECK is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_D256_NONMASK_NO_EDGE_CHECK="
                f"{CK_DEBUG_BWD_D256_NONMASK_NO_EDGE_CHECK}"
            ]

        if CK_DEBUG_BWD_DQ_D256_NONMASK_NO_EDGE_CHECK is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_DQ_D256_NONMASK_NO_EDGE_CHECK="
                f"{CK_DEBUG_BWD_DQ_D256_NONMASK_NO_EDGE_CHECK}"
            ]

        if CK_DEBUG_BWD_DKDV_D256_NONMASK_NO_EDGE_CHECK is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_DKDV_D256_NONMASK_NO_EDGE_CHECK="
                f"{CK_DEBUG_BWD_DKDV_D256_NONMASK_NO_EDGE_CHECK}"
            ]

        if CK_DEBUG_BWD_DKDV_NONMASK_NO_EDGE_CHECK is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_DKDV_NONMASK_NO_EDGE_CHECK="
                f"{CK_DEBUG_BWD_DKDV_NONMASK_NO_EDGE_CHECK}"
            ]

        if CK_DEBUG_BWD_SPLIT_DV_D256_NONMASK_NO_EDGE_CHECK is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DV_D256_NONMASK_NO_EDGE_CHECK="
                f"{CK_DEBUG_BWD_SPLIT_DV_D256_NONMASK_NO_EDGE_CHECK}"
            ]

        if CK_DEBUG_BWD_SPLIT_LAUNCH_DQ_FIRST is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_LAUNCH_DQ_FIRST="
                f"{CK_DEBUG_BWD_SPLIT_LAUNCH_DQ_FIRST}"
            ]

        if CK_DEBUG_BWD_SPLIT_LAUNCH_DQ_FIRST_AUTO is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_LAUNCH_DQ_FIRST_AUTO="
                f"{CK_DEBUG_BWD_SPLIT_LAUNCH_DQ_FIRST_AUTO}"
            ]

        if CK_DEBUG_BWD_SPLIT_LAUNCH_DK_FIRST is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_LAUNCH_DK_FIRST="
                f"{CK_DEBUG_BWD_SPLIT_LAUNCH_DK_FIRST}"
            ]

        if CK_DEBUG_BWD_SPLIT_DKDV_PRECAST_SGRAD is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DKDV_PRECAST_SGRAD="
                f"{CK_DEBUG_BWD_SPLIT_DKDV_PRECAST_SGRAD}"
            ]

        if CK_DEBUG_BWD_SPLIT_DKDV_FRESH_DS is not None:
            cc_flag += [
                f"-DCK_TILE_DEBUG_BWD_SPLIT_DKDV_FRESH_DS={CK_DEBUG_BWD_SPLIT_DKDV_FRESH_DS}"
            ]

        if CK_DEBUG_BWD_SPLIT_DKDV_FRESH_P is not None:
            cc_flag += [
                f"-DCK_TILE_DEBUG_BWD_SPLIT_DKDV_FRESH_P={CK_DEBUG_BWD_SPLIT_DKDV_FRESH_P}"
            ]

        if CK_DEBUG_BWD_SPLIT_DKDV_D256_INLINE_DS_REMAP is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DKDV_D256_INLINE_DS_REMAP="
                f"{CK_DEBUG_BWD_SPLIT_DKDV_D256_INLINE_DS_REMAP}"
            ]

        if CK_DEBUG_BWD_SPLIT_DKDV_D256_DS_LDS_MATERIALIZE is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DKDV_D256_DS_LDS_MATERIALIZE="
                f"{CK_DEBUG_BWD_SPLIT_DKDV_D256_DS_LDS_MATERIALIZE}"
            ]

        if CK_DEBUG_BWD_SPLIT_DKDV_POST_GEMM3_DS_LDS_BOUNDARY is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DKDV_POST_GEMM3_DS_LDS_BOUNDARY="
                f"{CK_DEBUG_BWD_SPLIT_DKDV_POST_GEMM3_DS_LDS_BOUNDARY}"
            ]

        if CK_DEBUG_BWD_SPLIT_DKDV_USE_GENERIC_LDS_LAYOUT is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DKDV_USE_GENERIC_LDS_LAYOUT="
                f"{CK_DEBUG_BWD_SPLIT_DKDV_USE_GENERIC_LDS_LAYOUT}"
            ]

        if CK_DEBUG_BWD_SPLIT_DKDV_LATE_QDO_STORE_LAMBDA is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DKDV_LATE_QDO_STORE_LAMBDA="
                f"{CK_DEBUG_BWD_SPLIT_DKDV_LATE_QDO_STORE_LAMBDA}"
            ]

        if CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DV_GEMM1 is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DV_GEMM1="
                f"{CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DV_GEMM1}"
            ]

        if CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4 is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4="
                f"{CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4}"
            ]

        if CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_KT_ONLY is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_KT_ONLY="
                f"{CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_KT_ONLY}"
            ]

        if CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_DS_LDS_ONLY is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_DS_LDS_ONLY="
                f"{CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_DS_LDS_ONLY}"
            ]

        if CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_ONE_SLICE is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_ONE_SLICE="
                f"{CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_ONE_SLICE}"
            ]

        if CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_SINK_MODE is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_SINK_MODE="
                f"{CK_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_SINK_MODE}"
            ]

        if CK_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS="
                f"{CK_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS}"
            ]

        if CK_DEBUG_BWD_SPLIT_DV_EARLY_Q_LSE_STORE is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DV_EARLY_Q_LSE_STORE="
                f"{CK_DEBUG_BWD_SPLIT_DV_EARLY_Q_LSE_STORE}"
            ]

        if CK_DEBUG_BWD_SPLIT_DV_GEMM1_BSMEM is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DV_GEMM1_BSMEM="
                f"{CK_DEBUG_BWD_SPLIT_DV_GEMM1_BSMEM}"
            ]

        if CK_DEBUG_BWD_SPLIT_DV_GEMM1_BSMEM_D256 is not None:
            cc_flag += [
                "-DCK_TILE_DEBUG_BWD_SPLIT_DV_GEMM1_BSMEM_D256="
                f"{CK_DEBUG_BWD_SPLIT_DV_GEMM1_BSMEM_D256}"
            ]

        if CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT_ONLY is not None:
            cc_flag += [
                "-DCK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT_ONLY="
                f"{CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT_ONLY}"
            ]

        ck_tile_force_row_store = os.environ.get("CK_TILE_DEBUG_FORCE_ROW_STORE")
        if ck_tile_force_row_store is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_FORCE_ROW_STORE={ck_tile_force_row_store}"]

        ck_tile_owner_tag = os.environ.get("CK_TILE_DEBUG_OWNER_TAG")
        if ck_tile_owner_tag is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_OWNER_TAG={ck_tile_owner_tag}"]

        ck_tile_owner_scan = os.environ.get("CK_TILE_DEBUG_OWNER_SCAN")
        if ck_tile_owner_scan is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_OWNER_SCAN={ck_tile_owner_scan}"]

        ck_tile_owner_scan_limit = os.environ.get("CK_TILE_DEBUG_OWNER_SCAN_LIMIT")
        if ck_tile_owner_scan_limit is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_OWNER_SCAN_LIMIT={ck_tile_owner_scan_limit}"]

        ck_tile_owner_scan_abs_min = os.environ.get("CK_TILE_DEBUG_OWNER_SCAN_ABS_MIN")
        if ck_tile_owner_scan_abs_min is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_OWNER_SCAN_ABS_MIN={ck_tile_owner_scan_abs_min}"]

        ck_tile_o_glb_after = os.environ.get("CK_TILE_DEBUG_O_GLB_AFTER")
        if ck_tile_o_glb_after is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_O_GLB_AFTER={ck_tile_o_glb_after}"]

        ck_tile_o_glb_b = os.environ.get("CK_TILE_DEBUG_O_GLB_B")
        if ck_tile_o_glb_b is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_O_GLB_B={ck_tile_o_glb_b}"]

        ck_tile_o_glb_h = os.environ.get("CK_TILE_DEBUG_O_GLB_H")
        if ck_tile_o_glb_h is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_O_GLB_H={ck_tile_o_glb_h}"]

        ck_tile_o_glb_m = os.environ.get("CK_TILE_DEBUG_O_GLB_M")
        if ck_tile_o_glb_m is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_O_GLB_M={ck_tile_o_glb_m}"]

        ck_tile_o_glb_n = os.environ.get("CK_TILE_DEBUG_O_GLB_N")
        if ck_tile_o_glb_n is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_O_GLB_N={ck_tile_o_glb_n}"]

        ck_tile_o_glb_m2 = os.environ.get("CK_TILE_DEBUG_O_GLB_M2")
        if ck_tile_o_glb_m2 is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_O_GLB_M2={ck_tile_o_glb_m2}"]

        ck_tile_o_glb_n2 = os.environ.get("CK_TILE_DEBUG_O_GLB_N2")
        if ck_tile_o_glb_n2 is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_O_GLB_N2={ck_tile_o_glb_n2}"]

        # Imitate https://github.com/ROCm/composable_kernel/blob/c8b6b64240e840a7decf76dfaa13c37da5294c4a/CMakeLists.txt#L190-L214
        hip_version = get_hip_version()
        if hip_version > Version('5.5.00000'):
            cc_flag += ["-mllvm", "--lsr-drop-solution=1"]
        if hip_version > Version('5.7.23302'):
            cc_flag += ["-fno-offload-uniform-block"]
        if hip_version > Version('6.1.40090'):
            cc_flag += ["-mllvm", "-enable-post-misched=0"]
        if hip_version > Version('6.2.41132'):
            cc_flag += ["-mllvm", "-amdgpu-early-inline-all=true",
                        "-mllvm", "-amdgpu-function-calls=false"]
        if hip_version > Version('6.2.41133') and hip_version < Version('6.3.00000'):
            cc_flag += ["-mllvm", "-amdgpu-coerce-illegal-types=1"]

        
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++20"] + generator_flag + maybe_hipify_v2_flag,
            "nvcc": cc_flag + generator_flag + maybe_hipify_v2_flag,
        }

        include_dirs = [
            Path(this_dir) / "csrc" / "composable_kernel" / "include",
            Path(this_dir) / "csrc" / "composable_kernel" / "library" / "include",
            Path(this_dir) / "csrc" / "composable_kernel" / "example" / "ck_tile" / "01_fmha",
        ]

        ext_modules.append(
            CUDAExtension(
                name="flash_attn_2_cuda",
                sources=renamed_sources,
                extra_compile_args=extra_compile_args,
                include_dirs=include_dirs,
            )
        )


def get_package_version():
    with open(Path(this_dir) / "flash_attn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


def get_wheel_url():
    torch_version_raw = parse(torch.__version__)
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = get_platform()
    flash_version = get_package_version()
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

    if IS_ROCM:
        torch_hip_version = get_hip_version()
        hip_version = f"{torch_hip_version.major}{torch_hip_version.minor}"
        wheel_filename = f"{PACKAGE_NAME}-{flash_version}+rocm{hip_version}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"
    else:
        # Determine the version numbers that will be used to determine the correct wheel
        # We're using the CUDA version used to build torch, not the one currently installed
        # _, cuda_version_raw = get_cuda_bare_metal_version(CUDA_HOME)
        torch_cuda_version = parse(torch.version.cuda)
        # For CUDA 11, we only compile for CUDA 11.8, and for CUDA 12 we only compile for CUDA 12.3
        # to save CI time. Minor versions should be compatible.
        torch_cuda_version = parse("11.8") if torch_cuda_version.major == 11 else parse("12.3")
        # cuda_version = f"{cuda_version_raw.major}{cuda_version_raw.minor}"
        cuda_version = f"{torch_cuda_version.major}"

        # Determine wheel URL based on CUDA version, torch version, python version and OS
        wheel_filename = f"{PACKAGE_NAME}-{flash_version}+cu{cuda_version}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"

    wheel_url = BASE_WHEEL_URL.format(tag_name=f"v{flash_version}", wheel_name=wheel_filename)

    return wheel_url, wheel_filename


class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all flash attention installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """

    def run(self):
        if FORCE_BUILD:
            return super().run()

        wheel_url, wheel_filename = get_wheel_url()
        print("Guessing wheel URL: ", wheel_url)
        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)

            # Make the archive
            # Lifted from the root wheel processing command
            # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)
            os.rename(wheel_filename, wheel_path)
        except (urllib.error.HTTPError, urllib.error.URLError):
            print("Precompiled wheel not found. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().run()


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            nvcc_threads = max(1, int(NVCC_THREADS))

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
            # Assume worst-case peak observed memory usage of ~5GB per NVCC thread.
            # Limit: peak_threads = max_jobs * nvcc_threads and peak_threads * 5GB <= free_memory.
            max_num_jobs_memory = max(1, int(free_memory_gb / (5 * nvcc_threads)))

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            print(
                f"Auto set MAX_JOBS to `{max_jobs}`, NVCC_THREADS to `{nvcc_threads}`. "
                "If you see memory pressure, please use a lower `MAX_JOBS=N` or `NVCC_THREADS=N` value."
            )
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "flash_attn.egg-info",
            "flash_attn.cute",
            "flash_attn.cute.*",
        )
    ),
    author="Tri Dao",
    author_email="tri@tridao.me",
    description="Flash Attention: Fast and Memory-Efficient Exact Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dao-AILab/flash-attention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": NinjaBuildExtension}
    if ext_modules
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "einops",
    ],
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
)
