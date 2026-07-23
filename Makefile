SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.DELETE_ON_ERROR:

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest
GPU_ARCHS ?= gfx1100
MAX_JOBS ?= 8
DIST_DIR ?= dist
DEVICE ?= cuda:0
VERIFY_SDIST_BUILD ?= 1
FULL_BUILD_STAMP := build/release/full_gfx1100.json

MINIMAL_FAMILY ?= standard
MINIMAL_OPTDIM ?= 64
MINIMAL_DTYPE ?= fp16
MINIMAL_NONCAUSAL_ONLY ?= TRUE
MINIMAL_BATCH_ONLY ?= TRUE

SMOKE_SEQLENS ?= 128
RELEASE_DIMS ?= 32,64,128,256
RELEASE_SEQLENS ?= 768,1024,2048
RELEASE_REPEATS ?= 2
RELEASE_GRAD_SCALE ?= 100

FP16_FWD_ATOL ?= 2.0e-3
FP16_BWD_SCALED_ATOL ?= 3.0e-3
FP16_BWD_REL_ATOL ?= 1.0e-3
BF16_FWD_ATOL ?= 2.0e-2
BF16_BWD_SCALED_ATOL ?= 3.0e-2
BF16_BWD_REL_ATOL ?= 3.0e-2
BF16_BOUNDARY_BWD_SCALED_ATOL ?= 4.1e-2
BF16_BOUNDARY_BWD_REL_ATOL ?= 8.0e-3
BF16_BOUNDARY_MAX_ULP ?= 1

RELEASE_ENV = env \
	BUILD_TARGET='rocm' \
	GPU_ARCHS='$(GPU_ARCHS)' \
	MAX_JOBS='$(MAX_JOBS)' \
	FLASH_ATTN_CK_PROFILE='release' \
	FLASH_ATTENTION_FORCE_BUILD='TRUE' \
	FLASH_ATTENTION_SKIP_CUDA_BUILD='FALSE' \
	FLASH_ATTENTION_ALLOW_REMOTE_WHEEL='FALSE'

VENDOR_SENTINELS := \
	csrc/composable_kernel/example/ck_tile/01_fmha/generate.py \
	csrc/composable_kernel/include/ck/ck.hpp \
	csrc/composable_kernel/LICENSE \
	csrc/cutlass/include/cutlass/cutlass.h \
	csrc/cutlass/LICENSE.txt \
	csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp \
	csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_pipeline_kr_ktr_vr_iglp.hpp \
	csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_qmajor_pipeline_kr_ktr_vr_iglp.hpp \
	csrc/composable_kernel/include/ck_tile/ops/gemm/warp/warp_wmma_gemm_gfx11_utils.hpp \
	csrc/flash_attn_ck/generated_sources_gfx1100.txt \
	BUILDING_ROCM.md \
	Makefile \
	requirements-build.txt \
	requirements-runtime.txt \
	scripts/audit_bf16_boundary_fp32.py \
	scripts/release_extension_stamp.py \
	scripts/test_prebuilt_flash_attn_ck.py \
	tests/release_ck_nodes.txt \
	VENDORED_DEPENDENCIES.json \
	VENDORED_FILES.txt \
	THIRD_PARTY_NOTICES.md

.PHONY: help doctor vendor-check freeze-check build-minimal build-full \
	assert-local-full-extension wheel sdist verify-sdist test-smoke test-bf16-fp32-audit \
	release-test-config-check test-release-pytest test-release clean

help:
	@printf '%s\n' \
		'Frozen FA4 CK source build (offline; no submodules or source downloads)' \
		'' \
		'Targets:' \
		'  doctor        Check the local ROCm/Python build toolchain.' \
		'  vendor-check  Check required vendored CK and CUTLASS source files.' \
		'  freeze-check  Check provenance plus the no-submodule Git layout.' \
		'  build-minimal Build an in-place standard D64/FP16 debug closure.' \
		'  build-full    Build the complete in-place gfx1100 release extension.' \
		'  wheel         Build a full local-source wheel under DIST_DIR.' \
		'  sdist         Build a source archive under DIST_DIR without compiling.' \
		'  verify-sdist  Extract the sdist and build its minimal closure offline.' \
		'  test-smoke    Build the minimal closure and run one GPU correctness case.' \
		'  test-bf16-fp32-audit  Audit known BF16 boundaries with true FP32 SDPA.' \
		'  test-release-pytest   Run 40 frozen standard/deterministic/varlen nodes.' \
		'  test-release  Run strict FP16 and ULP-aware BF16 release matrices.' \
		'  clean         Remove build products created by these targets.' \
		'' \
		'Common overrides:' \
		'  PYTHON=python GPU_ARCHS=gfx1100 MAX_JOBS=8 DEVICE=cuda:0' \
		'  DIST_DIR=dist VERIFY_SDIST_BUILD=1' \
		'' \
		'Minimal closure overrides:' \
		'  MINIMAL_FAMILY=standard MINIMAL_OPTDIM=64 MINIMAL_DTYPE=fp16' \
		'  MINIMAL_NONCAUSAL_ONLY=TRUE MINIMAL_BATCH_ONLY=TRUE'

doctor:
	@printf 'Python: '
	@$(PYTHON) --version
	@$(PYTHON) -c 'from pathlib import Path; import packaging, psutil, torch, wheel; from torch.utils.cpp_extension import ROCM_HOME; assert torch.version.hip, "PyTorch is not a ROCm build"; assert ROCM_HOME, "PyTorch could not locate ROCm"; hipcc=Path(ROCM_HOME) / "bin" / "hipcc"; assert hipcc.is_file(), "missing ROCm compiler: " + str(hipcc); print("torch=" + torch.__version__ + ", rocm=" + str(torch.version.hip) + ", rocm_home=" + str(ROCM_HOME))'
	@for tool in cmake ninja tar; do \
		command -v "$$tool" >/dev/null || { echo "ERROR: missing required tool: $$tool" >&2; exit 1; }; \
	done
	@$(PYTHON) -c 'import torch; available=torch.cuda.is_available(); print("gpu_visible=" + str(available)); print("gpu=" + torch.cuda.get_device_name(0) if available else "gpu=not-required-for-cross-build")'
	@printf 'Release settings: GPU_ARCHS=%s MAX_JOBS=%s\n' '$(GPU_ARCHS)' '$(MAX_JOBS)'

vendor-check:
	@for path in $(VENDOR_SENTINELS); do \
		test -f "$$path" || { echo "ERROR: frozen source is incomplete; missing $$path" >&2; exit 1; }; \
	done
	@$(PYTHON) -c 'from pathlib import Path; inventory=Path("VENDORED_FILES.txt").read_text(encoding="utf-8").splitlines(); assert inventory and inventory == sorted(set(inventory)), "invalid vendored file inventory"; assert all(path.startswith(("csrc/composable_kernel/", "csrc/cutlass/")) for path in inventory), "invalid vendored path"; missing=[path for path in inventory if not Path(path).is_file()]; assert not missing, f"vendored inventory files missing: {len(missing)}"; print(f"Vendored file inventory: {len(inventory)} files")'
	@echo 'Vendored CK/CUTLASS source closure: OK'

freeze-check: vendor-check
	@$(PYTHON) -c 'import json; data=json.load(open("VENDORED_DEPENDENCIES.json", encoding="utf-8")); required={"schema_version", "snapshot", "target", "flash_attention", "composable_kernel", "cutlass", "validated_environment", "validated_artifact", "vendored_file_inventory", "validation_report"}; missing=sorted(required-data.keys()); assert not missing, "missing provenance keys: " + ", ".join(missing); print("Vendored provenance manifest: OK")'
	@if test -e .gitmodules; then \
		echo 'ERROR: root .gitmodules is forbidden in the frozen repository' >&2; exit 1; \
	fi
	@nested="$$(find csrc/composable_kernel csrc/cutlass -mindepth 1 \( -name .git -o -name .gitmodules \) -print -quit)"; \
	if test -n "$$nested"; then \
		echo "ERROR: nested Git metadata is forbidden: $$nested" >&2; exit 1; \
	fi
	@if test -d .git; then \
		gitlinks="$$(git ls-files --stage | awk '$$1 == "160000" { print $$4 }')"; \
		if test -n "$$gitlinks"; then \
			echo "ERROR: Gitlink entries remain: $$gitlinks" >&2; exit 1; \
		fi; \
		oversized="$$(git ls-files -z | xargs -0 -r stat -c '%s %n' | awk '$$1 >= 100000000 { print; exit }')"; \
		if test -n "$$oversized"; then \
			echo "ERROR: tracked file reaches the GitHub single-file limit: $$oversized" >&2; exit 1; \
		fi; \
		for path in $(VENDOR_SENTINELS); do \
			git ls-files --error-unmatch "$$path" >/dev/null 2>&1 || { \
				echo "ERROR: required frozen source is not tracked: $$path" >&2; exit 1; \
			}; \
		done; \
	fi
	@$(PYTHON) -c 'from pathlib import Path; import subprocess; inventory=Path("VENDORED_FILES.txt").read_text(encoding="utf-8").splitlines(); tracked=subprocess.check_output(["git", "ls-files", "csrc/composable_kernel", "csrc/cutlass"], text=True).splitlines() if Path(".git").exists() else inventory; assert tracked == inventory, "VENDORED_FILES.txt differs from the tracked CK/CUTLASS file set"'
	@echo 'Frozen repository layout: OK'

build-minimal: doctor freeze-check
	@rm -f $(FULL_BUILD_STAMP)
	@find . -maxdepth 1 -type f -name 'flash_attn_2_cuda*.so' -delete
	$(RELEASE_ENV) \
		FLASH_ATTN_CK_MINIMAL_DEBUG='TRUE' \
		FLASH_ATTN_CK_MINIMAL_FAMILY='$(MINIMAL_FAMILY)' \
		FLASH_ATTN_CK_MINIMAL_OPTDIM='$(MINIMAL_OPTDIM)' \
		FLASH_ATTN_CK_MINIMAL_DTYPE='$(MINIMAL_DTYPE)' \
		FLASH_ATTN_CK_MINIMAL_NONCAUSAL_ONLY='$(MINIMAL_NONCAUSAL_ONLY)' \
		FLASH_ATTN_CK_MINIMAL_BATCH_ONLY='$(MINIMAL_BATCH_ONLY)' \
		$(PYTHON) setup.py build_ext --inplace --force
	@extensions=( ./flash_attn_2_cuda*.so ); \
	test "$${#extensions[@]}" -eq 1 && test -f "$${extensions[0]}" || { \
		echo 'ERROR: minimal build did not produce exactly one local extension' >&2; exit 1; \
	}

build-full: doctor freeze-check
	@rm -f $(FULL_BUILD_STAMP)
	@find . -maxdepth 1 -type f -name 'flash_attn_2_cuda*.so' -delete
	$(RELEASE_ENV) \
		FLASH_ATTN_CK_MINIMAL_DEBUG='FALSE' \
		$(PYTHON) setup.py build_ext --inplace --force
	@extensions=( ./flash_attn_2_cuda*.so ); \
	test "$${#extensions[@]}" -eq 1 && test -f "$${extensions[0]}" || { \
		echo 'ERROR: full build did not produce exactly one local extension' >&2; exit 1; \
	}; \
	mkdir -p build/release; \
	$(PYTHON) scripts/release_extension_stamp.py write \
		--extension "$${extensions[0]}" \
		--stamp '$(FULL_BUILD_STAMP)' \
		--source-manifest csrc/flash_attn_ck/generated_sources_gfx1100.txt \
		--gpu-arch '$(GPU_ARCHS)'

assert-local-full-extension:
	@extensions=( ./flash_attn_2_cuda*.so ); \
	test "$${#extensions[@]}" -eq 1 && test -f "$${extensions[0]}" || { \
		echo 'ERROR: expected exactly one local full extension' >&2; exit 1; \
	}; \
	$(PYTHON) scripts/release_extension_stamp.py check \
		--extension "$${extensions[0]}" \
		--stamp '$(FULL_BUILD_STAMP)' \
		--source-manifest csrc/flash_attn_ck/generated_sources_gfx1100.txt \
		--gpu-arch '$(GPU_ARCHS)'

wheel: doctor freeze-check
	@mkdir -p '$(DIST_DIR)'
	@find '$(DIST_DIR)' -maxdepth 1 -type f -name '*.whl' -delete
	$(RELEASE_ENV) \
		FLASH_ATTN_CK_MINIMAL_DEBUG='FALSE' \
		$(PYTHON) setup.py build_ext --force bdist_wheel --dist-dir '$(DIST_DIR)'
	@wheels=( '$(DIST_DIR)'/*.whl ); \
	test "$${#wheels[@]}" -eq 1 && test -f "$${wheels[0]}" || { \
		echo "ERROR: expected exactly one wheel under $(DIST_DIR)" >&2; exit 1; \
	}; \
	$(PYTHON) -c 'from pathlib import PurePosixPath; import sys, zipfile; wheel=sys.argv[1]; names=zipfile.ZipFile(wheel).namelist(); extensions=[name for name in names if PurePosixPath(name).name.startswith("flash_attn_2_cuda") and name.endswith(".so")]; assert len(extensions) == 1, f"wheel must contain exactly one native extension, found {len(extensions)}"; print("Wheel native extension:", extensions[0])' "$${wheels[0]}"

sdist: freeze-check
	@mkdir -p '$(DIST_DIR)'
	@find '$(DIST_DIR)' -maxdepth 1 -type f -name '*.tar.gz' -delete
	$(RELEASE_ENV) \
		FLASH_ATTN_CK_MINIMAL_DEBUG='FALSE' \
		FLASH_ATTENTION_METADATA_ONLY='TRUE' \
		FLASH_ATTENTION_SKIP_CUDA_BUILD='TRUE' \
		$(PYTHON) setup.py sdist --dist-dir '$(DIST_DIR)'

verify-sdist: sdist
	@archives=( '$(DIST_DIR)'/*.tar.gz ); \
	test "$${#archives[@]}" -eq 1 && test -f "$${archives[0]}" || { \
		echo "ERROR: expected exactly one source archive under $(DIST_DIR)" >&2; exit 1; \
	}; \
	tmp="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	tar -xzf "$${archives[0]}" -C "$$tmp"; \
	roots=( "$$tmp"/* ); \
	test "$${#roots[@]}" -eq 1 && test -d "$${roots[0]}" || { \
		echo 'ERROR: malformed source archive root' >&2; exit 1; \
	}; \
	src="$${roots[0]}"; \
	for path in $(VENDOR_SENTINELS); do \
		test -f "$$src/$$path" || { echo "ERROR: sdist is missing $$path" >&2; exit 1; }; \
	done; \
	test ! -e "$$src/.gitmodules" || { echo 'ERROR: sdist contains .gitmodules' >&2; exit 1; }; \
	nested="$$(find "$$src/csrc/composable_kernel" "$$src/csrc/cutlass" -mindepth 1 \( -name .git -o -name .gitmodules \) -print -quit)"; \
	test -z "$$nested" || { echo "ERROR: sdist contains nested Git metadata: $$nested" >&2; exit 1; }; \
	python_bin="$$($(PYTHON) -c 'import sys; print(sys.executable)')"; \
	"$$python_bin" -c 'import json, pathlib, sys; root=pathlib.Path(sys.argv[1]); json.load((root / "VENDORED_DEPENDENCIES.json").open(encoding="utf-8")); expected=(root / "VENDORED_FILES.txt").read_text(encoding="utf-8").splitlines(); actual=sorted(str(path.relative_to(root)) for base in ("csrc/composable_kernel", "csrc/cutlass") for path in (root / base).rglob("*") if path.is_file()); missing=set(expected)-set(actual); unexpected=set(actual)-set(expected); assert not missing and not unexpected, f"sdist vendor inventory mismatch: missing={len(missing)}, unexpected={len(unexpected)}"' "$$src"; \
	(cd "$$src" && \
		$(RELEASE_ENV) \
			FLASH_ATTN_CK_MINIMAL_DEBUG='FALSE' \
			FLASH_ATTENTION_METADATA_ONLY='TRUE' \
			FLASH_ATTENTION_SKIP_CUDA_BUILD='TRUE' \
			"$$python_bin" setup.py --name >/dev/null); \
	case '$(VERIFY_SDIST_BUILD)' in \
		1) \
			$(MAKE) --no-print-directory doctor; \
			echo 'Building minimal closure from the extracted source archive...'; \
			(cd "$$src" && \
				$(RELEASE_ENV) \
					FLASH_ATTN_CK_MINIMAL_DEBUG='TRUE' \
					FLASH_ATTN_CK_MINIMAL_FAMILY='$(MINIMAL_FAMILY)' \
					FLASH_ATTN_CK_MINIMAL_OPTDIM='$(MINIMAL_OPTDIM)' \
					FLASH_ATTN_CK_MINIMAL_DTYPE='$(MINIMAL_DTYPE)' \
					FLASH_ATTN_CK_MINIMAL_NONCAUSAL_ONLY='$(MINIMAL_NONCAUSAL_ONLY)' \
					FLASH_ATTN_CK_MINIMAL_BATCH_ONLY='$(MINIMAL_BATCH_ONLY)' \
					"$$python_bin" setup.py build_ext --inplace --force); \
			extensions=( "$$src"/flash_attn_2_cuda*.so ); \
			test "$${#extensions[@]}" -eq 1 && test -f "$${extensions[0]}" || { \
				echo 'ERROR: extracted sdist build did not produce exactly one extension' >&2; exit 1; \
			}; \
			;; \
		0) echo 'Skipping extracted-tree compile because VERIFY_SDIST_BUILD=0.' ;; \
		*) echo 'ERROR: VERIFY_SDIST_BUILD must be 0 or 1' >&2; exit 1 ;; \
	esac; \
	echo "Verified offline source archive: $${archives[0]}"

test-smoke: build-minimal
	@mkdir -p testoutput/make-smoke
	$(PYTHON) scripts/test_prebuilt_flash_attn_ck.py \
		--device '$(DEVICE)' \
		--dims '$(MINIMAL_OPTDIM)' \
		--dtypes '$(MINIMAL_DTYPE)' \
		--causal false \
		--seqlens '$(SMOKE_SEQLENS)' \
		--repeats 1 \
		--fail-fast \
		--csv testoutput/make-smoke/correctness.csv \
		--summary-csv testoutput/make-smoke/summary.csv \
		--json testoutput/make-smoke/report.json

test-bf16-fp32-audit: release-test-config-check assert-local-full-extension
	@mkdir -p testoutput/make-release/bf16
	$(PYTHON) scripts/audit_bf16_boundary_fp32.py \
		--device '$(DEVICE)' \
		--grad-scale '$(RELEASE_GRAD_SCALE)' \
		--scaled-atol '$(BF16_BOUNDARY_BWD_SCALED_ATOL)' \
		--relative-atol '$(BF16_BOUNDARY_BWD_REL_ATOL)' \
		--max-ulp '$(BF16_BOUNDARY_MAX_ULP)'

release-test-config-check:
	@check() { test "$$2" = "$$3" || { echo "ERROR: $$1 must be '$$3' for the frozen release gate; got '$$2'" >&2; exit 1; }; }; \
		check GPU_ARCHS '$(GPU_ARCHS)' gfx1100; \
		check RELEASE_DIMS '$(RELEASE_DIMS)' 32,64,128,256; \
		check RELEASE_SEQLENS '$(RELEASE_SEQLENS)' 768,1024,2048; \
		check RELEASE_REPEATS '$(RELEASE_REPEATS)' 2; \
		check RELEASE_GRAD_SCALE '$(RELEASE_GRAD_SCALE)' 100; \
		check FP16_FWD_ATOL '$(FP16_FWD_ATOL)' 2.0e-3; \
		check FP16_BWD_SCALED_ATOL '$(FP16_BWD_SCALED_ATOL)' 3.0e-3; \
		check FP16_BWD_REL_ATOL '$(FP16_BWD_REL_ATOL)' 1.0e-3; \
		check BF16_FWD_ATOL '$(BF16_FWD_ATOL)' 2.0e-2; \
		check BF16_BWD_SCALED_ATOL '$(BF16_BWD_SCALED_ATOL)' 3.0e-2; \
		check BF16_BWD_REL_ATOL '$(BF16_BWD_REL_ATOL)' 3.0e-2; \
		check BF16_BOUNDARY_BWD_SCALED_ATOL '$(BF16_BOUNDARY_BWD_SCALED_ATOL)' 4.1e-2; \
		check BF16_BOUNDARY_BWD_REL_ATOL '$(BF16_BOUNDARY_BWD_REL_ATOL)' 8.0e-3; \
		check BF16_BOUNDARY_MAX_ULP '$(BF16_BOUNDARY_MAX_ULP)' 1

test-release-pytest: assert-local-full-extension
	@mkdir -p testoutput/make-release
	@rm -f testoutput/make-release/pytest.xml
	@mapfile -t nodes < tests/release_ck_nodes.txt; \
		test "$${#nodes[@]}" -eq 40 || { \
			echo "ERROR: expected 40 frozen pytest nodes, found $${#nodes[@]}" >&2; exit 1; \
		}; \
		$(PYTEST) -q --junitxml=testoutput/make-release/pytest.xml "$${nodes[@]}"; \
		$(PYTHON) -c 'import xml.etree.ElementTree as ET; root=ET.parse("testoutput/make-release/pytest.xml").getroot(); suite=root if root.tag == "testsuite" else root.find(".//testsuite"); tests=int(suite.attrib["tests"]); skipped=int(suite.attrib.get("skipped", 0)); failures=int(suite.attrib.get("failures", 0)); errors=int(suite.attrib.get("errors", 0)); assert tests == 40, f"expected 40 executed nodes, got {tests}"; assert skipped == 0, f"release nodes skipped: {skipped}"; assert failures == 0 and errors == 0, f"release node failures={failures}, errors={errors}"; print(f"Frozen pytest nodes: {tests} passed, 0 skipped")'

test-release: release-test-config-check
	@$(MAKE) --no-print-directory build-full
	@$(MAKE) --no-print-directory assert-local-full-extension
	@mkdir -p testoutput/make-release/fp16 testoutput/make-release/bf16
	@echo 'Running strict FP16 release matrix...'
	$(PYTHON) scripts/test_prebuilt_flash_attn_ck.py \
		--device '$(DEVICE)' \
		--dims '$(RELEASE_DIMS)' \
		--dtypes fp16 \
		--causal both \
		--seqlens '$(RELEASE_SEQLENS)' \
		--repeats '$(RELEASE_REPEATS)' \
		--grad-scale '$(RELEASE_GRAD_SCALE)' \
		--fp16-fwd-atol '$(FP16_FWD_ATOL)' \
		--bwd-scaled-atol '$(FP16_BWD_SCALED_ATOL)' \
		--bwd-rel-atol '$(FP16_BWD_REL_ATOL)' \
		--csv testoutput/make-release/fp16/correctness.csv \
		--summary-csv testoutput/make-release/fp16/summary.csv \
		--json testoutput/make-release/fp16/report.json
	@echo 'Running BF16 release matrix with strict gates plus the documented ULP-aware fallback...'
	$(PYTHON) scripts/test_prebuilt_flash_attn_ck.py \
		--device '$(DEVICE)' \
		--dims '$(RELEASE_DIMS)' \
		--dtypes bf16 \
		--causal both \
		--seqlens '$(RELEASE_SEQLENS)' \
		--repeats '$(RELEASE_REPEATS)' \
		--grad-scale '$(RELEASE_GRAD_SCALE)' \
		--bf16-fwd-atol '$(BF16_FWD_ATOL)' \
		--bwd-scaled-atol '$(BF16_BWD_SCALED_ATOL)' \
		--bwd-rel-atol '$(BF16_BWD_REL_ATOL)' \
		--bf16-boundary-bwd-scaled-atol '$(BF16_BOUNDARY_BWD_SCALED_ATOL)' \
		--bf16-boundary-bwd-rel-atol '$(BF16_BOUNDARY_BWD_REL_ATOL)' \
		--bf16-boundary-max-ulp '$(BF16_BOUNDARY_MAX_ULP)' \
		--csv testoutput/make-release/bf16/correctness.csv \
		--summary-csv testoutput/make-release/bf16/summary.csv \
		--json testoutput/make-release/bf16/report.json
	@$(PYTHON) -c 'import json; report=json.load(open("testoutput/make-release/bf16/report.json", encoding="utf-8")); allowed={(768, 64, 7168017), (2048, 64, 8448018), (2048, 256, 27648018)}; boundary={(int(row["seqlen"]), int(row["headdim"]), int(row["seed"])) for row in report["correctness"] if row.get("gate") == "bf16_ulp_boundary"}; unexpected=sorted(boundary-allowed); assert not unexpected, "unexpected BF16 ULP fallback scenarios: " + repr(unexpected); print("BF16 ULP fallback scenarios:", sorted(boundary))'
	@$(MAKE) --no-print-directory test-bf16-fp32-audit
	@$(MAKE) --no-print-directory test-release-pytest

clean:
	@rm -rf build dist build_logs .eggs ./*.egg-info \
		testoutput/make-smoke testoutput/make-release
	@find . -maxdepth 1 -type f -name 'flash_attn_2_cuda*.so' -delete
	@echo 'Removed local build products.'
