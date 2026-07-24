"""CLI: prepare a model's quantized weights ahead of benchmarking.

For SmolVLM, optimum cannot export Idefics3 to ONNX, but the model ships
pre-quantized ONNX graphs on the Hub (see docs/known-issues.md I13); this
CLI downloads them so a later benchmark run does not pay the download cost.
"""
from __future__ import annotations

import argparse

from vlmbench.backends.onnx_smolvlm import download_onnx_variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    args = parser.parse_args()
    if args.model == "smolvlm-256m" and args.backend == "onnx-int8":
        out = download_onnx_variant("HuggingFaceTB/SmolVLM-256M-Instruct", "int8")
        print(f"downloaded ONNX int8 graphs to {out}")
    else:
        raise SystemExit(f"no quantized-weight preparation for "
                         f"{args.model}/{args.backend}")


if __name__ == "__main__":
    main()
