"""CLI: export a model to a quantized backend ahead of benchmarking."""
from __future__ import annotations

import argparse

from vlmbench.backends.onnx_export import export_smolvlm_onnx_int8
from vlmbench._paths import quant_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    args = parser.parse_args()
    if args.model == "smolvlm-256m" and args.backend == "onnx-int8":
        out = export_smolvlm_onnx_int8(
            "HuggingFaceTB/SmolVLM-256M-Instruct",
            quant_dir(args.model, args.backend))
        print(f"exported to {out}")
    else:
        raise SystemExit(f"no exporter for {args.model}/{args.backend}")


if __name__ == "__main__":
    main()
