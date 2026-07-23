from __future__ import annotations

from pathlib import Path


def export_smolvlm_onnx_int8(source: str, out_dir) -> Path:
    from optimum.onnxruntime import ORTModelForVision2Seq
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime import ORTQuantizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = ORTModelForVision2Seq.from_pretrained(source, export=True)
    model.save_pretrained(out_dir)
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
    for onnx_file in out_dir.glob("*.onnx"):
        quantizer = ORTQuantizer.from_pretrained(out_dir, file_name=onnx_file.name)
        quantizer.quantize(save_dir=out_dir, quantization_config=qconfig)
    return out_dir
