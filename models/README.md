# models/

This directory contains GGUF model files for inference and testing.

## Supported Format

- **GGUF** (`.gguf`) - The only supported model format
- Models must be quantized or converted to GGUF format using llama.cpp tools

## Default Test Model

The default test model is: `Qwen3-8B-Q6_K.gguf`

You can override this by setting the `LLAMA_TEST_MODEL` environment variable:
```bash
export LLAMA_TEST_MODEL=/path/to/your/model.gguf
```

## Recommended Models

Any GGUF model compatible with llama.cpp will work. Popular options:
- Qwen3 series (Qwen3-8B, Qwen3-14B, Qwen3-30B-A3B)
- Gemma series (gemma-2b, gemma-7b)
- Mistral/Ministral series
- Phi series
- LLaMA series

## Quantization Levels

Common quantization formats (from highest to lowest quality):
- `F16` - 16-bit float (largest, highest quality)
- `Q8_0` - 8-bit quantization
- `Q6_K` - 6-bit k-quant (good balance)
- `Q5_K_M` - 5-bit k-quant medium
- `Q4_K_M` - 4-bit k-quant medium
- `Q4_K_S` - 4-bit k-quant small (smallest, fastest)

## Notes

- Model files are not tracked by git (listed in `.gitignore`)
- Ensure sufficient GPU VRAM for your chosen model size
- Use `n_gpu_layers=-1` in `LlamaConfig` to offload all layers to GPU
