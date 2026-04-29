#!/usr/bin/env python3
import argparse
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import run_2d_grounding as api_impl


class HFBackend:
    def __init__(
        self,
        model_name: str,
        dtype: str,
        device_map: str,
        attn_implementation: str | None,
        max_new_tokens: int,
        trust_remote_code: bool,
    ) -> None:
        model_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if dtype == "auto":
            model_kwargs["dtype"] = "auto"
        else:
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            model_kwargs["torch_dtype"] = dtype_map[dtype]
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        print(f"[HF] Loading model {model_name} with device_map={device_map}, dtype={dtype}")
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL 2D grounding locally with HuggingFace transformers."
    )
    parser.add_argument("--image", help="Local image path or http(s) URL.")
    parser.add_argument("--image-dir", help="Directory containing local images to process in batch.")
    parser.add_argument("--data-root", help="Dataset root for scene-based processing.")
    parser.add_argument("--mask-root", help="Optional root containing scene/frame mask folders for mask-guided refinement.")
    parser.add_argument("--memory-image", help="Optional local image path or http(s) URL to pass as an additional reference image for every inference.")
    parser.add_argument("--memory-root", help="Optional directory containing prompt-key memory images, e.g. memory_root/door_handle/*.png.")
    parser.add_argument("--require-memory", action="store_true", help="Fail when --memory-root is set but no memory image can be resolved.")
    parser.add_argument("--scene-json", help="Path to one scene annotation JSON, e.g. 421254.json.")
    parser.add_argument("--scene-json-dir", help="Directory containing many scene annotation JSON files.")
    parser.add_argument("--scene-id", help="Optional scene id to process one scene from --scene-json-dir.")
    parser.add_argument(
        "--scene-ids",
        help="Optional comma-separated scene ids to process from --scene-json-dir, e.g. 421254,421255.",
    )
    parser.add_argument("--frame-indices", help="Optional comma-separated frame indices to process, e.g. 0,5,10. Overrides --stride and --reverse.")
    parser.add_argument("--append-mode", action="store_true", help="Append new frame results to existing JSON files instead of overwriting. Only works with --frame-indices.")
    parser.add_argument("--prompt", help="Grounding prompt to send to the model.")
    parser.add_argument("--prompt-file", help="Optional text file containing one prompt per non-empty line.")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct", help="HF model id or local path.")
    parser.add_argument("--base-url", help="Ignored by the HuggingFace backend. Kept for CLI compatibility with the API script.")
    parser.add_argument("--mode", choices=["auto", "bbox", "point"], default="auto")
    parser.add_argument("--output-image", help="Optional output path for a rendered visualization in single-image mode.")
    parser.add_argument("--output-json", help="Single-image mode: save the raw model response. Batch mode: save the summary JSON.")
    parser.add_argument("--output-image-dir", help="Directory to save rendered visualizations in batch mode.")
    parser.add_argument("--output-json-dir", help="Deprecated in batch mode. Kept only for backward compatibility.")
    parser.add_argument("--glob", default="*.jpg,*.jpeg,*.png,*.webp,*.bmp")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--min-pixels", type=int, default=64 * 32 * 32)
    parser.add_argument("--max-pixels", type=int, default=9800 * 32 * 32)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map. Use auto for multi-GPU sharding.")
    parser.add_argument("--attn-implementation", default=None, help="Optional attention backend, e.g. flash_attention_2 or sdpa.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    args = parser.parse_args()
    api_impl.validate_args(args)
    return args


def build_backend(args: argparse.Namespace) -> HFBackend:
    return HFBackend(
        model_name=args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        trust_remote_code=args.trust_remote_code,
    )


def infer_hf_image_bytes(
    client: HFBackend,
    image_bytes: bytes,
    prompt: str,
    model: str,
    min_pixels: int,
    max_pixels: int,
    mime_type: str = "image/jpeg",
    memory_images: list[tuple[bytes, str, str]] | None = None,
) -> dict[str, Any]:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": "Current image to ground:"},
        {"type": "image", "image": image},
    ]
    if memory_images:
        user_content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "The following memory images are references selected for this prompt. "
                        "They may contain stitched ground-truth projections from nearby or previous frames. "
                        "Use them only as visual reference for the requested object category; output boxes "
                        "must be for the current image, not for the memory images."
                    ),
                },
            ]
        )
        for memory_index, (memory_image_bytes, _memory_mime_type, memory_label) in enumerate(memory_images, start=1):
            memory_image = Image.open(BytesIO(memory_image_bytes)).convert("RGB")
            user_content.extend(
                [
                    {"type": "text", "text": f"Reference memory {memory_index}: {memory_label}"},
                    {"type": "image", "image": memory_image},
                ]
            )
    user_content.append({"type": "text", "text": prompt})
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a visual grounding model. "
                        "Do not output reasoning, explanation, chain-of-thought, or analysis. "
                        "Do not use markdown fences. "
                        "Return only the final JSON result."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    inputs = client.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    target_device = client.model.device
    inputs = {key: value.to(target_device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated_ids = client.model.generate(
            **inputs,
            max_new_tokens=client.max_new_tokens,
            do_sample=False,
        )
    trimmed_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    text = client.processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return {
        "response_text": api_impl.sanitize_response_text(text or ""),
        "usage": None,
    }


def infer_hf(
    client: HFBackend,
    image_ref: str,
    prompt: str,
    model: str,
    min_pixels: int,
    max_pixels: int,
    memory_refs: list[str] | None = None,
) -> dict[str, Any]:
    memory_images = [
        (api_impl.load_image_bytes(memory_ref), api_impl.mime_type_for_image_ref(memory_ref), memory_ref)
        for memory_ref in (memory_refs or [])
    ]
    return infer_hf_image_bytes(
        client=client,
        image_bytes=api_impl.load_image_bytes(image_ref),
        prompt=prompt,
        model=model,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        mime_type=api_impl.mime_type_for_image_ref(image_ref),
        memory_images=memory_images,
    )


def patch_backend_functions() -> None:
    api_impl.require_api_key = lambda: ""
    api_impl.build_client = lambda *args, **kwargs: None
    api_impl.infer = infer_hf
    api_impl.infer_image_bytes = infer_hf_image_bytes


def main() -> None:
    args = parse_args()
    patch_backend_functions()
    client = build_backend(args)
    if args.scene_json or args.scene_json_dir or args.scene_id or args.scene_ids:
        api_impl.process_scene_jsons(client, args)
        return
    if args.image_dir:
        api_impl.process_batch(client, args)
        return

    prompts = api_impl.load_prompts(args.prompt, args.prompt_file)
    if len(prompts) == 1 and not args.memory_root:
        api_impl.process_one(
            client=client,
            image_ref=args.image,
            prompt=prompts[0]["text"],
            model=args.model,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            mode=args.mode,
            output_json=args.output_json,
            output_image=args.output_image,
            memory_refs=[args.memory_image] if args.memory_image else None,
        )
        return

    api_impl.process_one_with_prompts(
        client=client,
        image_ref=args.image,
        prompts=prompts,
        model=args.model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        mode=args.mode,
        output_json=args.output_json,
        output_image=args.output_image,
        memory_refs=[args.memory_image] if args.memory_image else None,
        memory_root=args.memory_root,
        require_memory=args.require_memory,
    )


if __name__ == "__main__":
    main()
