#!/usr/bin/env python3
import argparse
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import re

import requests
from openai import OpenAI
from PIL import Image, ImageColor, ImageDraw, ImageFont
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


COLORS = [
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "pink",
    "purple",
    "brown",
    "gray",
    "beige",
    "turquoise",
    "cyan",
    "magenta",
    "lime",
    "navy",
    "maroon",
    "teal",
    "olive",
    "coral",
    "lavender",
    "violet",
    "gold",
    "silver",
] + list(ImageColor.colormap.keys())

BASE_URLS = {
    "beijing": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "singapore": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "virginia": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL 2D grounding against one image or a directory of images."
    )
    parser.add_argument("--image", help="Local image path or http(s) URL.")
    parser.add_argument("--image-dir", help="Directory containing local images to process in batch.")
    parser.add_argument("--data-root", help="Dataset root for scene-based processing.")
    parser.add_argument("--scene-json", help="Path to one scene annotation JSON, e.g. 421254.json.")
    parser.add_argument("--scene-json-dir", help="Directory containing many scene annotation JSON files.")
    parser.add_argument("--scene-id", help="Optional scene id to process one scene from --scene-json-dir.")
    parser.add_argument("--prompt", help="Grounding prompt to send to the model.")
    parser.add_argument(
        "--prompt-file",
        help="Optional text file containing one prompt per non-empty line.",
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl-235b-a22b-instruct",
        help="DashScope model name.",
    )
    parser.add_argument(
        "--region",
        choices=sorted(BASE_URLS.keys()),
        default="beijing",
        help="DashScope region for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "bbox", "point"],
        default="auto",
        help="How to render the response. auto tries bbox first, then point.",
    )
    parser.add_argument(
        "--output-image",
        help="Optional output path for a rendered visualization in single-image mode.",
    )
    parser.add_argument(
        "--output-json",
        help="Single-image mode: save the raw model response. Batch mode: save the summary JSON.",
    )
    parser.add_argument(
        "--output-image-dir",
        help="Directory to save rendered visualizations in batch mode.",
    )
    parser.add_argument(
        "--output-json-dir",
        help="Deprecated in batch mode. Kept only for backward compatibility.",
    )
    parser.add_argument(
        "--glob",
        default="*.jpg,*.jpeg,*.png,*.webp,*.bmp",
        help="Comma-separated filename patterns for batch mode.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="In batch mode, process every N-th image after sorting. Default: 1.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=64 * 32 * 32,
        help="Minimum pixel budget passed to the API.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=9800 * 32 * 32,
        help="Maximum pixel budget passed to the API.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    source_flags = [bool(args.image), bool(args.image_dir), bool(args.scene_json), bool(args.scene_json_dir or args.scene_id)]
    if sum(source_flags) != 1:
        raise SystemExit("Specify exactly one source mode: --image, --image-dir, --scene-json, or --scene-json-dir/--scene-id.")
    if (args.image or args.image_dir) and bool(args.prompt) == bool(args.prompt_file):
        raise SystemExit("For image/image-dir mode, specify exactly one of --prompt or --prompt-file.")
    if (args.scene_json or args.scene_json_dir or args.scene_id) and (args.prompt or args.prompt_file):
        raise SystemExit("In scene-json mode, prompts are generated automatically from affordance categories; do not pass --prompt or --prompt-file.")
    if args.scene_id and not args.scene_json_dir:
        raise SystemExit("--scene-id requires --scene-json-dir.")
    if (args.scene_json or args.scene_json_dir or args.scene_id) and not args.data_root:
        raise SystemExit("Scene-json mode requires --data-root.")
    if args.image_dir and args.output_image:
        raise SystemExit("--output-image is only valid with --image.")
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1.")


def require_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing DASHSCOPE_API_KEY. Export it first, for example:\n"
            'export DASHSCOPE_API_KEY="sk-..."'
        )
    return api_key


def load_image_bytes(image_ref: str) -> bytes:
    if image_ref.startswith(("http://", "https://")):
        response = requests.get(image_ref, timeout=60)
        response.raise_for_status()
        return response.content
    return Path(image_ref).read_bytes()


def load_pil_image(image_ref: str) -> Image.Image:
    return Image.open(BytesIO(load_image_bytes(image_ref))).convert("RGB")


def is_url(image_ref: str) -> bool:
    return image_ref.startswith(("http://", "https://"))


def iter_image_refs(image_dir: str, glob_patterns: str) -> list[str]:
    root = Path(image_dir)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"--image-dir does not exist or is not a directory: {image_dir}")
    refs: list[Path] = []
    for pattern in [item.strip() for item in glob_patterns.split(",") if item.strip()]:
        refs.extend(root.glob(pattern))
    unique_refs = sorted({path.resolve() for path in refs if path.is_file()})
    if not unique_refs:
        raise SystemExit(f"No images found in {image_dir} matching patterns: {glob_patterns}")
    return [str(path) for path in unique_refs]


def slugify_prompt_key(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "prompt"


def prompt_key_from_text(text: str) -> str:
    lower = text.lower()
    keyword_map = [
        ("door_handle", ["door handle"]),
        ("drawer_handle", ["drawer handle"]),
        ("window_handle", ["window handle"]),
        ("left_remote", ["left remote"]),
        ("right_remote", ["right remote"]),
        ("light_switch", ["light switch"]),
        ("lamp_switch", ["lamp switch"]),
        ("power_plug", ["power plug"]),
        ("thermostatic_radiator_valve", ["thermostatic radiator valve"]),
    ]
    for key, phrases in keyword_map:
        if any(phrase in lower for phrase in phrases):
            return key
    return slugify_prompt_key(text[:80])


def affordance_key_from_category(category: str) -> str:
    normalized = category.strip().lower()
    mapping = {
        "door handle": "door_handle",
        "drawer handle": "drawer_handle",
        "window handle": "window_handle",
        "remote control": "remote_control",
        "switch": "switch",
        "light switch": "light_switch",
        "lamp switch": "lamp_switch",
        "power plug": "power_plug",
        "thermostatic radiator valve": "thermostatic_radiator_valve",
    }
    return mapping.get(normalized, slugify_prompt_key(normalized))


def prompt_from_affordance_category(category: str) -> dict[str, str]:
    normalized = category.strip().lower()
    prompt_map = {
        "door handle": (
            "door_handle",
            "Locate two separate tight bounding boxes for a true door handle mounted on a door and output JSON only: "
            "(1) the fixed base or mounting plate attached to the door, exclude the movable lever; "
            "(2) the movable lever or grip part used to open the door, exclude the base, door surface, and surrounding objects. "
            "Return JSON only as a list of objects with bbox_2d fields and no label field."
        ),
        "drawer handle": (
            "drawer_handle",
            "Locate only the movable pull or grip part of a drawer handle mounted on a drawer front. "
            "Exclude fixed mounts, screws, the drawer front, surrounding furniture, and any non-graspable base region. "
            "Only box the minimum directly operable area. Return JSON only as a list of objects with bbox_2d fields and no label field."
        ),
        "window handle": (
            "window_handle",
            "Locate only the movable handle part of a window handle that can be directly turned or grasped. "
            "Exclude the fixed base, mounting plate, window frame, and surrounding structure. "
            "Only box the minimum directly operable area. Return JSON only as a list of objects with bbox_2d fields and no label field."
        ),
        "remote control": (
            "remote_control",
            "Locate all remote control bodies that a robot gripper can directly pick up. "
            "Exclude tables, walls, shadows, black masked regions, and any non-remote objects. "
            "Use tight bounding boxes around each remote body. Return JSON only as a list of objects with bbox_2d fields and no label field."
        ),
        "switch": (
            "switch",
            "Locate only the pressable button, rocker, or toggle part of each switch that can be directly operated. "
            "Exclude the wall plate, frame, screws, surrounding wall, and any non-pressable base region. "
            "Only box the minimum directly operable area. Return JSON only as a list of objects with bbox_2d fields and no label field."
        ),
        "light switch": (
            "light_switch",
            "Locate only the pressable button, rocker, or toggle part of each light switch that can be directly operated. "
            "Exclude the wall plate, frame, screws, surrounding wall, and any non-pressable base region. "
            "Only box the minimum directly operable area. Return JSON only as a list of objects with bbox_2d fields and no label field."
        ),
        "lamp switch": (
            "lamp_switch",
            "Locate two separate tight bounding boxes for the lamp switch and output JSON only: "
            "(1) the central pressable button located in the middle of the switch panel, exclude the outer plate, frame, wall, and surrounding background; "
            "(2) the full switch panel or plate, including the button, exclude the surrounding wall and background. "
            "Return JSON only as a list of objects with bbox_2d fields and no label field."
        ),
        "power plug": (
            "power_plug",
            "Locate only the graspable plug head of each power plug that can be directly pulled or inserted. "
            "Exclude the cable, socket plate, wall, and surrounding surface. "
            "Only box the minimum directly operable area. Return JSON only as a list of objects with bbox_2d fields and no label field."
        ),
        "thermostatic radiator valve": (
            "thermostatic_radiator_valve",
            "Locate only the rotatable control knob or turning head of each thermostatic radiator valve that can be directly rotated. "
            "Exclude the pipe, radiator body, fixed base, wall, and surrounding structure. "
            "Only box the minimum directly operable area. Return JSON only as a list of objects with bbox_2d fields and no label field."
        ),
    }
    if normalized in prompt_map:
        key, text = prompt_map[normalized]
        return {"key": key, "text": text}
    return {
        "key": affordance_key_from_category(category),
        "text": (
            f"Locate only the minimum directly operable area for the affordance category '{category}'. "
            "Exclude fixed bases, non-interactable support structures, background, and surrounding objects. "
            "Return JSON only as a list of objects with bbox_2d fields and no label field."
        ),
    }


def load_prompts(prompt: str | None, prompt_file: str | None) -> list[dict[str, str]]:
    if prompt:
        return [{"key": prompt_key_from_text(prompt), "text": prompt}]
    assert prompt_file is not None
    path = Path(prompt_file)
    if not path.exists() or not path.is_file():
        raise SystemExit(f"--prompt-file does not exist or is not a file: {prompt_file}")
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        prompts.append({"key": prompt_key_from_text(stripped), "text": stripped})
    if not prompts:
        raise SystemExit(f"--prompt-file is empty: {prompt_file}")
    return prompts


def load_scene_annotation_map(json_path: str | Path) -> dict[int, list[str]]:
    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"Scene annotation JSON must be an object: {json_path}")
    frame_map: dict[int, list[str]] = {}
    for frame_key, categories in raw.items():
        if not isinstance(categories, list):
            continue
        cleaned = [str(item).strip() for item in categories if str(item).strip()]
        frame_map[int(frame_key)] = cleaned
    return frame_map


def resolve_scene_json_paths(scene_json: str | None, scene_json_dir: str | None, scene_id: str | None) -> list[Path]:
    if scene_json:
        return [Path(scene_json).expanduser().resolve()]
    assert scene_json_dir is not None
    root = Path(scene_json_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"--scene-json-dir does not exist or is not a directory: {scene_json_dir}")
    if scene_id:
        target = root / f"{scene_id}.json"
        if not target.exists():
            raise SystemExit(f"Scene json not found for scene id {scene_id}: {target}")
        return [target]
    return sorted(path for path in root.glob("*.json") if path.is_file())


def resolve_scene_image_refs(data_root: str, scene_id: str) -> list[str]:
    root = Path(data_root).expanduser().resolve()
    candidate_dirs: list[Path] = []
    scene_root = root / scene_id
    if scene_root.exists() and scene_root.is_dir():
        sequence_dirs = sorted(
            path for path in scene_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
        if sequence_dirs:
            candidate_dirs.append(sequence_dirs[0] / "hires_wide")
            candidate_dirs.append(sequence_dirs[0])
        candidate_dirs.append(scene_root / "hires_wide")
        candidate_dirs.append(scene_root)
    candidate_dirs.extend(
        [
            root / f"hires_wide_{scene_id}",
            root / "data" / scene_id / "hires_wide",
            root / "data" / scene_id,
            root / scene_id,
        ]
    )
    image_paths: list[Path] = []
    seen_dirs: set[Path] = set()
    for candidate in candidate_dirs:
        candidate = candidate.resolve()
        if candidate in seen_dirs:
            continue
        seen_dirs.add(candidate)
        if candidate.exists() and candidate.is_dir():
            image_paths = sorted(
                path for path in candidate.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            )
            if image_paths:
                break
    if not image_paths:
        raise SystemExit(f"Could not find scene images for scene id {scene_id} under data root {data_root}")
    return [str(path.resolve()) for path in image_paths]


def output_stem(image_ref: str) -> str:
    if is_url(image_ref):
        parsed = urlparse(image_ref)
        name = Path(parsed.path).name or "remote_image"
        return Path(name).stem or "remote_image"
    return Path(image_ref).stem


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def build_client(api_key: str, region: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=BASE_URLS[region])


def infer(
    client: OpenAI,
    image_ref: str,
    prompt: str,
    model: str,
    min_pixels: int,
    max_pixels: int,
) -> str:
    import base64

    base64_image = base64.b64encode(load_image_bytes(image_ref)).decode("utf-8")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a visual grounding model. "
                "Do not output reasoning, explanation, chain-of-thought, or analysis. "
                "Do not use markdown fences. "
                "Return only the final JSON result."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    completion = client.chat.completions.create(model=model, messages=messages)
    return sanitize_response_text(completion.choices[0].message.content or "")


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[len("```json") :].strip()
    if stripped.startswith("```"):
        stripped = stripped[len("```") :].strip()
    if stripped.endswith("```"):
        stripped = stripped[:-3].strip()
    return stripped


def parse_json_payload(text: str) -> Any:
    stripped = strip_json_fence(text)
    return json.loads(stripped)


def sanitize_detection_payload(payload: Any) -> Any:
    if isinstance(payload, list):
        sanitized: list[Any] = []
        for item in payload:
            if isinstance(item, dict):
                sanitized.append({key: value for key, value in item.items() if key != "label"})
            else:
                sanitized.append(item)
        return sanitized
    if isinstance(payload, dict):
        return {key: value for key, value in payload.items() if key != "label"}
    return payload


def sanitize_response_text(text: str) -> str:
    try:
        payload = parse_json_payload(text)
    except Exception:
        return text
    sanitized = sanitize_detection_payload(payload)
    return json.dumps(sanitized, ensure_ascii=False)


def coerce_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("Expected a JSON object or list.")


def get_font() -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=16)
    return ImageFont.load_default()


def render_bbox(image: Image.Image, payload: list[dict[str, Any]]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = get_font()
    for idx, item in enumerate(payload):
        if "bbox_2d" not in item:
            continue
        color = COLORS[idx % len(COLORS)]
        x1, y1, x2, y2 = item["bbox_2d"]
        abs_x1 = int(x1 / 1000 * width)
        abs_y1 = int(y1 / 1000 * height)
        abs_x2 = int(x2 / 1000 * width)
        abs_y2 = int(y2 / 1000 * height)
        left, right = sorted((abs_x1, abs_x2))
        top, bottom = sorted((abs_y1, abs_y2))
        draw.rectangle(((left, top), (right, bottom)), outline=color, width=4)
        label = item.get("label", f"item_{idx + 1}")
        draw.text((left + 6, top + 6), str(label), fill=color, font=font)
    return image


def render_point(image: Image.Image, payload: list[dict[str, Any]]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = get_font()
    radius = max(4, min(width, height) // 120)
    for idx, item in enumerate(payload):
        if "point_2d" not in item:
            continue
        color = COLORS[idx % len(COLORS)]
        x, y = item["point_2d"]
        abs_x = int(x / 1000 * width)
        abs_y = int(y / 1000 * height)
        draw.ellipse(
            [(abs_x - radius, abs_y - radius), (abs_x + radius, abs_y + radius)],
            fill=color,
        )
        label = item.get("label", f"point_{idx + 1}")
        draw.text((abs_x + radius + 2, abs_y + radius + 2), str(label), fill=color, font=font)
    return image


def render_response(image: Image.Image, response_text: str, mode: str) -> Image.Image:
    payload = coerce_list(parse_json_payload(response_text))
    if not payload:
        return image
    available_keys = {key for item in payload for key in item.keys()}
    selected_mode = mode
    if mode == "auto":
        if "bbox_2d" in available_keys:
            selected_mode = "bbox"
        elif "point_2d" in available_keys:
            selected_mode = "point"
        else:
            raise ValueError("Response JSON does not contain bbox_2d or point_2d.")
    if selected_mode == "bbox":
        return render_bbox(image, payload)
    return render_point(image, payload)


def save_json(output_json: str, response_text: str) -> None:
    ensure_parent_dir(output_json)
    Path(output_json).write_text(response_text + "\n", encoding="utf-8")


def save_render(output_image: str, image_ref: str, response_text: str, mode: str) -> None:
    image = load_pil_image(image_ref)
    rendered = render_response(image, response_text, mode)
    output_path = Path(output_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)


def process_one(
    client: OpenAI,
    image_ref: str,
    prompt: str,
    model: str,
    min_pixels: int,
    max_pixels: int,
    mode: str,
    output_json: str | None = None,
    output_image: str | None = None,
) -> str:
    response_text = infer(
        client,
        image_ref,
        prompt,
        model,
        min_pixels,
        max_pixels,
    )
    print(f"=== {image_ref} ===")
    print(response_text)

    if output_json:
        save_json(output_json, response_text)
        print(f"Saved JSON to {output_json}")

    if output_image:
        save_render(output_image, image_ref, response_text, mode)
        print(f"Saved visualization to {output_image}")

    return response_text


def render_batch_visualization(
    image_ref: str,
    responses: list[dict[str, Any]],
    output_image: str,
    mode: str,
) -> None:
    image = load_pil_image(image_ref)
    for idx, result in enumerate(responses):
        items = try_parse_detection_items(result["response"])
        if items == []:
            continue
        if mode in {"auto", "bbox"}:
            image = render_bbox(
                image,
                [{"bbox_2d": item["bbox_2d"], "label": result["prompt_key"]} for item in items if "bbox_2d" in item],
            )
            continue
        image = render_point(
            image,
            [{"point_2d": item["point_2d"], "label": result["prompt_key"]} for item in items if "point_2d" in item],
        )
    output_path = Path(output_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def process_one_with_prompts(
    client: OpenAI,
    image_ref: str,
    prompts: list[dict[str, str]],
    model: str,
    min_pixels: int,
    max_pixels: int,
    mode: str,
    output_json: str | None = None,
    output_image: str | None = None,
) -> dict[str, Any]:
    print(f"=== {image_ref} ===")
    all_results: list[dict[str, Any]] = []
    for idx, prompt_item in enumerate(prompts, start=1):
        prompt = prompt_item["text"]
        print(f"--- prompt {idx}/{len(prompts)} ---")
        response_text = infer(
            client,
            image_ref,
            prompt,
            model,
            min_pixels,
            max_pixels,
        )
        print(response_text)
        all_results.append(
            {
                "prompt_index": idx - 1,
                "prompt_key": prompt_item["key"],
                "prompt": prompt,
                "response": response_text,
            }
        )

    payload = {
        "image": image_ref,
        "model": model,
        "results": all_results,
    }

    if output_json:
        ensure_parent_dir(output_json)
        Path(output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved JSON to {output_json}")

    if output_image:
        render_batch_visualization(image_ref, all_results, output_image, mode)
        print(f"Saved visualization to {output_image}")

    return payload


def try_parse_detection_items(response_text: str) -> list[dict[str, Any]]:
    try:
        payload = parse_json_payload(response_text)
        return coerce_list(payload)
    except Exception:
        return []


def process_batch(client: OpenAI, args: argparse.Namespace) -> None:
    all_image_refs = iter_image_refs(args.image_dir, args.glob)
    selected_indices = set(range(0, len(all_image_refs), args.stride))
    prompts = load_prompts(args.prompt, args.prompt_file)
    image_root = Path(args.output_image_dir) if args.output_image_dir else None
    if image_root:
        image_root.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    summary_frames: list[dict[str, Any]] = []
    processed_count = 0
    iterator = enumerate(all_image_refs)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(all_image_refs), desc="Grounding frames", unit="frame")

    for frame_index, image_ref in iterator:
        stem = output_stem(image_ref)
        output_image = str(image_root / f"{stem}.png") if image_root else None

        if frame_index not in selected_indices:
            summary_frames.append(
                {
                    "frame_index": frame_index,
                    "image": image_ref,
                    "processed": False,
                    "skip_reason": f"stride={args.stride}",
                    "objects": {},
                }
            )
            continue

        try:
            if tqdm is not None:
                iterator.set_postfix_str(f"processing={Path(image_ref).name}")
            payload = process_one_with_prompts(
                client=client,
                image_ref=image_ref,
                prompts=prompts,
                model=args.model,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                mode=args.mode,
                output_image=output_image,
            )
            objects: dict[str, list[dict[str, Any]]] = {}
            for result in payload["results"]:
                items = try_parse_detection_items(result["response"])
                prompt_key = result["prompt_key"]
                objects.setdefault(prompt_key, [])
                objects[prompt_key].extend(build_detection_entries(prompt_key, items))
            summary_frames.append(
                {
                    "frame_index": frame_index,
                    "image": image_ref,
                    "processed": True,
                    "objects": objects,
                }
            )
            processed_count += 1
        except Exception as exc:
            failures.append((image_ref, str(exc)))
            print(f"[ERROR] {image_ref}: {exc}")
            summary_frames.append(
                {
                    "frame_index": frame_index,
                    "image": image_ref,
                    "processed": True,
                    "error": str(exc),
                    "objects": {},
                }
            )

    summary_payload = {
        "image_dir": args.image_dir,
        "model": args.model,
        "stride": args.stride,
        "prompts": prompts,
        "frames": summary_frames,
    }
    summary_path = args.output_json or str(Path(args.image_dir) / "grounding_summary.json")
    ensure_parent_dir(summary_path)
    Path(summary_path).write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved summary JSON to {summary_path}")

    print(f"Processed {processed_count}/{len(all_image_refs)} frames with stride {args.stride}.")
    if failures:
        raise SystemExit(
            "Batch completed with failures:\n"
            + "\n".join(f"- {image_ref}: {error}" for image_ref, error in failures)
        )


def dedupe_prompt_items(prompt_items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for item in prompt_items:
        key = f"{item['key']}::{item['text']}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def build_detection_entries(prompt_key: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bboxes = [
        item.get("bbox_2d")
        for item in items
        if isinstance(item.get("bbox_2d"), list) and len(item.get("bbox_2d")) == 4
    ]
    if prompt_key in {"door_handle", "lamp_switch"} and len(bboxes) == 2:
        return [
            {"bbox": bboxes[0], "label": 0},
            {"bbox": bboxes[1], "label": 1},
        ]
    return [{"bbox": bbox, "label": 1} for bbox in bboxes]


def build_scene_frame_summary(
    client: OpenAI,
    args: argparse.Namespace,
    scene_id: str,
    scene_json_path: Path,
    scene_image_refs: list[str],
) -> dict[str, Any]:
    frame_category_map = load_scene_annotation_map(scene_json_path)
    ordered_frame_indices = sorted(frame_category_map.keys())
    scene_vis_root = Path(args.output_image_dir) / scene_id if args.output_image_dir else None
    if scene_vis_root:
        scene_vis_root.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[int, str]] = []
    summary_frames: list[dict[str, Any]] = []
    iterator = ordered_frame_indices
    if tqdm is not None:
        iterator = tqdm(
            ordered_frame_indices,
            total=len(ordered_frame_indices),
            desc=f"Scene {scene_id}",
            unit="frame",
        )

    for frame_index in iterator:
        categories = frame_category_map.get(frame_index, [])
        frame_summary: dict[str, Any] = {
            "frame_index": frame_index,
            "categories": categories,
            "processed": True,
            "objects": {},
        }
        if frame_index < 0 or frame_index >= len(scene_image_refs):
            frame_summary["processed"] = False
            frame_summary["error"] = (
                f"frame index {frame_index} out of range for scene {scene_id} "
                f"with {len(scene_image_refs)} images"
            )
            summary_frames.append(frame_summary)
            failures.append((frame_index, frame_summary["error"]))
            continue

        image_ref = scene_image_refs[frame_index]
        frame_summary["image"] = image_ref
        prompt_items = dedupe_prompt_items(
            [prompt_from_affordance_category(category) for category in categories]
        )
        if not prompt_items:
            summary_frames.append(frame_summary)
            continue

        output_image = None
        if scene_vis_root:
            output_image = str(scene_vis_root / f"{frame_index:06d}_{output_stem(image_ref)}.png")

        try:
            if tqdm is not None:
                iterator.set_postfix_str(f"frame={frame_index}")
            payload = process_one_with_prompts(
                client=client,
                image_ref=image_ref,
                prompts=prompt_items,
                model=args.model,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                mode=args.mode,
                output_image=output_image,
            )
            objects: dict[str, list[dict[str, Any]]] = {}
            for result in payload["results"]:
                items = try_parse_detection_items(result["response"])
                prompt_key = result["prompt_key"]
                objects.setdefault(prompt_key, [])
                objects[prompt_key].extend(build_detection_entries(prompt_key, items))
            frame_summary["objects"] = objects
        except Exception as exc:
            frame_summary["error"] = str(exc)
            failures.append((frame_index, str(exc)))
            print(f"[ERROR] scene={scene_id} frame={frame_index}: {exc}")
        summary_frames.append(frame_summary)

    return {
        "scene_id": scene_id,
        "annotation_json": str(scene_json_path),
        "image_count": len(scene_image_refs),
        "requested_frame_count": len(ordered_frame_indices),
        "frames": summary_frames,
        "failures": [
            {"frame_index": frame_index, "error": error}
            for frame_index, error in failures
        ],
    }


def process_scene_jsons(client: OpenAI, args: argparse.Namespace) -> None:
    scene_json_paths = resolve_scene_json_paths(args.scene_json, args.scene_json_dir, args.scene_id)
    if not scene_json_paths:
        raise SystemExit("No scene json files found.")

    scenes: list[dict[str, Any]] = []
    scene_failures: list[tuple[str, str]] = []
    for scene_json_path in scene_json_paths:
        scene_id = scene_json_path.stem
        try:
            scene_image_refs = resolve_scene_image_refs(args.data_root, scene_id)
            scene_summary = build_scene_frame_summary(
                client=client,
                args=args,
                scene_id=scene_id,
                scene_json_path=scene_json_path,
                scene_image_refs=scene_image_refs,
            )
            scenes.append(scene_summary)
            if scene_summary["failures"]:
                scene_failures.extend(
                    (scene_id, f"frame {item['frame_index']}: {item['error']}")
                    for item in scene_summary["failures"]
                )
        except Exception as exc:
            scene_failures.append((scene_id, str(exc)))
            print(f"[ERROR] scene={scene_id}: {exc}")
            scenes.append(
                {
                    "scene_id": scene_id,
                    "annotation_json": str(scene_json_path),
                    "frames": [],
                    "error": str(exc),
                    "failures": [],
                }
            )

    summary_payload = {
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "model": args.model,
        "mode": args.mode,
        "scenes": scenes,
    }
    summary_path = args.output_json or str(Path.cwd() / "scene_grounding_summary.json")
    ensure_parent_dir(summary_path)
    Path(summary_path).write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved summary JSON to {summary_path}")

    if scene_failures:
        raise SystemExit(
            "Scene grounding completed with failures:\n"
            + "\n".join(f"- scene {scene_id}: {error}" for scene_id, error in scene_failures)
        )


def main() -> None:
    args = parse_args()
    api_key = require_api_key()
    client = build_client(api_key, args.region)
    if args.scene_json or args.scene_json_dir or args.scene_id:
        process_scene_jsons(client, args)
        return
    if args.image_dir:
        process_batch(client, args)
        return

    prompts = load_prompts(args.prompt, args.prompt_file)
    if len(prompts) == 1:
        process_one(
            client=client,
            image_ref=args.image,
            prompt=prompts[0]["text"],
            model=args.model,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            mode=args.mode,
            output_json=args.output_json,
            output_image=args.output_image,
        )
        return

    process_one_with_prompts(
        client=client,
        image_ref=args.image,
        prompts=prompts,
        model=args.model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        mode=args.mode,
        output_json=args.output_json,
        output_image=args.output_image,
    )


if __name__ == "__main__":
    main()
