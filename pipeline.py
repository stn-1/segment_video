import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

SCRIPT_DIR = Path(__file__).parent
DINOV2_REPO = SCRIPT_DIR / "dinov2"
WEIGHTS_PATH = DINOV2_REPO / "weights" / "dinov2_vitb14_reg4_pretrain.pth"

sys.path.insert(0, str(DINOV2_REPO))

COSINE_THRESHOLD = 0.82
KEYFRAMES_PER_SEGMENT = 5
ADAPTIVE_THRESHOLD = 4.0
MIN_SCENE_LEN = 60  # frames (~2s at 30fps)

DINO_TRANSFORM = transforms.Compose([
    transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

INTERNVL_TRANSFORM = transforms.Compose([
    transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


# ── Model loaders ────────────────────────────────────────────────────────────

def load_dinov2(device: str) -> torch.nn.Module:
    from dinov2.hub.backbones import _make_dinov2_model

    model = _make_dinov2_model(
        arch_name="vit_base",
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        pretrained=False,
    )
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device)


def load_internvl2(model_name_or_path: str, device: str):
    from transformers import AutoTokenizer, AutoModel

    # model_name_or_path: HuggingFace ID or local path
    # e.g. "OpenGVLab/InternVL2-2B" or "/path/to/local/model"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval().to(device)
    return tokenizer, model


# ── Stage 1: PySceneDetect ───────────────────────────────────────────────────

def stage1_detect_scenes(video_path: str) -> tuple[list, float]:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import AdaptiveDetector

    video = open_video(video_path)
    manager = SceneManager()
    manager.add_detector(AdaptiveDetector(
        adaptive_threshold=ADAPTIVE_THRESHOLD,
        min_scene_len=MIN_SCENE_LEN,
    ))
    manager.detect_scenes(video)

    scenes = []
    for start_tc, end_tc in manager.get_scene_list():
        scenes.append({
            "start_frame": start_tc.get_frames(),
            "end_frame": end_tc.get_frames(),
            "start_time": start_tc.get_seconds(),
            "end_time": end_tc.get_seconds(),
        })

    fps = video.frame_rate
    return scenes, fps


# ── Stage 2: DINOv2 verify & merge ──────────────────────────────────────────

def sample_frames(video_path: str, start_frame: int, end_frame: int, n: int = 5) -> list:
    total = end_frame - start_frame
    if total <= 0:
        return []

    if total <= n:
        indices = list(range(start_frame, end_frame))
    else:
        # evenly spaced across the segment
        indices = [start_frame + int(total * i / (n - 1)) for i in range(n)]
        indices[-1] = min(indices[-1], end_frame - 1)

    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append((idx, img))
    cap.release()
    return frames


@torch.no_grad()
def extract_segment_embedding(frames: list, model: torch.nn.Module, device: str) -> tuple:
    tensors = torch.stack([DINO_TRANSFORM(img) for _, img in frames]).to(device)
    embs = model(tensors)  # (N, 768) — CLS tokens
    embs = embs.cpu()
    mean_emb = embs.mean(dim=0)

    # keyframe = frame whose embedding is closest to the mean
    sims = F.cosine_similarity(embs, mean_emb.unsqueeze(0).expand_as(embs))
    kf_local = sims.argmax().item()
    kf_global_idx = frames[kf_local][0]

    return mean_emb, kf_global_idx


def stage2_verify_merge(video_path: str, scenes: list, model: torch.nn.Module, device: str) -> list:
    print(f"  Computing embeddings for {len(scenes)} raw scenes...")

    for scene in scenes:
        frames = sample_frames(video_path, scene["start_frame"], scene["end_frame"], KEYFRAMES_PER_SEGMENT)
        if not frames:
            scene["embedding"] = torch.zeros(768)
            scene["keyframe_idx"] = scene["start_frame"]
            continue
        emb, kf_idx = extract_segment_embedding(frames, model, device)
        scene["embedding"] = emb
        scene["keyframe_idx"] = kf_idx

    # greedy left-to-right merge, repeat until stable
    changed = True
    while changed:
        changed = False
        merged = []
        i = 0
        while i < len(scenes):
            if i + 1 < len(scenes):
                sim = F.cosine_similarity(
                    scenes[i]["embedding"].unsqueeze(0),
                    scenes[i + 1]["embedding"].unsqueeze(0),
                ).item()
                if sim > COSINE_THRESHOLD:
                    new_scene = {
                        "start_frame": scenes[i]["start_frame"],
                        "end_frame": scenes[i + 1]["end_frame"],
                        "start_time": scenes[i]["start_time"],
                        "end_time": scenes[i + 1]["end_time"],
                    }
                    frames = sample_frames(
                        video_path,
                        new_scene["start_frame"],
                        new_scene["end_frame"],
                        KEYFRAMES_PER_SEGMENT,
                    )
                    if frames:
                        emb, kf_idx = extract_segment_embedding(frames, model, device)
                        new_scene["embedding"] = emb
                        new_scene["keyframe_idx"] = kf_idx
                    else:
                        new_scene["embedding"] = (scenes[i]["embedding"] + scenes[i + 1]["embedding"]) / 2
                        new_scene["keyframe_idx"] = scenes[i]["keyframe_idx"]
                    merged.append(new_scene)
                    i += 2
                    changed = True
                    continue
            merged.append(scenes[i])
            i += 1
        scenes = merged

    return scenes


# ── Stage 3: Caption ─────────────────────────────────────────────────────────

def generate_caption(image_path: str, tokenizer, model, device: str) -> str:
    image = Image.open(image_path).convert("RGB")
    pixel_values = (
        INTERNVL_TRANSFORM(image)
        .unsqueeze(0)
        .to(torch.bfloat16)
        .to(device)
    )
    question = "<image>\nDescribe this travel scene briefly in one sentence."
    response = model.chat(
        tokenizer,
        pixel_values,
        question,
        {"max_new_tokens": 128, "do_sample": False},
    )
    return response


# ── Keyframe saver ───────────────────────────────────────────────────────────

def save_keyframe(video_path: str, frame_idx: int, out_path: Path) -> str:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(str(out_path), frame)
        return str(out_path)
    return ""


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Travel video segmentation pipeline")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", default="output.json", help="Output JSON path (default: output.json)")
    parser.add_argument("--keyframes-dir", default="keyframes", help="Directory to save keyframe images")
    parser.add_argument("--no-caption", action="store_true", help="Skip InternVL2 caption generation")
    parser.add_argument(
        "--internvl-model",
        default="OpenGVLab/InternVL2-2B",
        help="HuggingFace model ID or local path for InternVL2-2B",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run models on (default: cuda if available)",
    )
    args = parser.parse_args()

    keyframes_dir = Path(args.keyframes_dir)
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Video : {args.video}")

    # Load DINOv2
    print("\nLoading DINOv2 ViT-B/14 reg4...")
    dino = load_dinov2(args.device)

    # Stage 1
    print("\nStage 1 — PySceneDetect AdaptiveDetector...")
    raw_scenes, fps = stage1_detect_scenes(args.video)
    print(f"  → {len(raw_scenes)} raw scenes (fps={fps:.2f})")

    # Stage 2
    print("\nStage 2 — DINOv2 verify & merge...")
    final_scenes = stage2_verify_merge(args.video, raw_scenes, dino, args.device)
    print(f"  → {len(final_scenes)} segments after merge")

    # Free DINOv2 VRAM before loading InternVL2
    del dino
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # Load InternVL2
    tokenizer = caption_model = None
    if not args.no_caption:
        print(f"\nLoading InternVL2-2B from '{args.internvl_model}'...")
        tokenizer, caption_model = load_internvl2(args.internvl_model, args.device)

    # Save keyframes + generate captions
    print("\nSaving keyframes and generating captions...")
    results = []
    for i, scene in enumerate(final_scenes):
        kf_path = keyframes_dir / f"segment_{i:04d}.jpg"
        saved = save_keyframe(args.video, scene["keyframe_idx"], kf_path)

        caption = ""
        if not args.no_caption and saved and caption_model is not None:
            caption = generate_caption(saved, tokenizer, caption_model, args.device)
            print(f"  [{i:04d}] {scene['start_time']:.1f}s–{scene['end_time']:.1f}s | {caption}")

        results.append({
            "segment_id": i,
            "start_time": round(scene["start_time"], 3),
            "end_time": round(scene["end_time"], 3),
            "duration": round(scene["end_time"] - scene["start_time"], 3),
            "start_frame": scene["start_frame"],
            "end_frame": scene["end_frame"],
            "keyframe_path": saved,
            "embedding": scene["embedding"].tolist(),
            "caption": caption,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone.")
    print(f"  Segments : {len(results)}")
    print(f"  Output   : {args.output}")
    print(f"  Keyframes: {keyframes_dir}/")


if __name__ == "__main__":
    main()
