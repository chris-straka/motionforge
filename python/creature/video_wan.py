"""The free video step: Wan 2.2 TI2V-5B (Apache-2.0) image-to-video from
the two-camera start frame, sized for an 8 GB card.

    python python/creature/video_wan.py --start start_frame.png --out clip.mp4 \
        [--creature "a low-poly wolf"] [--move "shifts its weight and turns its head"] \
        [--frames 49] [--steps 20] [--seed 1] [--width 1024 --height 512]

The prompt is CanItCode's strict one: one creature, two fixed cameras
(front on the left, side on the right), moving on the spot, and a long
list of no's. The model streams its 5B transformer block by block
through the GPU (diffusers group offloading) because the weights alone
(10 GB in bf16) do not fit; the text encoder runs first and is freed.
Turing has no bf16, so the compute runs in fp16.

This step is pluggable: `detect.py --video` reads any composite video,
however it was made. Whether this model gives a usable clip is recorded
in docs/measurements.md.
"""

import argparse
import gc
import glob
import json
import os
import time

import torch

MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
LICENSE = "Apache-2.0"

PROMPT = ("Split-screen technical reference footage of exactly one {creature}, a plain untextured 3D model, "
          "shown twice at the same moment: the left half is a fixed front camera, the right half is a fixed side camera. "
          "The creature stands on the spot on a flat plain grey background and {move}, slowly and clearly, "
          "feet staying on the ground. Both halves show the identical motion from their own fixed camera. "
          "Static cameras, even flat studio light, sharp focus, nothing else in frame.")
NEGATIVE = ("dust, smoke, fog, particles, motion blur, camera shake, camera movement, pan, zoom, cut, scene change, "
            "second creature, people, text, watermark, logo, shadows on the ground, grass, rocks, trees, floor texture, "
            "colour change, texture, fur detail, lighting change, flicker, deformation, extra legs, missing legs, "
            "walking away, leaving the frame, turning around, jumping, flying, blurry, low quality")


def evict(path):
    try:
        fd = os.open(os.path.realpath(path), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except OSError:
        pass


def load_umt5_low_mem():
    """umT5-xxl (11 GB fp16) shard by shard: each shard's file pages are
    evicted as soon as its tensors are copied, so the peak stays near
    one model plus one shard instead of two models (a 22 GB job limit)."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, UMT5EncoderModel

    root = snapshot_download(MODEL, allow_patterns=["text_encoder/*"], local_files_only=True)
    cfg = AutoConfig.from_pretrained(MODEL, subfolder="text_encoder")
    cfg.torch_dtype = torch.float16
    with torch.device("meta"), init_empty_weights():
        enc = UMT5EncoderModel(cfg)
    import resource

    print(f"encoder skeleton built, rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB", flush=True)
    names = set(k for k, _ in enc.named_parameters()) | set(k for k, _ in enc.named_buffers())
    loaded = set()
    for shard in sorted(glob.glob(os.path.join(root, "text_encoder", "*.safetensors"))):
        # Tensor by tensor (the shards are fp32: twice the model), evicting
        # the shard's file pages as we go.
        with safe_open(shard, framework="pt", device="cpu") as f:
            for n, k in enumerate(f.keys()):
                if k in names:
                    set_module_tensor_to_device(enc, k, "cpu", value=f.get_tensor(k), dtype=torch.float16)
                    loaded.add(k)
                if n % 8 == 7:
                    evict(shard)
        evict(shard)
        gc.collect()
    enc.tie_weights()
    left = [k for k, p in enc.named_parameters() if p.device.type == "meta"]
    if left:
        raise SystemExit(f"text encoder: {len(left)} tensors not in the checkpoint, e.g. {left[:3]}")
    return enc.eval()


def load_low_mem(cls, subfolder):
    """A diffusers model shard by shard, tensor by tensor, in fp16, with
    the file pages evicted as we go (see load_umt5_low_mem)."""
    from accelerate.utils import set_module_tensor_to_device
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    root = snapshot_download(MODEL, allow_patterns=[f"{subfolder}/*"], local_files_only=True)
    cfg = cls.load_config(MODEL, subfolder=subfolder)
    # Built for real in fp16 (10 GB, so computed buffers such as rotary
    # tables exist); the checkpoint then overwrites the parameters in place.
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        model = cls.from_config(cfg)
    finally:
        torch.set_default_dtype(prev)
    names = set(k for k, _ in model.named_parameters()) | set(k for k, _ in model.named_buffers())
    for shard in sorted(glob.glob(os.path.join(root, subfolder, "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for n, k in enumerate(f.keys()):
                if k in names:
                    set_module_tensor_to_device(model, k, "cpu", value=f.get_tensor(k), dtype=torch.float16)
                if n % 8 == 7:
                    evict(shard)
        evict(shard)
        gc.collect()
    return model.eval()


def drop_page_cache(model, subfolder):
    """Evict a subfolder's weight files from the page cache: mmapped
    safetensors count against the job's memory limit after use."""
    try:
        from huggingface_hub import snapshot_download

        root = snapshot_download(model, allow_patterns=[f"{subfolder}/*"], local_files_only=True)
    except Exception:
        return
    for path in glob.glob(os.path.join(root, subfolder, "*.safetensors")):
        try:
            fd = os.open(os.path.realpath(path), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--creature", default="four-legged creature")
    ap.add_argument("--move", default="shifts its weight from side to side, lifts one foot and sets it down, and turns its head")
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--phase", choices=["all", "text", "sample"], default="all",
                    help="internal: the text encoder and the transformer run in separate processes")
    args = ap.parse_args()
    emb_path = os.path.splitext(args.out)[0] + ".prompt.pt"
    if args.phase == "all":
        import subprocess
        import sys

        for phase in ("text", "sample"):
            subprocess.run([sys.executable, os.path.abspath(__file__)] + sys.argv[1:] + ["--phase", phase], check=True)
        return
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel
    from diffusers.utils import export_to_video
    from PIL import Image
    from transformers import AutoTokenizer

    t0 = time.time()
    prompt = PROMPT.format(creature=args.creature, move=args.move)
    log = {"model": MODEL, "license": LICENSE, "prompt": prompt, "negative": NEGATIVE, "args": vars(args),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}

    if args.phase == "sample":
        saved = torch.load(emb_path)
        pos, neg = saved["pos"], saved["neg"]
        log["text_seconds"] = saved["seconds"]
        return sample(args, pos, neg, log, t0)

    # 1. Text (its own process: the 11 GB encoder is gone when the transformer loads).
    tok = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    enc = load_umt5_low_mem()
    from diffusers.hooks import apply_group_offloading

    apply_group_offloading(enc, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), offload_type="leaf_level", use_stream=True)

    def embed(text):
        ids = tok(text, padding="max_length", max_length=512, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        mask = ids.attention_mask
        with torch.no_grad():
            out = enc(ids.input_ids.cuda(), mask.cuda()).last_hidden_state.float()
        n = int(mask.sum())
        out = out[:, :n]
        return torch.cat([out, out.new_zeros(1, 512 - n, out.shape[-1])], dim=1)

    pos = embed(prompt).half().cpu()
    neg = embed(NEGATIVE).half().cpu()
    torch.save({"pos": pos, "neg": neg, "seconds": round(time.time() - t0, 1)}, emb_path)
    drop_page_cache(MODEL, "text_encoder")
    print(f"prompt embedded in {time.time() - t0:.0f}s -> {emb_path}")


def sample(args, pos, neg, log, t0):
    import subprocess

    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel
    from PIL import Image

    # 2. Transformer + VAE, streamed through the card.
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    transformer = load_low_mem(WanTransformer3DModel, "transformer")
    # fp8 storage halves the 10 GB of weights in RAM; compute stays fp16.
    transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.float16)
    gc.collect()
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL, vae=vae, transformer=transformer, text_encoder=None, tokenizer=None, torch_dtype=torch.float16)
    pipe.transformer.enable_group_offload(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), offload_type="leaf_level", use_stream=True)
    pipe.vae.to("cuda")
    pipe.vae.enable_tiling()
    image = Image.open(args.start).convert("RGB").resize((args.width, args.height))
    gen = torch.Generator("cuda").manual_seed(args.seed)
    t1 = time.time()
    frames = pipe(image=image, prompt_embeds=pos.cuda(), negative_prompt_embeds=neg.cuda(), height=args.height, width=args.width,
                  num_frames=args.frames, num_inference_steps=args.steps, guidance_scale=args.guidance, generator=gen).frames[0]
    log["sample_seconds"] = round(time.time() - t1, 1)
    log["peak_vram_mb"] = torch.cuda.max_memory_allocated() // 2 ** 20
    # PNG frames beside the video (what detect.py --frames and the eye want), then the system ffmpeg.
    fdir = os.path.splitext(args.out)[0] + "_frames"
    os.makedirs(fdir, exist_ok=True)
    import numpy as np

    for i, fr in enumerate(frames):
        if not isinstance(fr, Image.Image):
            arr = np.asarray(fr)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            fr = Image.fromarray(arr)
        fr.save(os.path.join(fdir, f"f{i:05d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", os.path.join(fdir, "f%05d.png"),
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args.out], check=True)
    log["frames"] = len(frames)
    log["seconds"] = round(time.time() - t0, 1)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"video: {args.out} ({len(frames)} frames, {log['sample_seconds']}s sampling, peak {log['peak_vram_mb']} MB)")


if __name__ == "__main__":
    main()
