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
    args = ap.parse_args()
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel
    from diffusers.utils import export_to_video
    from PIL import Image
    from transformers import AutoTokenizer, UMT5EncoderModel

    t0 = time.time()
    prompt = PROMPT.format(creature=args.creature, move=args.move)
    log = {"model": MODEL, "license": LICENSE, "prompt": prompt, "negative": NEGATIVE, "args": vars(args),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}

    # 1. Text first, then free the 11 GB encoder.
    tok = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=torch.float16)
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

    pos = embed(prompt).half()
    neg = embed(NEGATIVE).half()
    del enc
    gc.collect()
    torch.cuda.empty_cache()
    log["text_seconds"] = round(time.time() - t0, 1)

    # 2. Transformer + VAE, streamed through the card.
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    transformer = WanTransformer3DModel.from_pretrained(MODEL, subfolder="transformer", torch_dtype=torch.float16)
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
    export_to_video(frames, args.out, fps=args.fps)
    log["frames"] = len(frames)
    log["seconds"] = round(time.time() - t0, 1)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"video: {args.out} ({len(frames)} frames, {log['sample_seconds']}s sampling, peak {log['peak_vram_mb']} MB)")


if __name__ == "__main__":
    main()
