#!/usr/bin/env python3
import argparse, time
import torch
from PIL import Image
from transformers import AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="Describe this image in one short sentence.")
    ap.add_argument("--model", default="vikhyatk/moondream2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=dtype).to(device).eval()
    if device == "cuda":
        torch.cuda.synchronize()
    t_load = time.perf_counter() - t0

    img = Image.open(args.image).convert("RGB")
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    out = model.query(image=img, question=args.prompt)
    if device == "cuda":
        torch.cuda.synchronize()
    t_inf = time.perf_counter() - t1

    print(f"device={device} dtype={dtype}")
    print(f"load_s={t_load:.3f} infer_s={t_inf:.3f}")
    print(out.get('answer', out))


if __name__ == "__main__":
    main()


