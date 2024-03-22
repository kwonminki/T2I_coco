from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import argparse
import pandas as pd
import os





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COCO Sampling')
    parser.add_argument('--model', type=str, default='sd15', help='t2i model')
    parser.add_argument('--coco_path', type=str, default='coco', help='Path to COCO dataset')
    parser.add_argument('--sampling_steps', type=int, default=50, help='Number of sampling steps')
    parser.add_argument('--cfg', type=float, default=7.5, help='classifier free guidance coefficient')
    
    args = parser.parse_args()

    model_dict = {
        "sd14": "CompVis/stable-diffusion-v1-4",
        "sd15": "runwayml/stable-diffusion-v1-5",
        "sd20": "stabilityai/stable-diffusion-2",
        "sd21": "stabilityai/stable-diffusion-2-1",
        "sdxl10": "stabilityai/stable-diffusion-xl-base-1.0",
        "sdxlturbo": "stabilityai/sdxl-turbo",
    }

    if args.model in model_dict:
        model = model_dict[args.model]
    
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    df_sample = pd.read_parquet('parquet/subset_coco.parquet')
    output_dir = os.path.join(args.coco_path, model.split('/')[-1])
    output_folder = f"sample_{args.sampling_steps}_cfg_{args.cfg}"
    output_path = os.path.join(output_dir, output_folder)
    os.makedirs(output_path, exist_ok=True)

    for i, row in df_sample.iterrows():
        caption = row['caption']
        output_file = os.path.join(output_path, f"{i}.png")
        if not os.path.exists(output_file):
            image = pipe(caption, num_inference_steps=args.sampling_steps, guidance_scale=args.cfg)
        