#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import io
import argparse
import requests
import time

from gpu_nvidia import GPUIdleTimer
from hashes import hash_directory, hash_to_model_name
from models import ImageJob

def queue_prompt(args, prompt):
    p = {"prompt": prompt, "client_id": args.client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(args.comfy_server), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(args, filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(args.comfy_server, url_values)) as response:
        return response.read()

def get_history(args, prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(args.comfy_server, prompt_id)) as response:
        return json.loads(response.read())

def get_images(args, ws, prompt):
    prompt_id = queue_prompt(args, prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            # If you want to be able to decode the binary stream for latent previews, here is how you can do it:
            # bytesIO = BytesIO(out[8:])
            # preview_image = Image.open(bytesIO) # This is your preview in PIL image format, store it in a global
            continue #previews are binary data

    history = get_history(args, prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(args, image['filename'], image['subfolder'], image['type'])
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

def parse_size(size_str):
    """Parse a size string in the format 'widthxheight'."""
    try:
        width, height = map(int, size_str.split('x'))
        return width, height
    except ValueError:
        raise ValueError("Size must be in the format 'widthxheight', e.g., '512x512'.")

def generate_prompt(job: ImageJob, hashes: list[tuple[str, str]]):
  width, height = parse_size(job.size)
  model_name = hash_to_model_name(job.model, hashes)
  print(f"Using model {model_name} for job {job.id} with {job.batch_size} images of size {width}x{height}.")

  prompt = {
      "3": {
          "class_type": "KSampler",
          "inputs": {
              "cfg": job.cfg,
              "denoise": 1,
              "latent_image": [
                  "5",
                  0
              ],
              "model": [
                  "4",
                  0
              ],
              "negative": [
                  "7",
                  0
              ],
              "positive": [
                  "6",
                  0
              ],
              "sampler_name": "euler",
              "scheduler": "normal",
              "seed": job.seed,
              "steps": job.steps,
          }
      },
      "4": {
          "class_type": "CheckpointLoaderSimple",
          "inputs": {
              "ckpt_name": model_name,
          }
      },
      "5": {
          "class_type": "EmptyLatentImage",
          "inputs": {
              "batch_size": job.batch_size,
              "height": height,
              "width": width,
          }
      },
      "6": {
          "class_type": "CLIPTextEncode",
          "inputs": {
              "clip": [
                  "4",
                  1
              ],
              "text": job.prompt,
          }
      },
      "7": {
          "class_type": "CLIPTextEncode",
          "inputs": {
              "clip": [
                  "4",
                  1
              ],
              "text": job.negative_prompt,
          }
      },
      "8": {
          "class_type": "VAEDecode",
          "inputs": {
              "samples": [
                  "3",
                  0
              ],
              "vae": [
                  "4",
                  2
              ]
          }
      },
      "9": {
          "class_type": "SaveImage",
          "inputs": {
              "filename_prefix": "ComfyUI",
              "images": [
                  "8",
                  0
              ]
          }
      }
  }
  return prompt

def run_job(args, job, hashes):
  prompt = generate_prompt(job, hashes)

  ws = websocket.WebSocket()
  ws.connect("ws://{}/ws?clientId={}".format(args.comfy_server, args.client_id))
  images = get_images(args, ws, prompt)
  ws.close()

  output_images = []
  for node_id in images:
    for image_data in images[node_id]:
      image = Image.open(io.BytesIO(image_data))
      output_images.append(image)

  return output_images

def parse_args():
    parser = argparse.ArgumentParser(description="Run a job on ComfyUI server.")
    parser.add_argument('--job_server', type=str, default='http://127.0.0.1:5000', help='Job server address')
    parser.add_argument('--comfy_server', type=str, default='127.0.0.1:8188', help='ComfyUI server address')
    parser.add_argument('--client_id', type=str, default=str(uuid.uuid4()), help='Client ID for the WebSocket connection')
    parser.add_argument('--single_job', action='store_true', help='Run a single job and exit')
    parser.add_argument('--polling_interval', type=int, default=30, help='Polling interval in seconds for job fetching')
    parser.add_argument('--gpu_index', type=int, default=0, help='GPU index to monitor for idle state')
    parser.add_argument('--idle_threshold', type=int, default=900, help='Idle threshold in seconds for GPU')
    parser.add_argument('--safetensors_dir', type=str, default='safetensors', help='Directory to store safetensors files')
    return parser.parse_args()

def get_job(args):
    response = requests.get(f"{args.job_server}/api/get-job")
    if response.status_code == 200:
        return response.json()
    else:
        print("No jobs available or error fetching job.")
        return None

def upload_images(args, images, channel):
    url = f"{args.job_server}/api/upload"
    files = [('images', (f'image_{i}.png', io.BytesIO(image.tobytes()), 'image/png')) for i, image in enumerate(images)]
    response = requests.post(url, files=files, params={'channel': channel})
    if response.status_code == 200:
        print("Images uploaded successfully.")
    else:
        print(f"Failed to upload images: {response.status_code} - {response.text}")

def job_loop(args):
    idle_timer = GPUIdleTimer(gpu_index=args.gpu_index, idle_threshold=args.idle_threshold)
    idle_timer.load_nvml()  # Initialize NVML for GPU monitoring

    hashes = hash_directory(args.safetensors_dir)  # Load or hash safetensors files

    while True:
        # Increment the idle timer and check if the GPU is idle
        idle_timer.increment_timer()
        if not idle_timer.has_reached_idle_threshold():
            print(f"GPU {idle_timer.gpu_index} has only been idle for {idle_timer.idle_time:0.2f} seconds, waiting.")
            time.sleep(args.polling_interval)
            continue

        # Get the next job from the server
        print(f"GPU {idle_timer.gpu_index} has been idle for {idle_timer.idle_time:0.2f} seconds, getting next job.")
        job_data = get_job(args)
        if not job_data:
            print("No jobs available, waiting...")
            time.sleep(args.polling_interval)
            continue

        job = ImageJob(
            id=job_data[0],
            created_at=job_data[1],
            updated_at=job_data[1], # what about 2?
            job_type=job_data[3],
            user=job_data[4],
            prompt=job_data[5],
            negative_prompt=job_data[6],
            model=job_data[7],
            steps=job_data[8],
            seed=job_data[9],
            # not sure what 10 is
            size=job_data[11],
            batch_size=job_data[12],
            cfg=job_data[13]
        )
        print(f"Processing job: {job.id} with prompt: {job.prompt}")
        images = run_job(args, job, hashes)
        print(f"Job {job.id} completed with {len(images)} images.")
        if not images:
            print("No images generated, skipping upload.")
            continue

        upload_images(args, images, job.user)  # Upload images to the server


if __name__ == "__main__":
    args = parse_args()
    if args.single_job:
      job = ImageJob(
          id=str(uuid.uuid4()),
          created_at="2023-10-01T00:00:00Z",
          updated_at="2023-10-01T00:00:00Z",
          job_type="generate",
          user=1,
          prompt="A beautiful landscape",
          negative_prompt="bad hands, blurry, low quality",
          model="pony/realcartoon-pony.safetensors",
          steps=50,
          seed=42,
          size="512x512",
          batch_size=1,
          cfg=7.5
      )
      run_job(job)
    else:
        job_loop(args)
