import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import io
import time
import random

from gpu_nvidia import GPUIdleTimer
from hashes import hash_directory, hash_to_model_name
from models import ImageJob
from worker_base import get_job, upload_images, BaseWorkerArgs, login, base_parser


class ComfyWorkerArgs(BaseWorkerArgs):
    comfy_id: str
    comfy_server: str

    def __init__(self, comfy_id=uuid.uuid4().hex, comfy_server="", **kwargs):
        super().__init__(**kwargs)
        self.comfy_id = comfy_id
        self.comfy_server = comfy_server


def queue_prompt(args: ComfyWorkerArgs, prompt):
    p = {"prompt": prompt, "client_id": args.comfy_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(
        "http://{}/prompt".format(args.comfy_server), data=data
    )
    return json.loads(urllib.request.urlopen(req).read())


def get_image(args: ComfyWorkerArgs, filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    complete_url = "http://{}/view?{}".format(args.comfy_server, url_values)
    print(f"Fetching image from: {complete_url}")

    with urllib.request.urlopen(complete_url) as response:
        return response.read()


def get_history(args: ComfyWorkerArgs, prompt_id):
    with urllib.request.urlopen(
        "http://{}/history/{}".format(args.comfy_server, prompt_id)
    ) as response:
        return json.loads(response.read())


def get_images(args: ComfyWorkerArgs, ws, prompt):
    prompt_id = queue_prompt(args, prompt)["prompt_id"]
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message["type"] == "executing":
                data = message["data"]
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    break  # Execution is done
        else:
            # If you want to be able to decode the binary stream for latent previews, here is how you can do it:
            # bytesIO = BytesIO(out[8:])
            # preview_image = Image.open(bytesIO) # This is your preview in PIL image format, store it in a global
            continue  # Previews are binary data

    history = get_history(args, prompt_id)[prompt_id]
    for node_id in history["outputs"]:
        node_output = history["outputs"][node_id]
        images_output = []
        if "images" in node_output:
            for image in node_output["images"]:
                image_data = get_image(
                    args, image["filename"], image["subfolder"], image["type"]
                )
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images


def parse_size(size_str):
    """Parse a size string in the format 'widthxheight'."""
    try:
        width, height = map(int, size_str.split("x"))
        return width, height
    except ValueError:
        raise ValueError("Size must be in the format 'widthxheight', e.g., '512x512'.")


def generate_prompt(job: ImageJob, hashes: list[tuple[str, str]]):
    width, height = parse_size(job.resolution)
    model_name = hash_to_model_name(job.model, hashes)
    seed = random.randint(0, 2**32 - 1)
    print(
        f"Using model {model_name} for job {job.id} with {job.batch_size} images of size {width}x{height} and seed {seed}."
    )

    prompt = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": job.config_scale,
                "denoise": 1,
                "latent_image": ["5", 0],
                "model": ["4", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": seed,
                "steps": job.steps,
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": model_name,
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "batch_size": job.batch_size,
                "height": height,
                "width": width,
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": job.requested_prompt,
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": job.negative_prompt,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
        },
    }
    return prompt


def run_job(args: ComfyWorkerArgs, job, hashes):
    prompt = generate_prompt(job, hashes)

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(args.comfy_server, args.comfy_id))
    images = get_images(args, ws, prompt)
    ws.close()

    output_images = []
    for node_id in images:
        print(
            f"Processing images for node {node_id} with {len(images[node_id])} images."
        )
        for image_data in images[node_id]:
            image = Image.open(io.BytesIO(image_data))
            print(f"Image size: {image.size}, mode: {image.mode}")
            output_images.append(image)

    return output_images


def parse_args() -> ComfyWorkerArgs:
    parser = base_parser()
    parser.add_argument(
        "--comfy_id",
        type=str,
        default=uuid.uuid4().hex,
        help="Unique ID for the ComfyUI worker",
    )
    parser.add_argument(
        "--comfy_server",
        type=str,
        default="127.0.0.1:8188",
        help="ComfyUI server address",
    )
    return parser.parse_args(namespace=ComfyWorkerArgs)


def job_loop(args: ComfyWorkerArgs):
    client = login(args)

    idle_timer = GPUIdleTimer(
        gpu_index=args.gpu_index, idle_threshold=args.idle_threshold
    )
    idle_timer.load_nvml()  # Initialize NVML for GPU monitoring

    hashes = hash_directory(args.checkpoint_dir)  # Load or hash safetensors files

    while True:
        # Increment the idle timer and check if the GPU is idle
        idle_timer.increment_timer()
        if not idle_timer.has_reached_idle_threshold():
            print(
                f"GPU {idle_timer.gpu_index} has only been idle for {idle_timer.idle_time:0.2f} seconds, waiting."
            )
            time.sleep(args.polling_interval)
            continue

        # Get the next job from the server
        print(
            f"GPU {idle_timer.gpu_index} has been idle for {idle_timer.idle_time:0.2f} seconds, getting next job."
        )
        job = get_job(args, client, hashes)
        if not job:
            print("No jobs available, waiting...")
            time.sleep(args.polling_interval)
            continue

        # Create an ImageJob instance from the job data
        print(f"Processing job: {job.id} with prompt: {job.requested_prompt}")
        images = run_job(args, job, hashes)
        print(f"Job {job.id} completed with {len(images)} images.")
        if not images:
            print("No images generated, skipping upload.")
            continue

        upload_images(args, client, images, job)  # Upload images to the server


if __name__ == "__main__":
    args = parse_args()
    if args.single_job:
        job = ImageJob(
            id=str(uuid.uuid4()),
            requested_at="2023-10-01T00:00:00Z",
            started_at="2023-10-01T00:00:00Z",
            request_type="generate",
            requested_prompt="A beautiful landscape",
            negative_prompt="bad hands, blurry, low quality",
            model="pony/realcartoon-pony.safetensors",
            steps=50,
            resolution="512x512",
            batch_size=1,
            config_scale=7.5,
        )
        run_job(job)
    else:
        job_loop(args)
