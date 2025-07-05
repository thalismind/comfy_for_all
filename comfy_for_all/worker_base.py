import requests
import io
import argparse
import json
from pydantic import BaseModel

from models import ImageJob

DEFAULT_CHECKPOINT_DIR = 'checkpoints'
DEFAULT_CLIENT_FILE = 'client.json'
DEFAULT_GPU_INDEX = 0
DEFAULT_IDLE_THRESHOLD = 900  # seconds
DEFAULT_JOB_SERVER = 'http://127.0.0.1:5000'
DEFAULT_LORA_DIR = 'loras'
DEFAULT_POLLING_INTERVAL = 30
DEFAULT_SINGLE_JOB = False

class BaseWorkerArgs(argparse.Namespace):
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    client_file: str = DEFAULT_CLIENT_FILE
    gpu_index: int = DEFAULT_GPU_INDEX
    idle_threshold: int = DEFAULT_IDLE_THRESHOLD
    job_server: str = DEFAULT_JOB_SERVER
    lora_dir: str = DEFAULT_LORA_DIR
    polling_interval: int = DEFAULT_POLLING_INTERVAL
    single_job: bool = DEFAULT_SINGLE_JOB

    def __init__(
            self,
            checkpoint_dir: str,
            client_file: str,
            gpu_index: int,
            idle_threshold: int,
            job_server: str,
            lora_dir: str,
            polling_interval: int,
            single_job: bool,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.client_file = client_file
        self.gpu_index = gpu_index
        self.idle_threshold = idle_threshold
        self.job_server = job_server
        self.lora_dir = lora_dir
        self.polling_interval = polling_interval
        self.single_job = single_job

class BaseWorkerFile(BaseModel):
    worker_id: str

DEFAULT_WORKER_CLIENT_FILE = BaseWorkerFile(worker_id='N/A')

HashType = list[tuple[str, str]]

def get_job(args: BaseWorkerArgs, client: BaseWorkerFile, hashes: HashType) -> dict | None:
    # print(args, client, hashes)
    print("Fetching job from server as worker:", client.worker_id)

    data = {
        "checkpoints": [hash[0] for hash in hashes],
        "worker_id": client.worker_id,
    }
    response = requests.get(f"{args.job_server}/api/get-job", json=data)
    if response.status_code == 200:
        job_data = response.json()
        print("Received job data:", job_data)
        job = ImageJob(
            id=job_data['job_id'],
            requested_at=job_data['requested_at'],
            started_at=job_data['started_at'],
            request_type=job_data['request_type'],
            # requester=None,
            requested_prompt=job_data['requested_prompt'],
            negative_prompt=job_data['negative_prompt'],
            model=job_data['model'],
            steps=job_data['steps'],
            channel=job_data['channel'],
            image_link=job_data['image_link'],
            resolution=job_data['resolution'],
            batch_size=job_data['batch_size'],
            config_scale=job_data['config_scale']
        )
        return job
    else:
        print("No jobs available or error fetching job.")
        return None

def upload_images(args: BaseWorkerArgs, client: BaseWorkerFile, images, job: ImageJob):
    url = f"{args.job_server}/api/upload"
    files = []
    for i, image in enumerate(images):
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)  # Reset the stream position to the beginning
        files.append(("images", (f"image_{i}.png", image_bytes, "image/png")))

    response = requests.post(url, data={'worker_id': client.worker_id}, files=files, params={'channel': job.channel, 'job_id': job.id})
    if response.status_code == 200:
        print("Images uploaded successfully.")
    else:
        print(f"Failed to upload images: {response.status_code} - {response.text}")

def login(args: BaseWorkerArgs) -> BaseWorkerFile:
    # Check if client file exists and load it if it does
    client_data = DEFAULT_WORKER_CLIENT_FILE
    new_client = True

    try:
        with open(args.client_file, 'r') as f:
            loaded_data = json.load(f)
            client_data = BaseWorkerFile(**loaded_data)
            new_client = False
            print("Loaded existing client data:", client_data)
    except FileNotFoundError:
        print("Client file not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON from client file.")

    # Login and update the client file
    response = requests.get(f"{args.job_server}/api/init", json=client_data.model_dump())
    if response.status_code == 200:
        updated_data = response.json()
        client_data = BaseWorkerFile(**updated_data)
        with open(args.client_file, 'w') as f:
            json.dump(client_data.model_dump(), f)

        if new_client and "created" in updated_data:
            print(f"New client created with ID: {client_data.worker_id}")
        elif not new_client and "created" not in updated_data:
            print(f"Logged in with existing client ID: {client_data.worker_id}")
        else:
            print(f"Something weird happened, client data: {updated_data}")
    else:
        print("Failed to login.")

    return client_data

def base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Base worker arguments")
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR, help='Directory for checkpoints')
    parser.add_argument('--client_file', type=str, default=DEFAULT_CLIENT_FILE, help='File to store client data')
    parser.add_argument('--gpu_index', type=int, default=DEFAULT_GPU_INDEX, help='GPU index to monitor for idle state')
    parser.add_argument('--idle_threshold', type=int, default=DEFAULT_IDLE_THRESHOLD, help='Idle threshold in seconds for GPU')
    parser.add_argument('--job_server', type=str, default=DEFAULT_JOB_SERVER, help='Job server address')
    parser.add_argument('--lora_dir', type=str, default=DEFAULT_LORA_DIR, help='Directory for LoRA files')
    parser.add_argument('--polling_interval', type=int, default=DEFAULT_POLLING_INTERVAL, help='Polling interval in seconds')
    parser.add_argument('--single_job', action='store_true', help='Process a single job and exit')
    return parser
