#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import io

server_address = "10.2.2.81:8188"
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
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

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

def generate_prompt():
  prompt = {
      "3": {
          "class_type": "KSampler",
          "inputs": {
              "cfg": 8,
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
              "seed": 8566257,
              "steps": 20
          }
      },
      "4": {
          "class_type": "CheckpointLoaderSimple",
          "inputs": {
              "ckpt_name": "pony/realcartoon-pony.safetensors"
          }
      },
      "5": {
          "class_type": "EmptyLatentImage",
          "inputs": {
              "batch_size": 1,
              "height": 512,
              "width": 512
          }
      },
      "6": {
          "class_type": "CLIPTextEncode",
          "inputs": {
              "clip": [
                  "4",
                  1
              ],
              "text": "masterpiece best quality girl"
          }
      },
      "7": {
          "class_type": "CLIPTextEncode",
          "inputs": {
              "clip": [
                  "4",
                  1
              ],
              "text": "bad hands"
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

def run_job(job):
  prompt = generate_prompt()
  #set the text prompt for our positive CLIPTextEncode
  prompt["6"]["inputs"]["text"] = "masterpiece best quality man"

  #set the seed for our KSampler node
  prompt["3"]["inputs"]["seed"] = 5

  ws = websocket.WebSocket()
  ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
  images = get_images(ws, prompt)
  ws.close()

  for node_id in images:
    for image_data in images[node_id]:
      image = Image.open(io.BytesIO(image_data))
      image.show()

if __name__ == "__main__":
    job = None  # Replace with actual job data if needed
    run_job(job)
