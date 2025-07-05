# serve JSON files from a directory as jobs with the CFA API endpoints

import json
import os
import sys

from flask import Flask, request, jsonify

app = Flask(__name__)
if not os.path.isfile("config.py"):
    sys.exit("'config.py' not found! Please add it and try again.")
else:
    import config

# Keep a list of processed jobs to avoid reprocessing
processed_jobs = set()

# Keep a list of client IDs to avoid duplicates
worker_ids = set()

@app.route('/api/init', methods=['GET'])
def init_worker():
    # Read the client ID from the request JSON body
    data = request.get_json()
    print("Login data:", data)

    if not data or 'worker_id' not in data:
        return jsonify({"error": "worker_id is required"}), 400

    worker_id = data['worker_id']
    created_worker = False
    if worker_id == 'N/A':
        # If worker_id is 'N/A', generate a new one
        worker_id = f"worker_{len(worker_ids) + 1}"
        created_worker = True
        worker_ids.add(worker_id)
        print(f"Generated new worker ID: {worker_id}")
    else:
        print(f"Using existing worker ID: {worker_id}")

    # Return the worker ID and a success message
    return jsonify({
        "worker_id": worker_id,
        "message": "Worker initialized successfully",
        "created": created_worker,
    }), 200

@app.route('/api/get-job', methods=['GET'])
def get_job():
    # Get a job from the queue folder
    queue_folder = config.queue_folder
    if not os.path.exists(queue_folder):
        print("Queue folder does not exist")
        return jsonify({"error": "Queue folder does not exist"}), 404

    files = [f for f in os.listdir(queue_folder) if f.endswith('.json')]
    if not files:
        print("No jobs available")
        return jsonify({"error": "No jobs available"}), 404

    # Sort files by modification time to get the oldest job first
    files.sort(key=lambda f: os.path.getmtime(os.path.join(queue_folder, f)))

    # Find the first unprocessed job
    for file in files:
        job_file = os.path.join(queue_folder, file)
        if job_file not in processed_jobs:
            break
    else:
        return jsonify({"error": "All jobs have been processed"}), 404

    with open(job_file, 'r') as f:
        job_data = json.load(f)

    # Record the job as processed
    processed_jobs.add(job_file)

    return jsonify(job_data)

@app.route('/api/upload', methods=['POST'])
def upload_images():
    channel = request.args.get('channel')
    if 'images' not in request.files:
        return jsonify({"error": "No 'images' field in request"}), 400

    files = request.files.getlist('images')
    saved_files = []
    i = 0
    for file in files:
        if file.filename:
            file.save(f'{config.upload_folder}/{channel}_{i}.png')
            saved_files.append(file.filename)
            i += 1

    return jsonify({"message": "Images uploaded", "files": saved_files})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)