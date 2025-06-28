# serve JSON files from a directory as jobs with the CFA API endpoints

import json
import os
import sys

from flask import Flask, request, jsonify

app = Flask(__name__)
if not os.path.isfile("config.py"):
    sys.exit("'config.py' not found! Please add it and try again.")
else:
    import comfy_for_all.config as config

# Keep a list of processed jobs to avoid reprocessing
processed_jobs = set()

@app.route('/api/get-job', methods=['GET'])
def get_job():
    # Get a job from the queue folder
    queue_folder = config.queue_folder
    if not os.path.exists(queue_folder):
        return jsonify({"error": "Queue folder does not exist"}), 404

    files = [f for f in os.listdir(queue_folder) if f.endswith('.json')]
    if not files:
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
            file.save(f'uploads/{channel}_{i}.png')
            saved_files.append(file.filename)
            i += 1

    return jsonify({"message": "Images uploaded", "files": saved_files})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)