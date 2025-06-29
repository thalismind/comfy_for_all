import os
import hashlib
import json
import glob

def load_hashes(checkpoint_db='checkpoint_db.json'):
    if os.path.exists(checkpoint_db):
        with open(checkpoint_db, 'r') as file:
            return json.load(file).get('hashes', [])
    return []

def hash_to_model_name(hash, hashes=None):
    hashes = hashes or load_hashes()
    for file_hash in hashes:
        if file_hash[0] == hash:
            print(file_hash[1])
            return(file_hash[1])

def save_hashes(hashes, checkpoint_db='checkpoint_db.json'):
    with open(checkpoint_db, 'w') as file:
        json.dump({'hashes': hashes}, file, indent=4)

def hash_file(filepath):
    print(f"Hashing file: {filepath}")

    # check if JSON file exists and contains sha256 key
    json_filepath = filepath.replace('.safetensors', '.metadata.json')
    if os.path.exists(json_filepath):
        print(f"Found metadata JSON for {filepath}, using existing sha256 if available.")
        with open(json_filepath, 'r') as json_file:
            data = json.load(json_file)
            if 'sha256' in data:
                return data['sha256']

    # otherwise, compute the hash of the file
    print(f"Computing SHA256 for {filepath}")
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def add_file_hash_if_new(filepath):
    hashes = load_hashes()

    filename = os.path.basename(filepath)

    # Check if hash already exists in any [hash, filename] entry
    if not any(entry[1] == filename for entry in hashes):
        file_hash = hash_file(filepath)
        hashes.append([file_hash, filename])
        save_hashes(hashes)
        print(f":white_check_mark: Added new hash for {filename}")
    else:
        print(f":warning: Hash for {filename} already exists.")

    return hashes

def hash_directory(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    # Find all .safetensors files in the directory using globs
    files = glob.glob(os.path.join(directory, '**', '*.safetensors'), recursive=True)
    if not files:
        print(f"No .safetensors files found in {directory}.")
        return

    print(f"Found {len(files)} .safetensors files in {directory}.")
    for filepath in files:
        hashes = add_file_hash_if_new(filepath)

    print(f"Total hashes saved: {len(hashes)}")
    return hashes
