'''
Downloads models from Hugging Face to models/model-name.

Example:
python download-model.py facebook/opt-1.3b

NOTE: this file is based on https://github.com/oobabooga/text-generation-webui/blob/main/download-model.py and heavily modified for:

- accelerate download speed by using aria2c
- include unusual file types (4bit, GGML etc.)
- misc fixes to adapt to environments

Will sync upstream from time to time

'''

import argparse
import base64
import datetime
import json
import re
import sys
from pathlib import Path
import subprocess

import requests
import tqdm
from tqdm.contrib.concurrent import thread_map
import os

# import aria2p

# aria2 = aria2p.API(
#     aria2p.Client(
#         host="http://IP",
#         port=0,
#         secret="TODO"
#     )
# )

parser = argparse.ArgumentParser()
parser.add_argument('MODEL', type=str, default=None, nargs='?')
parser.add_argument('--branch', type=str, default='main', help='Name of the Git branch to download from.')
parser.add_argument('--threads', type=int, default=8, help='Number of files to download simultaneously.')
parser.add_argument('--text-only', action='store_true', help='Only download text files (txt/json).')
parser.add_argument('--output', type=str, default=None, help='The folder where the model should be saved.')
args = parser.parse_args()

def get_file(url, output_folder):
    r = requests.get(url, stream=True)
    with open(output_folder / Path(url.rsplit('/', 1)[1]), 'wb') as f:
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024
        with tqdm.tqdm(total=total_size, unit='iB', unit_scale=True, bar_format='{l_bar}{bar}| {n_fmt:6}/{total_fmt:6} {rate_fmt:6}') as t:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)

def get_file_by_aria2(url, output_folder):
    filename = url.split('/')[-1]

    # r = requests.get(url, stream=True)
    # total_size = int(r.headers.get('content-length', 0))
    
    if (output_folder / Path(filename)).exists() and not (output_folder / Path(f"{filename}.aria2")).exists():
        print(f"Downloaded: {filename}")
        return

    # full_dir = f"{Path('/app/text-generation-webui/') / output_folder}"
    # print(f"aria2p downloading {url} to {full_dir} as {filename}")

    # aria2.add_uris(url, options={'dir': full_dir, 'out': filename})

    # /app/text-generation-webui/models

    aria_command = f"aria2c -c -x 16 -s 16 -k 1M {url} -d {output_folder} -o {filename}"

    print(f"Running: {aria_command}")
    # # call command line aria2c to download
    subprocess.run(aria_command, shell=True, check=True)

def sanitize_branch_name(branch_name):
    pattern = re.compile(r"^[a-zA-Z0-9._-]+$")
    if pattern.match(branch_name):
        return branch_name
    else:
        raise ValueError("Invalid branch name. Only alphanumeric characters, period, underscore and dash are allowed.")

def select_model_from_default_options():
    models = {
        "Pygmalion 6B original": ("PygmalionAI", "pygmalion-6b", "b8344bb4eb76a437797ad3b19420a13922aaabe1"),
        "Pygmalion 6B main": ("PygmalionAI", "pygmalion-6b", "main"),
        "Pygmalion 6B dev": ("PygmalionAI", "pygmalion-6b", "dev"),
        "Pygmalion 2.7B": ("PygmalionAI", "pygmalion-2.7b", "main"),
        "Pygmalion 1.3B": ("PygmalionAI", "pygmalion-1.3b", "main"),
        "Pygmalion 350m": ("PygmalionAI", "pygmalion-350m", "main"),
        "OPT 6.7b": ("facebook", "opt-6.7b", "main"),
        "OPT 2.7b": ("facebook", "opt-2.7b", "main"),
        "OPT 1.3b": ("facebook", "opt-1.3b", "main"),
        "OPT 350m": ("facebook", "opt-350m", "main"),
    }
    choices = {}

    print("Select the model that you want to download:\n")
    for i,name in enumerate(models):
        char = chr(ord('A')+i)
        choices[char] = name
        print(f"{char}) {name}")
    char = chr(ord('A')+len(models))
    print(f"{char}) None of the above")

    print()
    print("Input> ", end='')
    choice = input()[0].strip().upper()
    if choice == char:
        print("""\nThen type the name of your desired Hugging Face model in the format organization/name.

Examples:
PygmalionAI/pygmalion-6b
facebook/opt-1.3b
""")

        print("Input> ", end='')
        model = input()
        branch = "main"
    else:
        arr = models[choices[choice]]
        model = f"{arr[0]}/{arr[1]}"
        branch = arr[2]

    return model, branch

def get_download_links_from_huggingface(model, branch):
    base = "https://huggingface.co"
    page = f"/api/models/{model}/tree/{branch}"
    cursor = b""

    links = []
    sha256 = []
    classifications = []
    has_pytorch = False
    has_pt = False
    has_safetensors = False
    is_lora = False

    while True:
        api_url = f"{base}{page}"
        if cursor != b"":
                api_url = f"{api_url}?cursor={cursor.decode()}"
                
        # print(api_url)
        
        content = requests.get(api_url).content
        
        # print(content)

        dict = json.loads(content)
        if len(dict) == 0:
            break

        for i in range(len(dict)):
            fname = dict[i]['path']
            if not is_lora and fname.endswith(('adapter_config.json', 'adapter_model.bin')):
                is_lora = True

            is_pytorch = re.match("(pytorch|adapter)_model.*\.bin", fname)
            is_safetensors = re.match(".*\.safetensors", fname)
            is_pt = re.match(".*\.pt", fname)
            is_tokenizer = re.match("tokenizer.*\.model", fname)
            is_text = re.match(".*\.(txt|json|py|md)", fname) or is_tokenizer
            is_4bit = re.match("int4.*\.bin", fname)
            is_ggml = re.match(".*(ggml|GGML).*\.bin", fname)

            if any((is_pytorch, is_safetensors, is_pt, is_tokenizer, is_text, is_ggml, is_4bit)):
                if 'lfs' in dict[i]:
                    sha256.append([fname, dict[i]['lfs']['oid']])
                if is_text:
                    links.append(f"https://huggingface.co/{model}/resolve/{branch}/{fname}")
                    classifications.append('text')
                    continue
                if not args.text_only:
                    links.append(f"https://huggingface.co/{model}/resolve/{branch}/{fname}")
                    if is_safetensors:
                        has_safetensors = True
                        classifications.append('safetensors')
                    elif is_pytorch:
                        has_pytorch = True
                        classifications.append('pytorch')
                    elif is_pt:
                        has_pt = True
                        classifications.append('pt')
            else:
                print(f'Skipping: {fname}')

        cursor = base64.b64encode(f'{{"file_name":"{dict[-1]["path"]}"}}'.encode()) + b':50'
        cursor = base64.b64encode(cursor)
        cursor = cursor.replace(b'=', b'%3D')

    # If both pytorch and safetensors are available, download safetensors only
    if (has_pytorch or has_pt) and has_safetensors:
        for i in range(len(classifications)-1, -1, -1):
            if classifications[i] in ['pytorch', 'pt']:
                links.pop(i)

    return links, sha256, is_lora

def download_files(file_list, output_folder, num_threads=8):
    thread_map(lambda url: get_file_by_aria2(url, output_folder), file_list, max_workers=num_threads)

if __name__ == '__main__':
    # determine the root of the repo and cd to it
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    os.chdir(ROOT)
    print(f"Working directory changed to: {ROOT}")

    model = args.MODEL
    branch = args.branch
    if model is None:
        model, branch = select_model_from_default_options()
    else:
        if model[-1] == '/':
            model = model[:-1]
            branch = args.branch
        if branch is None:
            branch = "main"
        else:
            try:
                branch = sanitize_branch_name(branch)
            except ValueError as err_branch:
                print(f"Error: {err_branch}")
                sys.exit()

    links, sha256, is_lora = get_download_links_from_huggingface(model, branch)

    if args.output is not None:
        base_folder = args.output
    else:
        base_folder = 'models' if not is_lora else 'loras'

    output_folder = f"{'_'.join(model.split('/')[-2:])}"
    if branch != 'main':
        output_folder += f'_{branch}'

    # Creating the folder and writing the metadata
    output_folder = Path(base_folder) / output_folder
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    with open(output_folder / 'huggingface-metadata.txt', 'w') as f:
        f.write(f'url: https://huggingface.co/{model}\n')
        f.write(f'branch: {branch}\n')
        f.write(f'download date: {str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}\n')
        sha256_str = ''
        for i in range(len(sha256)):
            sha256_str += f'    {sha256[i][1]} {sha256[i][0]}\n'
        if sha256_str != '':
            f.write(f'sha256sum:\n{sha256_str}')

    # Downloading the files
    print(f"Downloading the model to {output_folder}")
    download_files(links, output_folder, args.threads)
    print()
