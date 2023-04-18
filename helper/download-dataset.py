'''
Downloads datasets from Hugging Face to datasets/dataset-name.

Example:
python download-dataset.py tatsu-lab/alpaca

NOTE: this file is based on `download-model.py` and heavily modified

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

parser = argparse.ArgumentParser()
parser.add_argument('DATASET', type=str, default=None, nargs='?')
parser.add_argument('--branch', type=str, default='main', help='Name of the Git branch to download from.')
parser.add_argument('--threads', type=int, default=8, help='Number of files to download simultaneously.')
parser.add_argument('--output', type=str, default=None, help='The folder where the dataset should be saved.')
args = parser.parse_args()

def get_file(url_info, output_folder):
    url = url_info['link']
    filename = url_info['path']
    dir_and_basename = filename.rsplit('/', 1)
    dir = dir_and_basename[0] if len(dir_and_basename) == 2 else ''

    if not (output_folder / dir).exists():
        (output_folder / dir).mkdir(parents=True, exist_ok=True)

    total_size = url_info['size']

    output_file = output_folder / Path(filename)

    if output_file.exists() and output_file.stat().st_size == total_size:
        print(f"Downloaded: {output_file}")
        return

    with open(output_folder / filename, 'wb') as f:
        r = requests.get(url, stream=True)
        block_size = 1024
        with tqdm.tqdm(postfix=f'File: {filename}', total=total_size, unit='iB', unit_scale=True, bar_format='{l_bar}{bar}| {n_fmt:6}/{total_fmt:6} {rate_fmt:6} {postfix}') as t:
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

def get_download_links_from_huggingface(dataset, branch, dir=None):
    base = "https://huggingface.co"
    page = f"/api/datasets/{dataset}/tree/{branch}"
    if dir is not None:
        page = f'{page}/{dir}'

    cursor = b""

    links = []
    sha256 = []

    # TODO support passing in
    token = os.environ.get("HUGGINGFACE_TOKEN")
    headers = { "authorization" : f'Bearer {token}'}

    while True:
        api_url = f"{base}{page}?cursor={cursor.decode()}"
        # print(api_url)
        content = requests.get(api_url, headers=headers).content
        # print(content)

        dict = json.loads(content)

        if 'error' in dict:
            # print(f'Ignoring error: {dict["error"]}')
            break

        if len(dict) == 0:
            break

        for i in range(len(dict)):
            item = dict[i]

            fname = item['path']

            # print(f'{item}')

            if item['type'] == 'directory':
                print(f'Gathering: {fname}')

                dir_links, dir_sha256 = get_download_links_from_huggingface(dataset, branch, dir=fname)

                links.extend(dir_links)                
                # TODO: also add dir_sha256, fix fname in the process

            if item['type'] == 'file':
                link = f"https://huggingface.co/datasets/{dataset}/resolve/{branch}/{fname}"
                # print(f'{link}')

                url_info = {"link": link, "path": fname, "size": item["size"]}

                if 'lfs' in item:
                    oid = item['lfs']['oid']
                    sha256.append([fname, oid])
                    url_info["sha256sum"] = oid

                links.append(url_info)

        cursor_file = dict[-1]["path"]
        # print(f'Using file {cursor_file} as cursor')
        curson_json = f'{{"file_name":"{cursor_file}"}}'
        cursor = base64.b64encode(curson_json.encode()) + b':50'
        cursor = base64.b64encode(cursor)
        cursor = cursor.replace(b'=', b'%3D')

    return links, sha256

def download_files(file_list, output_folder, num_threads=8):
    thread_map(lambda url_info: get_file(url_info, output_folder), file_list, max_workers=num_threads)

if __name__ == '__main__':
    # determine the root of the repo and cd to it
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    os.chdir(ROOT)
    print(f"Working directory changed to: {ROOT}")

    dataset = args.DATASET
    branch = args.branch
    if dataset is None:
        print("Error: Please specify a dataset to download.")
        sys.exit()
    else:
        if dataset[-1] == '/':
            dataset = dataset[:-1]
            branch = args.branch
        if branch is None:
            branch = "main"
        else:
            try:
                branch = sanitize_branch_name(branch)
            except ValueError as err_branch:
                print(f"Error: {err_branch}")
                sys.exit()

    if args.output is not None:
        base_folder = args.output
    else:
        base_folder = 'datasets'

    output_folder = f"{'_'.join(dataset.split('/')[-2:])}"
    if branch != 'main':
        output_folder += f'_{branch}'

    # Creating the folder and writing the metadata
    output_folder = Path(base_folder) / output_folder
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    links_file = output_folder / 'links.json'

    links, sha256 = [], []

    # TODO expire logic
    if links_file.exists():    
        with open(links_file, 'r') as f:
            links = json.load(f)
            # TODO fix sha256
    else:
        links, sha256 = get_download_links_from_huggingface(dataset, branch)
        with open(links_file, 'w') as f:
            json.dump(links, f)

    with open(output_folder / 'huggingface-metadata.txt', 'w') as f:
        f.write(f'url: https://huggingface.co/{dataset}\n')
        f.write(f'branch: {branch}\n')
        f.write(f'download date: {str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}\n')
        sha256_str = ''
        for i in range(len(sha256)):
            sha256_str += f'    {sha256[i][1]} {sha256[i][0]}\n'
        if sha256_str != '':
            f.write(f'sha256sum:\n{sha256_str}')

    # Downloading the files
    print(f"Downloading the dataset to {output_folder}")
    download_files(links, output_folder, args.threads)
    print()


