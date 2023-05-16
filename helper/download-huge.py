# Tested with the following commands
#
# 1. One complete run
#
# > python helper/download-huge.py bigcode/the-stack 'data/arc/*' 'data/cmake/*'|grep -v Redirecting
# Resume downloading of 0 LFS files
# Found 4 LFS files to download from 4 matched files
#
# 2. Interupt and resume downloading
#
# Resume downloading of 4 LFS files
# Found 1 LFS files to download from 4 matched files
#
# 3. Remove a file and resume downloading
#
# > rm datasets/bigcode_the-stack/data/cmake/train-00002-of-00003.parquet
# > python helper/download-huge.py bigcode/the-stack 'data/arc/*' 'data/cmake/*'|grep -v Redirecting
#
# Resume downloading of 5 LFS files
# Found 0 LFS files to download from 3 matched files


from huggingface_hub import Repository, repository
import argparse
import os
import re
from pathlib import Path
import subprocess
import json
import tqdm
from tqdm.contrib.concurrent import thread_map


parser = argparse.ArgumentParser()
parser.add_argument('REPO', type=str, default=None, nargs='?')
parser.add_argument('PATHS', type=str, default=None, nargs='*', help='Relative glob pattern(s) to download the dataset/model from')
parser.add_argument('-m', '--model', action='store_true', help='By default this script downloads datasets, pass -d to download models instead')
parser.add_argument('--branch', type=str, default='main', help='Name of the Git branch to download from')
parser.add_argument('--output', type=str, default=None, help='The folder where the dataset/model should be saved.')
parser.add_argument('--threads', type=int, default=8, help='Number of files to download simultaneously.')
args = parser.parse_args()

def sanitize_branch_name(branch_name):
    pattern = re.compile(r"^[a-zA-Z0-9._-]+$")
    if pattern.match(branch_name):
        return branch_name
    else:
        raise ValueError("Invalid branch name. Only alphanumeric characters, period, underscore and dash are allowed.")

def get_file_by_aria2(url_info, output_folder):
    url = url_info['link']
    filename = url_info['path']
    dir_and_basename = filename.rsplit('/', 1)
    dir = dir_and_basename[0] if len(dir_and_basename) == 2 else ''

    if not (output_folder / dir).exists():
        (output_folder / dir).mkdir(parents=True, exist_ok=True)

    total_size = url_info['size']

    output_file = output_folder / Path(filename)

    if output_file.exists() and output_file.stat().st_size == total_size and not (output_folder / Path(f"{filename}.aria2")).exists():
        print(f"Downloaded: {output_file}")
        return

    aria_command = f"aria2c -c -x 16 -s 16 -k 1M {url} -d {output_folder} -o {filename}"

    print(f"Running: {aria_command}")

    token = os.environ.get("HUGGINGFACE_TOKEN")
    aria_command = f"{aria_command} --header='authorization: Bearer {token}'"
    # # call command line aria2c to download
    subprocess.run(aria_command, shell=True, check=True)

def download_files(file_list, output_folder, num_threads=8):
    thread_map(lambda url_info: get_file_by_aria2(url_info, output_folder), file_list, max_workers=num_threads)

def parse_lfs_file(file, root, repo_type, repo, branch):
    file = Path(file)
    is_lfs = False
    oid = None
    size = None
    base_url = "https://huggingface.co/datasets/" if repo_type == 'dataset' else "https://huggingface.co/"
    fname = file.relative_to(root)

    url_info = None

    if not file.is_dir() and file.stat().st_size < 2048:
        with open(file, 'r', encoding='latin1') as f:
            lines = f.readlines(5)
            for line in lines:
                if line.startswith('version https://git-lfs.github.com/spec/'):
                    url_info = {}
                elif line.startswith('oid sha256:'):
                    url_info["sha256sum"] = line.split('sha256:')[1].strip()
                elif line.startswith('size '):
                    url_info["path"] = f"{fname}"
                    url_info["size"] = int(line.split('size ')[1].strip())
                    url_info["link"] = f"{base_url}{repo}/resolve/{branch}/{fname}"
    return url_info

def get_download_links(paths, output_folder, repo_type, remote_repo, branch):
    all_files = []
    lfs_files = []
    for path_pattern in paths:
        files = output_folder.glob(path_pattern)
        for file in files:
            fname = f'{file.relative_to(output_folder)}'
            if not fname.endswith('.aria2'):
                all_files.append(fname)
            url_info = parse_lfs_file(file, output_folder, repo_type, remote_repo, branch)
            if url_info is not None:
                lfs_files.append(url_info)
    return lfs_files, all_files

if __name__ == '__main__':
    # determine the root of the repo and cd to it
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    os.chdir(ROOT)
    print(f"Working directory changed to: {ROOT}")

    HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

    remote_repo = args.REPO
    branch = args.branch

    if remote_repo is None:
        print("Error: Please specify a dataset or a model to download.")
        sys.exit()
    else:
        if remote_repo[-1] == '/':
            remote_repo = remote_repo[:-1]
            branch = args.branch
        if branch is None:
            branch = "main"
        else:
            try:
                branch = sanitize_branch_name(branch)
            except ValueError as err_branch:
                print(f"Error: {err_branch}")
                sys.exit()

    repo_type = 'model' if args.model else 'dataset'
    if args.output is not None:
        base_folder = args.output
    else:
        base_folder = 'models' if args.model else 'datasets'

    output_folder = f"{'_'.join(remote_repo.split('/')[-2:])}"
    if branch != 'main':
        output_folder += f'_{branch}'

    output_folder = Path(base_folder) / output_folder

    repo = Repository(local_dir=output_folder, clone_from=remote_repo, repo_type=repo_type, skip_lfs_files=True, use_auth_token=HF_TOKEN)

    links_file = output_folder / 'links.json'

    lfs_files, all_files = get_download_links(args.PATHS, output_folder, repo_type, remote_repo, branch)

    lfs_files = []

    # This is to prevent partial download from losing LFS information
    lfs_files_old = []
    if links_file.exists():    
        with open(links_file, 'r') as f:
            lfs_files_old = json.load(f)
            lfs_files.extend(lfs_files_old)

    lfs_files_new, all_files = get_download_links(args.PATHS, output_folder, repo_type, remote_repo, branch)
    lfs_files.extend(lfs_files_new)

    with open(links_file, 'w') as f:
        json.dump(lfs_files, f)

    print(f"Resume downloading of {len(lfs_files_old)} LFS files")

    download_files(lfs_files_old, output_folder, args.threads)

    print(f"Found {len(lfs_files_new)} LFS files to download from {len(all_files)} matched files")

    download_files(lfs_files_new, output_folder, args.threads)
