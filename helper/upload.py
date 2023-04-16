from huggingface_hub import Repository, repository
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('REPO', type=str, default='utensil/storage', nargs='?')
parser.add_argument('--pull', type=bool, default=False, help='Whether to pull before push')

args = parser.parse_args()

if __name__ == '__main__':
    # determine the root of the repo and cd to it
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    os.chdir(ROOT)
    print(f"Working directory changed to: {ROOT}")

    HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

    remote_repo = args.REPO
    local_repo = remote_repo.split('/')[-1]

    repo = Repository(local_dir=local_repo, clone_from=remote_repo, repo_type='dataset', skip_lfs_files=True, use_auth_token=HF_TOKEN)

    if args.pull:
        repo.git_pull()

    
    if not repo.is_repo_clean():
        repo.git_add(auto_lfs_track=True)
        # print(f'Files to be uploaded:\n\n{repository.files_to_be_staged(local_repo)}')
        # repo.push_to_hub()
        print('Upload succeeded.')
    else:
        print('Nothing to upload, exiting...')

