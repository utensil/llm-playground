from huggingface_hub import Repository, repository
import argparse
import os

DEFAULT_REPO = 'utensil/storage'
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

parser = argparse.ArgumentParser()
parser.add_argument('REPO', type=str, default=DEFAULT_REPO, nargs='?', help="The HF repo to upload to: <user_name>/<repo_name>")
parser.add_argument('ROOT', type=str, default=DEFAULT_ROOT, nargs='?', help="The local repo will clone to ROOT/<repo_name>")
parser.add_argument('-m', '--model', type=bool, nargs='?', default=False, const=True, help='Set repo type to model (default: dataset)')
parser.add_argument('-p', '--pull', type=bool, nargs='?', default=False, const=True, help='Whether to pull before push')
parser.add_argument('-u', '--upload', type=bool, nargs='?', default=False, const=True, help='Whether to upload local changes')
parser.add_argument('-l', '--lfs', type=bool, nargs='?', default=False, const=True, help='Whether to pull LFS files')

args = parser.parse_args()

if __name__ == '__main__':
    HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
    
    ROOT = args.ROOT
    os.chdir(ROOT)
    
    remote_repo = args.REPO
    local_repo = remote_repo.split('/')[-1]

    repo_type = 'model' if args.model else 'dataset'

    repo = Repository(local_dir=local_repo, clone_from=remote_repo, repo_type=repo_type, skip_lfs_files=not args.lfs, use_auth_token=HF_TOKEN)

    if args.pull:
        repo.git_pull(args.lfs)

    if args.upload:
        if not repo.is_repo_clean():
            repo.git_add(auto_lfs_track=True)
            # print(f'Files to be uploaded:\n\n{repository.files_to_be_staged(local_repo)}')
            repo.push_to_hub()
            print('Upload succeeded.')
        else:
            print('Nothing to upload, exiting...')

