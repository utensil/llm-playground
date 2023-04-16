cd "$(dirname "$0")"  # change directory to where the script is located

pip3 install -r requirements.txt

sudo apt-get update

cat packages.txt | xargs sudo apt-get install -y

