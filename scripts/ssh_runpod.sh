if [ ! -f ~/.ssh/id_ed25519 ]; then
    mkdir -p ~/.ssh/
    echo $RUNPOD_SSH_KEY | sed -e 's/_/\n/g' > ~/.ssh/id_ed25519
fi
ssh $username@ssh.runpod.io -i ~/.ssh/id_ed25519