WORKSPACE=${WORKSPACE:-"/workspace"}
JUPYTER_PASSWORD=${JUPYTER_PASSWORD:-"pytorch"}

echo "Launching Jupyter Lab..."
cd /
jupyter lab --allow-root --no-browser --port=8888 --ip=* --ServerApp.token=$JUPYTER_PASSWORD --ServerApp.allow_origin=* --ServerApp.preferred_dir=$WORKSPACE
