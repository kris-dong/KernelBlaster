# KernelBlaster

## Build the container and launch it
docker build . -t kernelblaster -f docker/Dockerfile

docker run --rm -it --name=kernelblaster \
    --privileged --gpus all --cap-add=SYS_ADMIN --device /dev/fuse \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --ipc=host --net=host \
    -e USER_NAME=$(whoami) \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    -v $(pwd):/kernelblaster \
    kernelblaster \
    dev # start a bash shell inside the container


## Within the container

export OPENAI_API_KEY=[your key here]

bash scripts/run_single_kernelblaster.sh
