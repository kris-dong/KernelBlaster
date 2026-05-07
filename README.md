# KernelBlaster

> Author and affiliation information has been removed for anonymized review.
> See the original (non-anonymized) repository for paper, authorship, and
> contributor details.

## Build the container and launch it

```bash
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
```


## Within the container

```bash
export OPENAI_API_KEY=[your key here]

bash scripts/run_single_kernelblaster.sh
```
