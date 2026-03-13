# KernelBlaster

## Paper

Corresponding paper: [arXiv:2602.14293](http://arxiv.org/abs/2602.14293)

**Authors**  
[Kris Shengjun Dong](https://people.eecs.berkeley.edu/~chrisdong/), [Sahil Modi](https://www.linkedin.com/in/sahil-modi), [Dima Nikiforov](https://www.linkedin.com/in/dima-n/), [Sana Damani](https://sanadamani.com/), Edward Lin, [Siva Kumar Sastry Hari](https://sivahari.github.io/), [Christos Kozyrakis](https://web.stanford.edu/~kozyraki/)

**Affiliation:** NVIDIA, University of California, Berkeley

***Note:** This repository hosts an archival release of KernelBlaster. The initial commit in this repository does not reflect the original authorship; Most of the original work was contributed to by Kris Shengjun Dong during her 2025 summer internship at NVIDIA.

## Contributors

Main code contributors: Kris Shengjun Dong, Sahil Modi, and Dima Nikiforov.


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
