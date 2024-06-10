---
layout: distill
title: Set up CARLA with Docker
description:
tags: Tutorials
categories: gsoc-2024
date: 2024-06-10
permalink: /blog/2024/carla-docker/
# featured: true

authors:
  - name: Zebin Huang
    affiliations:
      name: Edinburgh Centre for Robotics

bibliography: posts.bib
---

This guide will walk you through the process of setting up the CARLA simulator (version 0.9.14) using Docker.

## Prerequisites

- Docker installed on your system.
- NVIDIA Docker if you are using NVIDIA GPUs for rendering.

## Pull the CARLA Docker Image

Start by pulling the official CARLA image from Docker Hub:

```bash
docker pull carlasim/carla:0.9.14
```

## Run CARLA in a Docker Container

To run CARLA with Docker, you can use the following command. This setup forwards necessary ports and configures the environment for GPU usage and offscreen rendering:

```bash
sudo docker run -p 2000-2002:2000-2002 --privileged --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:0.9.14 /bin/bash ./CarlaUE4.sh -RenderOffScreen
```

## Installing CARLA and its Python API

### Installing CARLA on Debian-based Systems

Debian packages of CARLA are available for both Ubuntu 18.04 (Bionic Beaver) and Ubuntu 20.04 (Focal Fossa). However, the officially supported version is Ubuntu 18.04. Here's how to install CARLA using the Debian package repository:

1. **Add the CARLA Repository to Your System:**
This step involves adding the GPG key for the CARLA repository to your system and then adding the repository itself.

    ```bash
    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1AF1527DE64CB8D9
    sudo add-apt-repository "deb [arch=amd64] http://dist.carla.org/carla $(lsb_release -sc) main"
    ```

2. **Install CARLA:**
Update your package list and install CARLA. The installation directory will be `/opt/carla-simulator/`.

    ```bash
    sudo apt-get update
    sudo apt-get install carla-simulator
    cd /opt/carla-simulator
    ```

### Manual Installation from GitHub

If the Debian server is down, as has been reported in issues like [CARLA GitHub Issue #7017](https://github.com/carla-simulator/carla/issues/7017), you can manually install CARLA:

1. **Install Required System Dependency:**
Before downloading CARLA, make sure that all necessary dependencies are installed.

    ```bash
    sudo apt-get -y install libomp5
    ```

2. **Download CARLA Release:**
Use `wget` to download the CARLA release directly from its S3 bucket. Adjust the version number as needed for the specific version you are installing.

    ```bash
    # Download CARLA version 0.9.14
    wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.14.tar.gz
    ```

3. **Unpack CARLA:**
Unpack the downloaded archive to the desired directory, typically `/opt/carla-simulator/`.

    ```bash
    tar -xzvf CARLA_0.9.14.tar.gz -C /opt/carla-simulator/
    ```

4. **Install the CARLA Python Module:**
Install the CARLA Python module and its dependencies to ensure that you can interact with CARLA using Python scripts.

    ```bash
    # Ensure pip is up to date
    pip install --upgrade pip
    # Install CARLA Python API
    python -m pip install carla==0.9.14
    # Install additional Python dependencies required for CARLA
    python -m pip install -r /opt/carla-simulator/PythonAPI/examples/requirements.txt
    ```

## Check the Server is Running

To verify the status of the CARLA server, there are several methods you can use to check if it is running properly:

1. **Check Container Processes**:
If you have launched the CARLA server in a Docker container, you can check the list of processes by running the `ps` command. In the command line of your container, enter the following command:

    ```bash
    ps aux | grep CarlaUE4
    ```

    If the CARLA server is running, you should see a process listed that includes `CarlaUE4.sh` or a similar name.

2. **Check Log Output**:
The CARLA server typically outputs logs. If you start the server's Docker container without running it in the background, you should be able to see the outputs directly in your terminal.
3. **Check Network Listening Ports**:
The CARLA server listens on specific ports by default (e.g., 2000-2002).
On the host machine, you can use the following command to check if these ports are being listened to:

    ```bash
    sudo netstat -tulpn | grep LISTEN
    ```

    Look for ports `2000`, `2001`, and `2002` in the output to see if any process is bound to these ports. The output should be similar to:

    ```bash
    tcp        0      0 0.0.0.0:2001            0.0.0.0:*               LISTEN      532651/CarlaUE4-Lin
    tcp        0      0 0.0.0.0:2000            0.0.0.0:*               LISTEN      532651/CarlaUE4-Lin
    tcp        0      0 0.0.0.0:2002            0.0.0.0:*               LISTEN      532651/CarlaUE4-Lin
    ```

4. **Testing Connection with CARLA Python API**:
If you have a Python environment set up with the CARLA Python client library, you can try writing a simple script to attempt a connection to the server:

    ```python
    import carla

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        print("CARLA server is running!")
        # Optionally, get some additional data
        world = client.get_world()
        print("Connected to world: ", world.get_map().name)
    except RuntimeError:
        print("Cannot connect to CARLA server.")
    ```

    This script will attempt to connect to the CARLA server running locally on port 2000. If the connection is successful, it will print that the server is running.

## Troubleshooting Common Errors

When working with CARLA and Docker, you might encounter errors such as connection timeouts or ALSA audio issues. Below are some common errors.

### Connection Timeout

After running the following command:

```bash
sudo nvidia-docker run -p 2000-2002:2000-2002 -it --rm  carlasim/carla:0.9.14 /bin/bash
SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl
```

Error Message:

```bash
Cannot connect to CARLA server: time-out of 10000ms while waiting for the simulator, make sure the simulator is ready and connected to localhost:2000
```

Solution:

- Ensure that the Docker container is running and the ports are correctly mapped.
- Use the following command to check if CARLA is actively listening on the expected ports:

    ```bash
    sudo netstat -tulpn | grep LISTEN
    ```

### ALSA Audio Errors

When running CARLA, you might see repeated ALSA errors. These generally indicate missing audio configurations, which are common in Docker containers but don't affect the simulation:

```bash
(carla_0914_env) (base) Procuda% docker run -e DISPLAY=$DISPLAY -it --net=host --gpus all carlasim/carla:0.9.14 /bin/bash ./CarlaUE4.sh -opengl -world-port=2000 -RenderOffScreen
4.26.2-0+++UE4+Release-4.26 522 0
Disabling core dumps.
sh: 1: xdg-user-dir: not found
ALSA lib confmisc.c:767:(parse_card) cannot find card '0'
ALSA lib conf.c:4528:(_snd_config_evaluate) function snd_func_card_driver returned error: No such file or directory
ALSA lib confmisc.c:392:(snd_func_concat) error evaluating strings
ALSA lib conf.c:4528:(_snd_config_evaluate) function snd_func_concat returned error: No such file or directory
ALSA lib confmisc.c:1246:(snd_func_refer) error evaluating name
ALSA lib conf.c:4528:(_snd_config_evaluate) function snd_func_refer returned error: No such file or directory
ALSA lib conf.c:5007:(snd_config_expand) Evaluate error: No such file or directory
ALSA lib pcm.c:2495:(snd_pcm_open_noupdate) Unknown PCM default
ALSA lib confmisc.c:767:(parse_card) cannot find card '0'
ALSA lib conf.c:4528:(_snd_config_evaluate) function snd_func_card_driver returned error: No such file or directory
ALSA lib confmisc.c:392:(snd_func_concat) error evaluating strings
ALSA lib conf.c:4528:(_snd_config_evaluate) function snd_func_concat returned error: No such file or directory
ALSA lib confmisc.c:1246:(snd_func_refer) error evaluating name
ALSA lib conf.c:4528:(_snd_config_evaluate) function snd_func_refer returned error: No such file or directory
ALSA lib conf.c:5007:(snd_config_expand) Evaluate error: No such file or directory
ALSA lib pcm.c:2495:(snd_pcm_open_noupdate) Unknown PCM default
ALSA lib confmisc.c:767:(parse_card) cannot find card '0'
ALSA lib conf.c:4528:(_snd_config_evaluate) function snd_func_card_driver returned error: No such file or directory
ALSA lib confmisc.c:392:(snd_func_concat) error evaluating strings
ALSA lib conf.c:4528:(_snd_config_evaluate) function snd_func_concat returned error: No such file or directory
ALSA lib confmisc.c:1246:(snd_func_refer) error evaluating name
ALSA lib conf.c:4528:(_snd_config_evaluate) function snd_func_refer returned error: No such file or directory
ALSA lib conf.c:5007:(snd_config_expand) Evaluate error: No such file or directory
ALSA lib pcm.c:2495:(snd_pcm_open_noupdate) Unknown PCM default
ALSA lib confmisc.c:767:(parse_card) cannot find card '0'
ALSA lib conf.c:4528:(_snd_config_evaluate) function snd_func_card_driver returned error: No such file or directory
ALSA lib confmisc.c:392:(snd_func_concat) error evaluating strings
ALSA lib conf.c:4528:(_snd_config_evaluate) function snd_func_concat returned error: No such file or directory
ALSA lib confmisc.c:1246:(snd_func_refer) error evaluating name
ALSA lib conf.c:4528:(_snd_config_evaluate) function snd_func_refer returned error: No such file or directory
ALSA lib conf.c:5007:(snd_config_expand) Evaluate error: No such file or directory
ALSA lib pcm.c:2495:(snd_pcm_open_noupdate) Unknown PCM default
```

### "xdg-user-dir: not found" Error

When running CARLA, especially in Docker environments, you might encounter the `xdg-user-dir: not found` error. This error typically occurs because the `xdg-user-dirs` package, which is responsible for managing user directories like Downloads, Desktop, etc., is not installed or accessible in the container. While this error generally does not impact the functionality of CARLA if you're not using these directories (see [this issue](https://github.com/carla-simulator/carla/issues/3514)).
