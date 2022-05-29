#!/bin/bash

# Abort if any commands fail.
set -e

# Get nesting user calling sudo, and the nested user, which is root if this script is called with sudo,
# and $NESTING_USER otherwise.
NESTING_USER="$(logname)"
NESTED_USER="$(whoami)"

DOCKERFILE_URL="https://melb.zischke.net.au/index.php/s/bfkvI8wZzCqBRdT/download"

# This functions ensures that the files whose names are passed as arguments
# is owned by $NESTING_USER.
nonsudo_ownership () {
    # Check for root, since nonroot permissions are not adequate
    # for changing file ownership in many cases, and nothing
    # needs to be done if this file wasn't run as root on behalf
    # of a user with sudo permissions anyway.
    if [[ "$NESTED_USER" == "root" ]]; then
        if [[ "$(uname)" == "Darwin" ]]; then
            # Assume MacOS.
            chown "${NESTING_USER}" "$@"
        else
            # Assume linux.
            chown "${NESTING_USER}:${NESTING_USER}" "$@"
        fi
    fi
}

cp SPXTR.csv detail/SP500
nonsudo_ownership detail/SP500/SPXTR.csv

(
    cd detail

    # Load docker image, downloading it or creating it from detail/Dockerfile if necessary. Note that the Dockerfile
    # depends on particular versions of software being available, so will quickly become out of date. For this reason
    # we package the docker image into a tarball, and download and load the image for all future executions of this
    # script. Note that the Dockerfile does not contain any instructions to run any of the analyses whose results are
    # presented in the paper, only to obtain software dependencies required to conduct the analyses and reproduce the
    # results and the paper, consisting of R, some R packages, latex, and some latex packages. See detail/Dockerfile
    # for the specific packages used.
    if [[ -f "docker_image.tar.gz" ]]; then
        docker load --input docker_image.tar.gz
    else
        # If $DOCKERFILE_URL can be reached, download docker image from there, and load. Otherwise build from Dockerfile.
        if wget --spider --quiet "${DOCKERFILE_URL}" ; then
            wget "${DOCKERFILE_URL}" -O docker_image.tar.gz
            nonsudo_ownership docker_image.tar.gz
            docker load --input docker_image.tar.gz
        else
            docker build -t sampling_variability_forecast_combinations .
            docker save sampling_variability_forecast_combinations | gzip -9 > docker_image.tar.gz
            nonsudo_ownership docker_image.tar.gz            
        fi
    fi

    # Reproduce the results and the paper, which will initially reside at detail/sampling_variability_forecast_combinations.pdf.
    docker run -ti --rm -v "$PWD":/home/docker/paper -w /home/docker/paper -u docker sampling_variability_forecast_combinations /bin/bash -c \
        ./sampling_variability_forecast_combinations.sh
)

# Unload docker image, which would otherwise take up 2GB of hard drive space.
docker image rm sampling_variability_forecast_combinations

# Copy the paper to a more obvious location, outside of the detail/ subdirectory.
cp detail/sampling_variability_forecast_combinations.pdf paper.pdf
nonsudo_ownership paper.pdf

