#!/bin/bash

# Abort if any commands fail.
set -e

# Get nesting user calling sudo, and the nested user, which is root if this script is called with sudo,
# and $NESTING_USER otherwise.
NESTING_USER="$(logname)"
NESTED_USER="$(whoami)"

(
    cd detail

    # Load docker image, creating it from detail/Dockerfile if necessary. Note that the Dockerfile depends on particular
    # versions of software being available, so will quickly become out of date. For this reason we package the docker
    # image into a tarball, which then forms part of the github repository, and load the image from there for all future
    # executions of this script. Note that the Dockerfile does not contain any instructions to run any of the analyses
    # whose results are presented in the paper, only to obtain software dependencies required to conduct the analyses
    # and reproduce the results and the paper, consisting of R, some R packages, latex, and some latex packages.
    # See detail/Dockerfile for the specific packages used.
    if [[ -f "docker_image.tar.gz" ]]; then
        docker load --input docker_image.tar.gz
    else
        docker build -t sampling_variability_forecast_combinations .
        docker save sampling_variability_forecast_combinations | gzip -9 > docker_image.tar.gz

        # If this script was run using root, change the ownership of docker_image.tar.gz to
        # the nesting user.
        if [[ "$NESTED_USER" == "root" ]]; then
            if [[ "$(uname)" == "Darwin" ]]; then
                # Assume macos.
                chown "${NESTING_USER}" docker_image.tar.gz
            else
                # Assume linux.
                chown "${NESTING_USER}:${NESTING_USER}" docker_image.tar.gz
            fi
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

# If this script was run using root, change the ownership of paper.pdf
# to the nesting user.
if [[ "$NESTED_USER" == "root" ]]; then
    if [[ "$(uname)" == "Darwin" ]]; then
        # Assume macos.
        chown "${NESTING_USER}" paper.pdf
    else
        # Assume linux.
        chown "${NESTING_USER}:${NESTING_USER}" paper.pdf
    fi
fi

