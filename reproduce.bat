COPY %cd%\SPXTR.csv %cd%\detail\SP500

REM Download docker image if required.
REM This is an 800 MB download, and progress is not displayed, so be patient.
if not exist %cd%\detail\docker_image.tar.gz (
    powershell -Command "(New-Object Net.WebClient).DownloadFile('https://www.dropbox.com/s/5febdw1j5i3poxh/docker_image.tar.gz?dl=1', '%cd%\detail\docker_image.tar.gz')"
)

REM Load docker image. Note that this image only contains software dependencies, and does not contain results of any pre-run
REM analyses that would appear in the paper. The debendencies consist of R, some R packages, latex, and some latex packages.
REM See detail/Dockerfile for the specific packages used.
docker load --input %cd%\detail\docker_image.tar.gz

REM Reproduce paper within docker container using above image, but first convert dos line endings to unix.
docker run -ti --rm -v %cd%\detail:/home/docker/paper -w /home/docker/paper sampling_variability_forecast_combinations /bin/bash -c ^
    "shopt -s globstar && dos2unix ** > /dev/null 2>&1 && su docker ./sampling_variability_forecast_combinations.sh"

REM Unload docker image, which would otherwise take up 2GB of hard drive space.
docker image rm sampling_variability_forecast_combinations

REM Copy the paper to a more obvious location, ouside of the detail/ subdirectory.
COPY %cd%\detail\sampling_variability_forecast_combinations.pdf %cd%\paper.pdf

PAUSE

