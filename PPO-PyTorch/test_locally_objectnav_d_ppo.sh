#!/usr/bin/env bash

DOCKER_NAME="objectnav_ppo_train"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done

nvidia-docker run -v $(pwd)/data:/data \
    --runtime=nvidia \
    -e "TRACK_CONFIG_FILE=challenge_objectnav2020.local.rgbd.yaml" \
    ${DOCKER_NAME}\

#-e "TRACK_CONFIG_FILE=/challenge_objectnav2020.local.rgbd.yaml" \
