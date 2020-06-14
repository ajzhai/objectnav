FROM fairembodied/habitat-challenge:testing_2020_habitat_base_docker

# install dependencies in the habitat conda environment
RUN /bin/bash -c ". activate habitat; pip install torch"

ADD ppo_local_planner_no_slam.py ppo_local_planner_no_slam.py
ADD train.sh train.sh
ADD challenge_objectnav2020.local.rgbd.yaml challenge_objectnav2020.local.rgbd.yaml

ENV TRACK_CONFIG_FILE "challenge_objectnav2020.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash train.sh"]
