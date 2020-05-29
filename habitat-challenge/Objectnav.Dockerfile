FROM fairembodied/habitat-challenge:testing_2020_habitat_base_docker

# install dependencies in the habitat conda environment
RUN /bin/bash -c ". activate habitat; pip install torch"

ADD agent.py agent.py
ADD submission.sh submission.sh
ADD configs/challenge_objectnav2020.local.rgbd.yaml /challenge_objectnav2020.local.rgbd.yaml
ENV AGENT_EVALUATION_TYPE remote

ENV TRACK_CONFIG_FILE "/challenge_objectnav2020.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh"]
