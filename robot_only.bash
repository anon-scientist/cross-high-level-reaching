#!/bin/bash
# starts a Python script that demonstrates use of the robot, without environment or RL framework

SRC_PATH=${1}
PROJECT_DIR="cross-high-level-reaching"
ROOT_PATH="${SRC_PATH}/${PROJECT_DIR}"
if [ -d "$ROOT_PATH" ]; then
    echo "Root path ${ROOT_PATH} confirmed!"
else
    echo "Root path ${ROOT_PATH} does not exsist. Please use > robot_only.bash [path_to_your_git_folder] to set the correct directory!"
    exit 1
fi

export PYTHONPATH=$PYTHONPATH:${ROOT_PATH}/src
export PYTHONPATH=$PYTHONPATH:${SRC_PATH}/icrl/src
export PYTHONPATH=$PYTHONPATH:${SRC_PATH}/cl_experiment/src
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export GZ_VERSION="8"
export GZ_DISTRO="harmonic"
export GZ_IP="127.0.0.1"
export GZ_PARTITION="$(hostname)"
export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:+${GZ_SIM_RESOURCE_PATH}:}${SRC_PATH}/icrl/models/robot_arm_cartesian_control"

LOCAL_EXECUTION="TRUE" # Change if you want with/without gui
if [ $LOCAL_EXECUTION == "TRUE" ]
then
    echo "LOCAL EXECUTION == TRUE"
    sim_options=" -r --render-engine ogre";
else
    sim_options=" -r -s --headless-rendering --render-engine ogre2";
fi

echo "${ROOT_PATH}/simulation/gazebo/robot-arm.sdf"

gz sim ${sim_options} "${ROOT_PATH}/simulation/gazebo/robot-arm.sdf"  &

python3 robot_only.py
