#!/bin/bash
SRC_PATH=${1}
PROJECT_DIR="cross-high-level-reaching"
ROOT_PATH="${SRC_PATH}/${PROJECT_DIR}"

ROOT_PATH="${SRC_PATH}/${PROJECT_DIR}"
if [ -d "$ROOT_PATH" ]; then
    echo "Root path ${ROOT_PATH} confirmed!"
else
    echo "Root path ${ROOT_PATH} does not exsist. Please use > inspect_topics.bash [path_to_your_git_folder] to set the correct directory!"
    exit 1
fi

export GZ_VERSION="8"
export GZ_DISTRO="harmonic"
export GZ_IP="127.0.0.1"
export GZ_PARTITION="$(hostname)"
export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:+${GZ_SIM_RESOURCE_PATH}:}${SRC_PATH}/icrl/models/robot_arm_joints_control"

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

sleep 10 # WAITING FOR THE SIM TO LOAD

echo "--------------------------------------------"
echo "|           INSPECTING TOPICS              |"
echo "--------------------------------------------"

gz topic -l

echo "--------------------------------------------"
echo "|           INSPECTING SERVICES             |"
echo "--------------------------------------------"

gz service -l