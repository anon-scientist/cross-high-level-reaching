#!/bin/bash
PROCESSES=(
    "gz.*sim"
    "robot_arm.sdf"
    "gazebo_simulator"
    "Experiment.py"
    "ruby"
    "gz"
)

function print_message {
    echo ${3}
}

function print_info { print_message "BLUE"   "INFO" "${*}" ; }
function print_warn { print_message "YELLOW" "WARN" "${*}" ; }
function print_ok   { print_message "GREEN"  "OK"   "${*}" ; }
function print_err  { print_message "RED"    "ERR"  "${*}" ; }
function print_part { print_message "CYAN"   "PART" "${*}" ; }
function print_unk  { print_message "PURPLE" "UNK"  "${*}" ; }

function check_process {
    pgrep -f "${1}"
}

function eval_state {
    local state=$?

    if (( $state == 0 ))
        then print_ok "success ${1}"
        else print_err "failed ${1}"
    fi
    return $state
}

function kill_process {
    pkill -9 -f "${1}"
}

function execute_check {
    print_info "check process ${entry}"
    eval_state $(check_process "${entry}")
}

function execute_kill {
    print_info "try to kill ${entry}"
    eval_state $(kill_process "${entry}")
}

function execute_watchout {
    print_info "watchout for possible zombies"
    for entry in ${PROCESSES[@]}
    do
        execute_check &&
        execute_kill
    done
}

function execute_state {
    state=$?
    if (( $state == 0 ))
        then print_ok "success (${1})"
        else print_err "failed (${1})"
    fi
    return $state
}

# *------------ COMMON DEFINITIONS ----------------------
EXP_ID="robotarm"
SRC_PATH="/home/git"
PROJECT_DIR="cross-high-level-reaching"
echo ARGS $#
if [ "$#" == "1" ] ; then
SRC_PATH=${1} ;
fi
ROOT_PATH="${SRC_PATH}/${PROJECT_DIR}"
if [ -d "$ROOT_PATH" ]; then
    echo "Root path ${ROOT_PATH} confirmed!"
else
    echo "Root path ${ROOT_PATH} does not exsist. Please use > main.bash [path_to_your_git_folder] to set the correct directory!"
    exit 1
fi
# *-------------------------------------------------------

# PYTHONPATH - PYTHONPATH - PYTHONPATH --------------------------------
export PYTHONPATH=$PYTHONPATH:${ROOT_PATH}/src
export PYTHONPATH=$PYTHONPATH:${SRC_PATH}/icrl/src
export PYTHONPATH=$PYTHONPATH:${SRC_PATH}/dcgmm/src
export PYTHONPATH=$PYTHONPATH:${SRC_PATH}/cl_experiment/src
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
echo "PYTHONPATH: ${PYTHONPATH}"
# *--------------------------------------------------------------------

# GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ
export GZ_VERSION="8"
export GZ_DISTRO="harmonic"
export GZ_IP="127.0.0.1"
export GZ_PARTITION="$(hostname)"
export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:+${GZ_SIM_RESOURCE_PATH}:}${SRC_PATH}/icrl/models/robot_arm_cartesian_control"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
echo $GZ_SIM_RESOURCE_PATH
# GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ

# kill zombies
execute_watchout

# start gazebo
LOCAL_EXECUTION="TRUE"
if [ $LOCAL_EXECUTION == "TRUE" ]
then
    echo "LOCAL EXECUTION == TRUE"
    sim_options=" -r --render-engine ogre";
else
    sim_options=" -r -s --headless-rendering --render-engine ogre2";
fi

gz sim ${sim_options} "${ROOT_PATH}/simulation/gazebo/robot-arm.sdf"  &

echo Executing cross_high_level_reaching.Experiment &

# +++
python3 -m cross_high_level_reaching.Experiment                                                                             \
        --benchmark                                         robotarm                                                        \
        --exp_id                                            "${EXP_ID}"                                                     \
        --root_dir                                          "${ROOT_PATH}"                                                  \
        --obs_per_sec_sim_time                              15                                                              \
        --training_duration                                 20                                                              \
        --evaluation_duration                               2                                                               \
        --training_duration_unit                            timesteps                                                       \
        --evaluation_duration_unit                          episodes                                                        \
        --max_steps_per_episode                             40                                                              \
        --task_list                                         hammer hammer hammer                                            \
        --start_task                                        0                                                               \
        --eval_start_task                                   1                                                               \
        --exploration_start_task                            1                                                               \
        --start_task_ar                                     2                                                               \
        --training_duration_task_0                          100                                                             \
        --gamma                                             0.8                                                             \
        --train_batch_size                                  32                                                              \
        --algorithm                                         DQN                                                             \
        --dqn_fc1_dims                                      128                                                             \
        --dqn_fc2_dims                                      64                                                              \
        --dqn_adam_lr                                       1e-4                                                            \
        --dqn_dueling                                       no                                                              \
        --dqn_target_network                                yes                                                             \
        --dqn_target_network_update_freq                    200                                                             \
        --goal_discrepency_threshold                        0.1                                                             \
        --action_speed                                      0.1                                                             \
        --reward_skew                                       1.0                                                             \
        --exploration                                       eps-greedy                                                      \
        --initial_epsilon                                   1.0                                                             \
        --final_epsilon                                     0.2                                                             \
        --epsilon_delta                                     0.0005                                                          \
        --eps_replay_factor                                 0.8                                                             \
        --replay_buffer                                     default                                                         \
        --capacity                                          200                                                             \
        --per_alpha                                         0.6                                                             \
        --per_beta                                          0.6                                                             \
        --per_eps                                           1e-6                                                            \
        --per_delta_beta                                    0.00005                                                         \
        ; execute_state "Experiment"
# ---

echo DONE

# kill zombies
execute_watchout