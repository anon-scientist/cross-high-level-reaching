from datetime import datetime
from gazebo_sim.agent import RLAgent ;
from gazebo_sim.learner import DQNLearner
from cross_high_level_reaching.EnvironmentWrapper import RobotArmWrapper
from cl_experiment.parsing import Command_Line_Parser, Kwarg_Parser ;

def main():
    print(f'Begin execution at: {datetime.now()}')
    args_dict = Command_Line_Parser().parse_args() ;
    p = Kwarg_Parser(**args_dict) ;
    p.add_argument("--obs_per_sec_sim_time", type=int, required=True, default=15) ;
    p.add_argument("--algorithm", type=str, required=True) ;
    config, unparsed = p.parse_known_args() ;
    
    # instantiate environment
    # compute nsec delay between two observations
    # complicated by the fact that gazebo computes step durations only to msec precision
    # so a frame rate of 30 means that the delay between two frames is 33msec, but not 33.3333 msec
    # so we have to round down if we want to work with nsec delays     
    hz = 30. ; # we have to know this, definedi n the robot sdf file, camera sensor plugin
    nsec_per_frame = int(1000./hz) * 1000000. ;
    nsec = nsec_per_frame * (hz / config.obs_per_sec_sim_time) ;
    print("Assumed time per frame: ", nsec) ;
    
    env = RobotArmWrapper(step_duration_nsec=nsec,**args_dict)

    # instantiate learner
    if config.algorithm == "DQN":
        learner = DQNLearner(n_actions=len(env.action_entries),
                                obs_space=env.get_input_dims(),
                                config=None,**args_dict) ;

    # instantiate agent
    agent = RLAgent(env, learner,**args_dict)

    # execute experiment
    agent.go()
    print(f'Finish execution at: {datetime.now()}')
    agent.mop_up(); # Terminates debug thread so program can exit

if __name__== "__main__":
    main()