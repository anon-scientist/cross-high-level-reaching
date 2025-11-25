import random as rnd
from src.cross_high_level_reaching.EnvironmentWrapper import RobotArmWrapper

args_dict = {"task_list":["hammer","push"],
             "action_speed":"0.1",
             "reward_skew":"1.0",
             "max_steps_per_episode":"30",
             "goal_discrepency_threshold":"0.1"}


# instantiate environment
# compute nsec delay between two observations
# complicated by the fact that gazebo computes step durations only to msec precision
# so a frame rate of 30 means that the delay between two frames is 33msec, but not 33.3333 msec
# so we have to round down if we want to work with nsec delays     
hz = 30. ; # we have to know this, definedi n the robot sdf file, camera sensor plugin
nsec_per_frame = int(1000./hz) * 1000000. ;
nsec = nsec_per_frame * (hz / 15) ; # obs_per_sec_sim_time = 15

env = RobotArmWrapper(nsec,**args_dict) ;

print("Define a task") ;
env.switch(0) ;

end = False
iter = 0
terminated = False
truncated = False
while not end:
    if terminated or truncated: env.reset()
    action_index = rnd.randint(1,4) # For the possible actions look in Environment.py
    obs, reward, terminated, truncated, info = env.step(action_index)
    print(f"--- Step {iter}\nAction {action_index}\nState: {obs}\nReward: {reward}\n")

    if iter >= 30:
        end=True
        break
    iter+=1
