"""
$-task high-level reaching scenario
"""
import numpy as np
from gazebo_sim.simulation.Environment import *
from cross_high_level_reaching.EnvironmentManager import RobotArmEnvironmentManager
from gazebo_sim.simulation.PandaRobot import PandaRobot as Robot
from cl_experiment.parsing import Kwarg_Parser

class Task():
    def __init__(self, name, goals):
        self.name = name
        self.goals = goals

    def get_milestone_amount(self):
        return len(self.goals)

class RobotAction():
    def __init__(self,label,amount):
        self.label = label
        self.amount = amount

class RobotArmWrapper():
    def __init__(self,step_duration_nsec=100 * 1000 * 1000,**kwargs) -> None:
        self.config = self.parse_args(**kwargs)
        self.task_list = self.config.task_list
        self.reward_skew = self.config.reward_skew
        self.max_steps_per_episode = self.config.max_steps_per_episode

        # Possible Tasks
        tasks = {}
        tasks["hammer"] = Task("hammer", [[0.5,-0.2,0.2]])
        tasks["push_wall"] = Task("push_wall", [[0.4,0.3,0.1]])
        tasks["faucet_close"] = Task("faucet_close", [[0.8,0.0,0.2]])
        tasks["push_back"] = Task("push_back", [[0.4,0.0,0.1]])
        tasks["stick_pull"] = Task("stick_pull", [[0.5,0.3,0.6]])
        tasks["handle_press_side"] = Task("handle_press_side", [[0.7,0.1,0.3]])
        tasks["push"] = Task("push", [[-0.7,0.0,0.6]])
        tasks["shelf_place"] = Task("shelf_place", [[-0.7,0.3,0.0]])
        tasks["window_close"] = Task("window_close", [[0.5,-0.4,0.7]])
        tasks["peg_unplug_side"] = Task("peg_unplug_side", [[0.5,0.4,0.4]])

        ## Action space of the Robot
        action_speed = self.config.action_speed
        actions = []
        actions.append(RobotAction("left", [-action_speed,0.0,0.0]))
        actions.append(RobotAction("right", [action_speed,0.0,0.0]))
        actions.append(RobotAction("forward", [0.0,0.0,-action_speed]))
        actions.append(RobotAction("backwards", [0.0,0.0,action_speed]))
        actions.append(RobotAction("down", [0.0,-action_speed,0.0]))
        actions.append(RobotAction("up", [0.0,action_speed,0.0]))

        self.robot = Robot(actions)

        self.step_duration_nsec = step_duration_nsec
        self.observation_shape = [6]
        self.tasks = tasks
        self.goal_discrepency_threshold = self.config.goal_discrepency_threshold
        
        
        self.action_entries = actions
        self.nr_actions = len(self.action_entries)

        self.step_count = 0        
        self.task_index = 0

        self.task_milestone = 0

        self.manager = RobotArmEnvironmentManager(self,**kwargs)

        self.info = {
           'input_dims': self.observation_shape,
           'number_actions': len(actions),
           'terminate_cond': 'unassigned',
        }
    
    def get_current_status(self):
        return (self.info['object'][0], self.info['terminate_cond'])
    
    def get_nr_of_tasks(self):
        return len(self.task_list)

    def get_input_dims(self):
        return self.observation_shape

    def get_observation(self, task_id):
        response = self.manager.get_data()
        return np.array([self.manager.state.current[0],self.manager.state.current[1],self.manager.state.current[2],self.tasks[task_id].goals[self.task_milestone][0],self.tasks[task_id].goals[self.task_milestone][1],self.tasks[task_id].goals[self.task_milestone][2]],dtype=np.float32) ;

    def switch(self, task_index: int) -> None:
        self.task_id = self.task_list[task_index]
        print("switching to task", self.task_id, task_index)

        self.manager.perform_switch(task_index)
        self.reset()

    def reset(self):
        self.current_name = self.task_id
        self.info['object'] = (self.current_name,)
        self.step_count = 0

        # stop and wait until robot has arrived at initial position
        self.manager.perform_reset() ;

        state = self.get_observation(self.task_id) ;

        _, _, _ = self.compute_reward(state)

        return (state, self.info)

    def step(self, action_index: int):
        self.perform_action(action_index=action_index)
        self.step_count += 1

        state = self.get_observation(self.task_id)

        ## compute reward
        reward, terminated, truncated = self.compute_reward(state) ;
        print(f'Step: {self.step_count:<4}, Action: {self.action_entries[action_index].label:<8}, Task: {self.current_name:<12}, State: {state}, Reward: {reward:<5}')

        return state,reward,terminated,truncated, self.info ;


    def perform_action(self, action_index:int)->None:
        """ high level action execution """
        if self.config.debug=="yes": print(f'action request at tick [{self.step_count}]')
        action = self.action_entries[action_index] # select action to publish to GZ
        if self.config.debug=="yes": 
            print(f'action i={action_index} published at tick [{self.step_count}]')
       
        self.manager.gz_perform_action(action)


    def compute_reward(self, state):
        truncated = False
        terminated = False
        
        # Get relevant vectors
        current = state[:3]
        target = state[3:]

        # Calculate target distance from start:
        magnitude = np.linalg.norm(np.array(target) - np.array(self.manager.state.start))

        # Euclidean Distance
        dist = np.linalg.norm(current - target)
        normalized_distance = dist/magnitude

        reward = self.skew_reward(1 - normalized_distance,self.reward_skew)
        self.info['terminate_cond'] = "COND: Normal" ;

        if dist < self.goal_discrepency_threshold:
            truncated = True
            self.info['terminate_cond'] = "COND: GOAL REACHED"
            reward = 1.0
        if self.manager.last_move_illegal:  # INVALID MOVE
            self.info['terminate_cond'] = "COND: ILLEGAL MOVE"
            reward = -2.0
        if self.step_count >= self.max_steps_per_episode:
            terminated = True
            self.info['terminate_cond'] = f"COND: MAX STEPS REACHED, CLOSENESS:{normalized_distance:.3f}"

        return reward, truncated, terminated ;

    def skew_reward(self, reward, base=2):
        return np.power(reward,base)

    def parse_args(self, **kwargs):
        parser = Kwarg_Parser(**kwargs) ;
        # ----
        parser.add_argument("--reward_skew", type=float, default=1.0,required=False ) ;
        parser.add_argument("--action_speed", type=float, default=0.1,required=False ) ;
        parser.add_argument("--goal_discrepency_threshold", type=float, default=0.1,required=False ) ;
        cfg,unparsed = parser.parse_known_args() ;
  
        # parse superclass params
        old_cfg = GenericEnvironment.parse_args(self, **kwargs) ;
        
        for attr in dir(old_cfg):
          # exclude magic methods and private fields
          if len(attr) > 2 and (attr[0] != "_" and attr[1] != "_"):
            setattr(cfg, attr,getattr(old_cfg, attr)) ;
        return cfg ;

    def close(self):
        self.manager.destroy_node()
