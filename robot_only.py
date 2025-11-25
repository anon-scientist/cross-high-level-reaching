
"""
CRoSS benchmark suite, HLR  benchmark. Example for standalone use of provided
robot manager, without RL framework
"""
from gazebo_sim.simulation.PandaRobot import PandaRobot
from src.cross_high_level_reaching.EnvironmentManager import RobotArmEnvironmentManager
end=False

class RobotAction():
    def __init__(self,label,amount):
        self.label = label
        self.amount = amount

class Wrapper():
    def __init__(self,robot):
        self.robot = robot
        self.goal_discrepency_threshold = 0.1

action_speed = 0.1
actions = []
actions.append(RobotAction("backwards", [-action_speed,0.0,0.0]))# 0
actions.append(RobotAction("forward", [action_speed,0.0,0.0]))# 1
actions.append(RobotAction("downwards", [0.0,0.0,-action_speed]))# 2
actions.append(RobotAction("upwards", [0.0,0.0,action_speed]))# 3
actions.append(RobotAction("left", [0.0,-action_speed,0.0]))# 4
actions.append(RobotAction("right", [0.0,action_speed,0.0]))# 5

wrapper = Wrapper(PandaRobot(actions))

manager = RobotArmEnvironmentManager(wrapper)
manager.perform_switch(0)
iter = 0
while not end:
    if (iter % 30) == 0:
        print(f"Current Pos: {manager.state.current}. RESETTING ...")
        manager.perform_reset()
        
    userin = input(f"Move with [WASDRF]. Current Pos: {manager.state.current}: ")
    illegal = False
    if userin == "w" or userin == "W":
        illegal = manager.gz_perform_action(actions[1])
    elif userin == "a" or userin == "A":
        illegal = manager.gz_perform_action(actions[5])
    elif userin == "s" or userin == "S":
        illegal = manager.gz_perform_action(actions[0])
    elif userin == "d" or userin == "D":
        illegal = manager.gz_perform_action(actions[4])
    elif userin == "r" or userin == "R":
        illegal = manager.gz_perform_action(actions[3])
    elif userin == "f" or userin == "F":
        illegal = manager.gz_perform_action(actions[2])
    elif userin == "e" or userin == "exit" or userin == "q" or userin == "quit":
        end = True
        break;
    else:
        print("Illegal Input. To exit write 'e' or 'exit'")
        continue
    if illegal:
        print("This move is illegal and cannot be performed. Try other actions.")
    iter += 1


