import time ;
import numpy as np ;

from typing import * ;
from threading import Lock ;
from scipy.spatial.transform import Rotation

import gz.transport13 as gz
from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model
from gz.transport13 import Node ;
from gz.msgs10.empty_pb2 import Empty ;
from gz.msgs10.scene_pb2 import Scene ;
from gz.msgs10.world_control_pb2 import WorldControl ;
from gz.msgs10.boolean_pb2 import Boolean ;
from gz.msgs10.pose_v_pb2 import Pose_V ;

# To Calculate the joint angles via kinematics
class State():
    def __init__(self, start):
        self.reset(start)
    def reset(self, start=None):
        if start:
            self.start = list(start)
        self.previous = list(self.start)
        self.current = list(self.start)
    def move(self,direction):
        self.previous = list(self.current)
        self.current[0] += direction[0]
        self.current[1] += direction[1]
        self.current[2] += direction[2]
    def revert(self):
        self.current = list(self.previous)
    def discrepency(self,actual):
        return np.linalg.norm(np.array(self.current)-np.array(actual))


class RobotArmEnvironmentManager(Node):
    def __init__(self,env_wrapper,**kwargs):
        self.init_node()
        self.mutex = Lock()

        self.env_wrapper = env_wrapper
        self.robot_name='/model/panda_arm'
        self.world_name='/world/robot-arm'

        self.step = 0
        self.task_milestone = 0
        self.last_move_illegal = False
        self.last_obs_time = 0

        if self.subscribe(Pose_V,f'{self.world_name}/dynamic_pose/info',self.gz_handle_observation_callback):
            print("Subscribed to dynamic_pose/info!")

        if self.subscribe(Model,f'{self.robot_name}/joints/state',self.gz_handle_joint_states_callback):
            print("Subscribed to joints/state!")

        self.gz_actions = {}
        index = 1
        for joint in self.env_wrapper.robot.joints:
            joint_name = f"joint{index}"
            index +=1
            self.gz_actions[joint_name] = self.advertise(f'{self.robot_name}/joint/panda_{joint_name}/0/cmd_pos',Double)

        self.wait_for_simulation()

        self.offset = [self.env_wrapper.robot.hand.position.x,self.env_wrapper.robot.hand.position.y,self.env_wrapper.robot.hand.position.z]
        print(f"Panda Hand Position at: {self.offset}")

        self.state = State(self.offset)

        self.world_control_service = f'{self.world_name}/control'
        self.res_req = WorldControl()

    def init_node(self):
        super().__init__()

    def wait_for_simulation(self):
        #print("WAITING FOR THE SIMULATION! THIS IS A WORKAROUND SINCE THE PANDA ROBOT BREAKS THE SCENE/INFO SERVICE!",level=logger.LogLevel.WARN)
        time.sleep(5)
        return

    def request_scene(self):
        result = False;
        start_time = time.time()
        #log(f'Waiting for {self.world_name}/scene/info ...')
        while result is False:
            # Request the scene information
            result, response = self.request(f'{self.world_name}/scene/info', Empty(), Empty, Scene, 1)
            print(f'\rWaiting for {self.world_name}/scene/info ... {(time.time() - start_time):.2f} sec', end='')
            time.sleep(0.1)
        print('\nScene received!')
        return response

    def get_step(self):
        return self.step

    def get_data(self):
        return self.env_wrapper.robot

    def get_position(self):
        return self.state.current
    
    def get_orientation(self):
        return self.env_wrapper.robot.hand.rotation
    
    def get_orientation_euler(self):
        return Rotation.from_quat(self.env_wrapper.robot.hand.rotation)

    def get_last_obs_time(self):
        return self.last_obs_time

    def gz_handle_observation_callback(self,msg):
        with self.mutex:
            self.env_wrapper.robot.hand_callback(msg)
            self.last_obs_time = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nsec

    def gz_handle_joint_states_callback(self,msg):
        with self.mutex:
            self.env_wrapper.robot.joints_callback(msg)

    def gz_perform_action(self, action):
        self.step += 1
        self.state.move(action.amount)
        self.perform_arm_move(self.state.current)
            
            

    def perform_switch(self, task_index:str):
        pass

    def perform_reset(self):
        print("\n/* --------------- Reset Robot Joints --------------- */\n")
        self.task_milestone = 0
        self.state.reset()
        self.perform_arm_move(self.state.start,5)
        if self.last_move_illegal:
            raise Exception("Reset move was illegal! This should never happen")
        print("\n/* -------------------------------------------------- */\n")

    def perform_arm_move(self,position,max_time_waiting_in_sec=1.5):
        joint_actions = self.env_wrapper.robot.compute_inverse_kinematics(position)
        if joint_actions:
            index = 1
            for joint in self.env_wrapper.robot.joints:
                joint_name = f"joint{index}"
                self.perform_joint_rotation(joint_name,joint_actions[index-1])
                index += 1
            time.sleep(0.5)
            distance = self.state.discrepency([self.env_wrapper.robot.hand.position.x,self.env_wrapper.robot.hand.position.y,self.env_wrapper.robot.hand.position.z])
            i = 0
            while distance > self.env_wrapper.goal_discrepency_threshold:
                distance = self.state.discrepency([self.env_wrapper.robot.hand.position.x,self.env_wrapper.robot.hand.position.y,self.env_wrapper.robot.hand.position.z])
                time.sleep(0.1)
                if i > (max_time_waiting_in_sec * 10):
                    #raise Exception("Reset Took Over 10 seconds! Something is obviously not working!")
                    print(f"WARNING: Took Over {max_time_waiting_in_sec} seconds! Assuming arm reached currect position!")
                    break
                i+=1
            print(f"Distance Discrepancy: {distance:.2f}")
            print(f"Action published!")
            self.last_move_illegal = False
        else:
            print(f"ILLEGAL MOVE:{self.state.current}! INSTEAD -> DIDN'T MOVE!")
            self.state.revert()
            self.last_move_illegal = True


    def perform_joint_rotation(self,joint_name,target_rotation):
        message = Double()
        message.data = target_rotation
        success = self.gz_actions[joint_name].publish(message)
        if not success and self.env_wrapper.debug:
            print(f"Error sending message to {joint_name}!!!!")

    def trigger_pause(self, pause):
        raise Exception("THIS DOES NOT WORK IN THIS SCENARIO! ALL SERVICES ARE BROKEN WHEN THE PANDA ROBOT IS IN THE SCENE.")
