## Energy-Efficient Autonomous Driving with Adaptive Perception and Robust Decision



This repo is the implementation of the following paper:

**Energy-Efficient Autonomous Driving with Adaptive Perception and Robust Decision**  


### main code files
- algs<br>
    - pdqn<br>
       Implementaion for our reinforcement learning algorithm, 
    - replay_buffer<br>
       relay buffer of our reinforcement learning, which is used to store experiences and sample experiences.  
       
- gym_carla<br>
Gym-like carla environment for vehicle agent controlled by reinforcement learning.
    - carla_env.py<br>
    Main module for Gym-like Carla environment, which shares the same APIs as classical [Gym](https://gymnasium.farama.org/).
    Function "reset" is an initialization at the beginning of an episode and Function "step" includes state generation and reward calculation. It includes all the processes of the autonomous vehicle interacting with the environment, including how to use cameras and lidar to obtain environmental data, how to process data, and how to perform actions. In addition, for feature extraction algorithms such as [sparseBev](https://github.com/MCG-NJU/SparseBEV), [sparsefusion](https://github.com/yichen928/SparseFusion), [bevfusion](https://github.com/mit-han-lab/bevfusion), etc., we use their open source code and run them in another process. For simplicity, they transmit data through files.,
    - agent/global_planner.py.
        This module provides a high level route plan, which set a global map route for each vehicle. In addition, BasicAgent implements an algorithm that navigates the scene and contains PID controllers to perform lateral and longitudinal control
    - util. The file bridge_functions.py includes transfer functions for onboard sensors. The file sensor.yaml includes the location and rotation of all sensors, including camera, lidar, imu, and gnss. The classification.py includes the model of our quality classification model
- main.
    The pdqn_multi_lane_sumo.py is the training file of our framework on the Carla simulation. The process.py file has Two functions that are used to start a process or kill a process about simulations.
- Sumo. The bridge_helper.py introduces how to connect the carla simulator and the sumo simulator. In the experiment, carla is responsible for rendering, and sumo is responsible for recording vehicle information and power consumption, etc.

 

## Getting started
1. Install and setup [the CARLA simulator](https://carla.readthedocs.io/en/latest/start_quickstart/#a-debian-carla-installation), set the executable CARLA_PATH in gym_carla/setting.py. Then install [the SUMO simulator](https://sumo.dlr.de/docs/index.html#introduction) and put its path to the environment path: SUMO_HOME

2. Setup conda environment
```shell
$ conda create -n env_name python=3.8
$ conda activate env_name
```
3. Clone the repo and Install the dependent package
```shell
$ git clone https://github.com/xiayuyang/Ene-AD.git
$ pip install -r requirements.txt
```
4. Train the RL agent in the multi-lane scenario
```shell
$ python ./main/trainer/pdqn_multi_lane_sumo.py
```

## License
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Acknowledgements
Our code is based on several repositories:
- [gym-carla](https://github.com/cjy1992/gym-carla.git)
