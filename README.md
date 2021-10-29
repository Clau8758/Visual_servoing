# Visual_servoing

To run the visual servo script on either the simulation or real UR-robot the following steps should be followed.

1. Start the URSIM virtual machine on external computer and start the program "URSim UR5"

2. Connect the two computer using a ethernet cable and use a cross-over cable if neccessary 

3. Run the command:
roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=192.168.0.2 kinematics_config:=/home/test/UR_cali/ur5e_cali.yaml

3. Start the remote control program on the URSim interface. The ROS terminal should acknowledge a successful connection

4. Run the command:
roslaunch ur_robot_driver example_rviz.launch

5. You can confirm the connection is working by running:
rosrun ur_robot_driver test_move

6. Start the visual servoing script. Remember to be inside the correct folder in the terminal. The folder is: /catkin_ws/src/visual_servo/src/:
rosrun visual_servo ibvs_visual_servoing.py
