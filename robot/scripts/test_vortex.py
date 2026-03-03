from aira.robot import arm, start_vision_display
import time


# program to test the robot.py file
start_vision_display()  # camera + YOLO view in background; move_to_object no longer blocks on key
a = arm()

# start in the home position
a.go_to('home')


### PICK UP FROM RACK
# if True:
a.go_to('rack_watch')

# open gripper a little wider for a better view
a.set_gripper_position(600, speed=5000)

# move to 50ml eppendorf
a.move_to_object("50ml eppendorf", offset=[-40, 0], pick_type='toolhead_close')

# move down for a better view
a.z_level(180)

# hover over the 50ml eppendorf
a.move_to_object("50ml eppendorf", offset=[0, 0], pick_type='toolhead_close')

# go down over the 50ml eppendorf
a.z_level(102)

# close gripper to grab the 50ml eppendorf
a.set_gripper_position(200, speed=1000)

# move up just a tiny bit
a.z_level(110, speed=50, acc=80)

# drop the 50ml eppendorf into the rack
a.set_gripper_position(600, speed=5000)

# go back down over eppendorf to grab it again
a.z_level(102)

# close gripper to grab the 50ml eppendorf
a.set_gripper_position(200, speed=2000)

# move up to clear the rack
a.z_level(140, speed=50, acc=80)
a.z_level(275)


### END PICK UP FROM RACK ###

### VORTEX TUBE
# if True:
# go to the position where we can roughly see the vortex and move over the hole
a.go_to("vortex_watch")
a.move_to_object("vortex genie hole", offset=[0, 0], pick_type='ranked')

# move down the z-level and then slow down to the final position
a.z_level(270)
a.z_level(255, speed=50, acc=100)

# open the gripper first slow then fast
a.set_gripper_position(350, speed=1000)
a.set_gripper_position(500, speed=5000)

# move tool up to z-level
a.z_level(285)

# move left to bring the end over the tube
a.tool_move(dx=0, dy=-30)

### PRESS SEQUENCE ###
# move down to press the tube
a.z_level(266.5, speed=50, acc=100)

# release slightly to start vibration
a.z_level(268.5, speed=25, acc=50)

time.sleep(2)

# move up to stop
a.z_level(273, speed=50, acc=80)

# wait for 1 second to stop vibration
time.sleep(1)

# move up to clear the tube
a.z_level(285, speed=50, acc=100)

### END PRESS SEQUENCE ###

# move right to get over the tube
a.tool_move(dx=0, dy=30)

# open gripper a little wider
a.set_gripper_position(600, speed=5000)

# move over the tube to correct for any offset
a.move_to_object("50ml eppendorf", offset=[0, 0])

# move down to grab tube
a.z_level(255, speed=100, acc=80)

# close gripper to grab tube slowly
a.set_gripper_position(200, speed=1000)

# move up to clear the the vortex hole
a.z_level(270, speed=50, acc=100)
a.z_level(300)
### END VORTEX TUBE ###

### PLACE INTO RACK ###
# if True:
a.go_to('rack_watch')

# move to the rack hole
a.move_to_object("rack hole", offset=[-40, 0], pick_type='toolhead_close')

# move down for a slightly better view
a.z_level(200)

# move over the hole
a.move_to_object("rack hole", offset=[0, 0], pick_type='toolhead_close', speed=75, acc=100)

# move down to place into the rack
a.z_level(150)
a.z_level(120, speed=50, acc=80)

# open gripper to drop the tube
a.set_gripper_position(600, speed=5000)

# move up to clear the rack hole
a.z_level(175, speed=50, acc=80)

# go back home
a.go_to('home')

### END PLACE INTO RACK ###