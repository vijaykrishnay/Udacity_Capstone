#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

#############################################################

        #### Initializing the Controller object ####

        self.controller = Controller(accel_limit,
                                     brake_deadband,
                                     decel_limit,
                                     fuel_capacity,
                                     max_lat_accel,
                                     max_steer_angle,
                                     steer_ratio,
                                     vehicle_mass,
                                     wheel_base,
                                     wheel_radius)

        #### Subscribing to the three topics namely, "/vehicle/dbw_enabled", "/twist_cmd", "/current_velocity" ####

        rospy.Subscriber(Bool, '/vehicle/dbw_enabled', self.dbw_enabled_message, queue_size = 5)
        rospy.Subscriber(TwistStamped, '/twist_cmd', self.twist_cmd_message, queue_size = 5)
        rospy.Subscriber(TwistStamped, '/current_velocity', self.current_velocity_message, queue_size = 5)

        #### Initializing necessary parameters if no messages arrives ####

        ## Input ##

        self.dbw_enabled = None
        self.linear_velocity = None
        self.angular_velocity = None
        self.current_velocity = None

        ## Output ##

        self.Throttle = 0
        self.Brake = 0
        self.Steering = 0

        #### Calling the control function of the Class controller using the function "loop" ####

        self.loop()

    #### Initializing the parameters if the messages arrives ####

    def dbw_enabled_message(self, msg):
        self.dbw_enabled = msg

    def twist_cmd_message(self, msg):
        self.linear_velocity = msg.twist.linear.x
        self.angular_velocity = msg.twist.angular.z

    def current_velocity_message(self, msg):
        self.current_velocity = msg.twist.linear.x

#############################################################

    def loop(self):
        rate = rospy.Rate(50) # 50Hz
        while not rospy.is_shutdown():

            #### Finding the throttle, brake and steering values ####

            if all(i is not None for i in [self.dbw_enabled, self.linear_velocity, self.current_velocity]):
                self.Throttle, self.Brake, self.Steering = self.controller.control(self.linear_velocity,
                                                                                   self.angular_velocity,
                                                                                   self.current_velocity,
                                                                                   self.dbw_enabled)

            #### Publishing the throttle, brake and steering value only when the drive by wire is enabled ####
            if dbw_enabled:
                self.publish(self.Throttle, self.Brake, self.Steering)

            rate.sleep()

##############################################################


    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
