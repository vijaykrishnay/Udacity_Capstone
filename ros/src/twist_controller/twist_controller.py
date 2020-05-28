import rospy
from lowpass import LowPassFilter
from pid import PID
from yaw_controller import YawController

MAX_BRAKE = 700
GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, accel_limit, brake_deadband, decel_limit, fuel_capacity, max_lat_accel, max_steer_angle, steer_ratio, vehicle_mass, wheel_base, wheel_radius):
        
        #########################################################################
        
        #### PID Controller parameters ####
        Kp = 0.6
        Ki = 0.01
        Kd = 4
        mn = 0
        mx = 1
#         Kp = 0.5
#         Ki = 0.01
#         Kd = 2
#         mn = 0
#         mx = 1
        
        #### Initalizing the PID controller using Controller gains, minimum and maximum throttle value ####
        
        self.PID_controller = PID(Kp, Ki, Kd, mn, mx)
        
        #### Low Pass Filter parameters ###
        
        tau = 0.5
        ts = 0.02
        
        min_speed = 0
        
        #### Intialize the low pass filter using time constant (tau) and sample time ####
        
        self.Low_pass_filter = LowPassFilter(tau, ts)
        
        #### Initialize the yaw controller using the wheel_base, steer_ratio, min_speed, max_lat_accel and max_steer_angle ####
        
        self.Yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
        
        #### Vehicle parameters and contraints ####
        
        self.accel_limit = accel_limit
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        
        #### Time Initialization ####
        
        self.previous_time = rospy.get_time()
        
        self.Last_throttle = 0
         
        #########################################################################
        
        pass

    def control(self, linear_velocity, angular_velocity, current_velocity, dbw_enabled):
        
        #########################################################################
        
        #### Resetting the PID integrator error when the driver controls the vehicle ####
        
        if not dbw_enabled:
            self.PID_controller.reset()
        
        #### Filtering the noise in velcoity readings ####
        
        current_velocity = self.Low_pass_filter.filt(current_velocity)
        
        #### Throttle Control ####
        
        ## Error calculation ##
        Velocity_error = linear_velocity - current_velocity
        
        ## Sample time calculation ##
        current_time = rospy.get_time()
        sample_time = current_time - self.previous_time
        self.previous_time = current_time
        
        ## Throttle value from PID controller ##
        Throttle = self.PID_controller.step(Velocity_error,sample_time)
        if (Throttle - self.Last_throttle) > 0.01:
            Throttle = self.Last_throttle + 0.01
        
        self.Last_throttle = Throttle
        
        #### Steering Control ####
        
        Steering = self.Yaw_controller.get_steering(linear_velocity, angular_velocity, current_velocity)
        
        #### Brake Control ####
        
        ## For normal case, brake torque applied is equal to 0 ##
        Brake = 0
        
        ## For stopping the vehicle at low speed ##
        if linear_velocity == 0 and current_velocity < 0.1:
            Throttle = 0
            Brake = MAX_BRAKE
        
        ## For releasing throttle when current velocity becomes greater than required velocity ##
        elif Throttle > 0.1 and Velocity_error < 0:
            Throttle -= 0.01
        
        # For heavy braking when current velocity becomes greater than required velocity and throttle value is small ##    
        elif Throttle < 0.1 and Velocity_error < 0:
            Throttle = 0
            Deceleration = abs(max(Velocity_error, self.decel_limit))
            Brake = min(MAX_BRAKE, (Deceleration * self.vehicle_mass * self.wheel_radius))
         
        #########################################################################
       
        return Throttle, Brake, Steering
