import rospy
from lowpass import LowPassFilter
from pid import PID
from yaw_controller import YawController

MAX_BRAKE = 700
GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, accel_limit, brake_deadband, decel_limit, fuel_capacity, max_lat_accel, max_steer_angle, steer_ratio, vehicle_mass, wheel_base, wheel_radius):
        
        # PID Controller parameters
        Kp = 0.6
        Ki = 0.01
        Kd = 4.
        mn = 0.
        mx = 1.

        # Initalizing the PID controller using Controller gains, minimum and maximum throttle value
        self.PID_controller = PID(Kp, Ki, Kd, mn, mx)
        
        # Low Pass Filter parameters
        tau = 0.5
        ts = 0.02
        
        min_speed = 0
        
        # Intialize the low pass filter using time constant (tau) and sample time
        self.Low_pass_filter = LowPassFilter(tau, ts)
        
        # Initialize the yaw controller using the wheel_base, steer_ratio, min_speed, max_lat_accel and max_steer_angle
        self.Yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
        
        # Vehicle parameters and contraints
        self.accel_limit = accel_limit
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        
        #### Time Initialization ####
        self.previous_time = rospy.get_time()
        
        self.last_throttle = 0
    
    def control(self, linear_velocity, angular_velocity, current_velocity, dbw_enabled):
        """
        Calculates throttle, brake and steering based on twist data (linear, angular velocities), current velocity and dbw state.
        """

        # Resetting the PID integrator error when the driver controls the vehicle
        if not dbw_enabled:
            self.PID_controller.reset()
            return 0., 0., 0.
        
        # Filtering the noise in velcoity readings
        current_velocity = self.Low_pass_filter.filt(current_velocity)
        
        #### Throttle Control ####       
        # Error calculation
        velocity_error = linear_velocity - current_velocity
        
        # Sample time calculation
        current_time = rospy.get_time()
        sample_time = current_time - self.previous_time
        self.previous_time = current_time
        
        # Throttle value from PID controller
        throttle = self.PID_controller.step(velocity_error, sample_time)
        if (throttle - self.last_throttle) > 0.01:
            throttle = self.last_throttle + 0.01
        
        self.last_throttle = throttle
        
        #### Steering Control
        steering = self.Yaw_controller.get_steering(linear_velocity, angular_velocity, current_velocity)
        
        #### Brake Control ####        
        # For normal case, brake torque applied is equal to 0
        brake = 0
        
        if linear_velocity == 0 and current_velocity < 0.1:
            # Stop the vehicle at low speed
            throttle = 0
            brake = MAX_BRAKE
        elif throttle > 0.1 and velocity_error < 0:
            # Release throttle when current velocity becomes greater than required velocity
            throttle -= 0.01
        elif throttle < 0.1 and velocity_error < 0:
            # Heavy braking when current velocity becomes greater than required velocity and throttle value is small
            throttle = 0
            deceleration = abs(max(velocity_error, self.decel_limit))
            brake = min(MAX_BRAKE, (deceleration * self.vehicle_mass * self.wheel_radius))
         
        return throttle, brake, steering
