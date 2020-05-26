
GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, accel_limit, brake_deadband, decel_limit, fuel_capacity, max_lat_accel, max_steer_angle, steer_ratio, vehicle_mass, wheel_base, wheel_radius):

        #########################################################################

#         Steps: -
#             Initialize the yaw controller using the wheel_base, steer_ratio, min_speed, max_lat_accel and max_steer_angle
#             Initalize the PID controller using Controller gains, minimum and maximum throttle value
#             Intialize the low pass filter using time constant (tau) and sample time


        #########################################################################

        pass

    def control(self, linear_velocity, angular_velocity, current_velocity, dbw_enabled):

        #########################################################################

#         Use low pass filter to avoid noise in velocity
#         Use PID controller to get throttle value
#         Use yaw controller to get steering value
#         Initialize brake value accoring to the difference between linear_velocity and current_velocity
#         Return throttle, brake, steer

        #########################################################################

        return 1., 0., 0.
