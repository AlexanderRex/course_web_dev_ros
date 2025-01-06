#! /usr/bin/env python3
import rospy
import actionlib
import math
import time

from course_web_dev_ros.msg import (
    WaypointActionAction,
    WaypointActionFeedback,
    WaypointActionResult
)
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from tf import transformations


class WaypointActionClass(object):

    # States
    TURNING = 0
    STOPPED = 1
    MOVING = 2

    def __init__(self):
        """
        Create a SimpleActionServer for 'tortoisebot_as' that takes
        a WaypointActionAction goal (target position) and controls
        the robot to reach that goal with a state machine + PID/PD logic.
        """
        self._as = actionlib.SimpleActionServer(
            "tortoisebot_as",
            WaypointActionAction,
            execute_cb=self.on_goal,
            auto_start=False
        )
        self._as.start()

        # Publishers & Subscribers
        self._pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self._sub_odom = rospy.Subscriber("/odom", Odometry, self.odom_callback)

        # Feedback / result
        self._feedback = WaypointActionFeedback()
        self._result = WaypointActionResult()

        # Rate
        self._rate = rospy.Rate(10)  # 10 Hz (adjust as needed)

        # Robot state
        self._position = Point()
        self._yaw = 0.0

        # Target
        self._des_pos = Point()

        # Thresholds
        self._dist_threshold = 0.05   # when we consider the goal reached
        self._yaw_threshold = 0.01    # ~0.57 degrees

        # Current state (TURNING, STOPPED, MOVING)
        self._state = self.TURNING

        # PD for turning
        self._kp_turn = 2.0
        self._kd_turn = 0.5
        self._prev_turn_err = 0.0

        # PID for distance
        self._kp_dist = 1.0
        self._ki_dist = 0.01
        self._kd_dist = 0.05
        self._integral_dist = 0.0
        self._prev_dist_err = 0.0

        # Time to compute derivative terms
        self._time_prev = rospy.Time.now().to_sec()

        rospy.loginfo("Action Server 'tortoisebot_as' started with PID control")

    def odom_callback(self, msg):
        """
        Odom callback to update the robot's position and yaw.
        """
        self._position = msg.pose.pose.position

        # Convert quaternion to yaw
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        euler = transformations.euler_from_quaternion(quaternion)
        self._yaw = euler[2]

    def normalize_angle(self, angle):
        """
        Normalize an angle to the range [-pi, pi].
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def on_goal(self, goal):
        """
        Called when a new goal is received.
        goal.position is the target position (Point).
        We'll use a state machine approach with TURNING -> STOPPED -> MOVING.
        """
        rospy.loginfo("Goal received: {}".format(goal))
        success = True

        # Set desired position
        self._des_pos = goal.position

        # Reset errors for PID
        self._prev_turn_err = 0.0
        self._prev_dist_err = 0.0
        self._integral_dist = 0.0

        self._time_prev = rospy.Time.now().to_sec()

        # Start in TURNING state
        self._state = self.TURNING

        # Control loop
        while not rospy.is_shutdown():
            # Check if preempt (cancel) was requested
            if self._as.is_preempt_requested():
                rospy.loginfo("Goal was cancelled/preempted")
                self._as.set_preempted()
                success = False
                break

            # 1) Compute current errors
            # Distance error
            dx = self._des_pos.x - self._position.x
            dy = self._des_pos.y - self._position.y
            distance_error = math.sqrt(dx * dx + dy * dy)

            # Check if goal reached
            if distance_error < self._dist_threshold:
                rospy.loginfo("Goal reached!")
                break

            # Yaw error
            desired_yaw = math.atan2(dy, dx)
            yaw_error = desired_yaw - self._yaw
            yaw_error = self.normalize_angle(yaw_error)

            # 2) Compute time delta
            current_time = rospy.Time.now().to_sec()
            dt = current_time - self._time_prev
            if dt <= 0:
                dt = 0.01
            self._time_prev = current_time

            # 3) State machine
            if self._state == self.TURNING:
                # PD turning
                p_term = self._kp_turn * yaw_error
                d_term = self._kd_turn * (yaw_error - self._prev_turn_err) / dt
                turn_output = p_term + d_term
                self._prev_turn_err = yaw_error

                # Limit turn speed
                turn_output = max(min(turn_output, 1.0), -1.0)

                # Publish twist
                twist = Twist()
                twist.angular.z = turn_output
                twist.linear.x = 0.0
                self._pub_cmd_vel.publish(twist)

                # Transition condition
                if abs(yaw_error) < self._yaw_threshold:
                    rospy.loginfo("Now facing target. Switching to STOPPED state.")
                    self._state = self.STOPPED

            elif self._state == self.STOPPED:
                # Short phase to ensure the robot actually stops
                twist = Twist()
                # Could set linear.x and angular.z to 0
                self._pub_cmd_vel.publish(twist)
                # Move to MOVING
                rospy.loginfo("Robot stopped. Switching to MOVING state.")
                self._state = self.MOVING

            elif self._state == self.MOVING:
                # PID for distance
                p_dist = self._kp_dist * distance_error
                self._integral_dist += distance_error * dt
                # Optional anti-windup
                if self._integral_dist > 1.0:
                    self._integral_dist = 1.0
                elif self._integral_dist < -1.0:
                    self._integral_dist = -1.0

                i_dist = self._ki_dist * self._integral_dist

                d_dist = self._kd_dist * (distance_error - self._prev_dist_err) / dt
                self._prev_dist_err = distance_error

                lin_output = p_dist + i_dist + d_dist
                # Limit linear speed
                lin_output = max(min(lin_output, 1.0), -1.0)
                if lin_output < 0:
                    lin_output = 0.0  # Disallow reverse, if needed

                # We could also do a small PD-turn here to keep orientation
                # but let's keep it simple: if angle gets big, go back to TURNING
                if abs(yaw_error) > 0.15:  # if angle error grows > ~8.6 deg
                    rospy.loginfo("Yaw error too large, switching back to TURNING")
                    self._state = self.TURNING
                    continue

                twist = Twist()
                twist.linear.x = lin_output
                # Optional small correction for yaw (e.g. P-only)
                yaw_correction = 0.3 * yaw_error
                yaw_correction = max(min(yaw_correction, 0.5), -0.5)
                twist.angular.z = yaw_correction

                self._pub_cmd_vel.publish(twist)

            # Feedback
            self._feedback.position = self._position
            # Just store the name of the current state for feedback
            if self._state == self.TURNING:
                self._feedback.state = "TURNING"
            elif self._state == self.STOPPED:
                self._feedback.state = "STOPPED"
            else:
                self._feedback.state = "MOVING"

            self._as.publish_feedback(self._feedback)

            self._rate.sleep()

        # Stop the robot
        twist = Twist()
        self._pub_cmd_vel.publish(twist)

        if success:
            self._result.success = True
            self._as.set_succeeded(self._result)


if __name__ == "__main__":
    rospy.init_node("tortoisebot_as")
    server = WaypointActionClass()
    rospy.spin()
