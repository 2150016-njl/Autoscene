#!/usr/bin/env python3
"""Convert external ego feedback into the AutoScene world frame.

This node is started only when ``ego.execution_mode=external``. Simulation ego
state is consumed directly by scenario_runtime_node and is not echoed through an
adapter.
"""
import copy
import math

import rospy
from std_msgs.msg import String

from autoscene_msgs.msg import VehicleState


def wrap_rad(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quat(yaw):
    from geometry_msgs.msg import Quaternion

    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def rotate_xy(x, y, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * x - s * y, s * x + c * y


class EgoAdapterNode:
    def __init__(self):
        rospy.init_node("ego_adapter_node")

        self.frame_id = rospy.get_param("/runtime/frame_id", "map")
        self.ego_cfg = rospy.get_param("/ego", {})
        execution_mode = str(
            self.ego_cfg.get("execution_mode", "simulation")
        ).strip().lower()
        if execution_mode != "external":
            raise ValueError(
                "ego_adapter_node is only valid for ego.execution_mode=external"
            )

        self.ego_id = str(self.ego_cfg.get("id", "ego_001"))
        self.length = float(self.ego_cfg.get("length", 4.8))
        self.width = float(self.ego_cfg.get("width", 2.0))
        self.external_timeout_ms = int(
            self.ego_cfg.get("external_timeout_ms", 300)
        )

        origin = self.ego_cfg.get("origin")
        if not isinstance(origin, dict):
            raise ValueError("ego.origin is required in external mode")
        self.origin_x = float(origin.get("x", 0.0))
        self.origin_y = float(origin.get("y", 0.0))
        self.origin_yaw = float(
            origin.get("yaw", origin.get("heading", 0.0))
        )

        self.last_external_stamp = None
        self.last_status = "waiting_external_ego_state"

        self.state_feedback_pub = rospy.Publisher(
            "ego/state_feedback",
            VehicleState,
            queue_size=10,
        )
        self.status_pub = rospy.Publisher(
            "ego/adapter_status",
            String,
            queue_size=1,
        )
        rospy.Subscriber(
            "external/ego/state",
            VehicleState,
            self.on_external_ego_state,
            queue_size=10,
        )
        rospy.Timer(rospy.Duration(1.0), self.on_status_timer)

        rospy.loginfo(
            "ego_adapter_node started: external/ego/state -> ego/state_feedback, frame=%s",
            self.frame_id,
        )

    def make_invalid_feedback(self, msg, status):
        state = VehicleState()
        state.header.stamp = rospy.Time.now()
        state.header.frame_id = self.frame_id
        state.vehicle_id = msg.vehicle_id or self.ego_id
        state.entity_type = "real"
        state.length = msg.length or self.length
        state.width = msg.width or self.width
        state.pose.orientation.w = 1.0
        state.valid = False
        self.last_status = status
        return state

    def external_state_to_autoscene(self, msg):
        state = copy.deepcopy(msg)

        dx, dy = rotate_xy(
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            self.origin_yaw,
        )
        heading = wrap_rad(self.origin_yaw + float(msg.heading))

        state.pose.position.x = self.origin_x + dx
        state.pose.position.y = self.origin_y + dy
        state.pose.orientation = yaw_to_quat(heading)
        state.heading = heading

        state.twist.linear.x, state.twist.linear.y = rotate_xy(
            msg.twist.linear.x,
            msg.twist.linear.y,
            self.origin_yaw,
        )
        state.accel.linear.x, state.accel.linear.y = rotate_xy(
            msg.accel.linear.x,
            msg.accel.linear.y,
            self.origin_yaw,
        )

        state.header.stamp = rospy.Time.now()
        state.header.frame_id = self.frame_id
        state.vehicle_id = msg.vehicle_id or self.ego_id
        state.entity_type = "real"
        state.length = msg.length or self.length
        state.width = msg.width or self.width
        state.valid = True
        return state

    def on_external_ego_state(self, msg):
        if not msg.valid:
            self.last_external_stamp = None
            self.state_feedback_pub.publish(
                self.make_invalid_feedback(msg, "invalid_external_ego_state")
            )
            return

        state = self.external_state_to_autoscene(msg)
        self.last_external_stamp = state.header.stamp
        self.last_status = "external_ego_state_ok"
        self.state_feedback_pub.publish(state)

    def external_state_fresh(self):
        if self.last_external_stamp is None:
            return False
        age_ms = (
            rospy.Time.now() - self.last_external_stamp
        ).to_sec() * 1000.0
        return age_ms <= self.external_timeout_ms

    def on_status_timer(self, _event):
        status = (
            self.last_status
            if self.external_state_fresh()
            else "waiting_external_ego_state"
        )
        self.status_pub.publish(String(data=status))

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    EgoAdapterNode().spin()
