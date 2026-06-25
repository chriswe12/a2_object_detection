from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    EnvironmentVariable,
    PathJoinSubstitution,
    PythonExpression,
    TextSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Declare all launch arguments
    declared_arguments = [
        DeclareLaunchArgument(
            "gpu",
            default_value="off",
            description="Run on GPU? Options: 'local', 'remote' (default), 'off'",
            choices=["local", "remote", "off"],
        ),
        DeclareLaunchArgument(
            "GPU_user",
            default_value=EnvironmentVariable("USER"),
            description="Username to use on the jetson xavier GPU",
        ),
        DeclareLaunchArgument(
            "input_camera_name",
            default_value="/camera",
            description="Name of the camera, i.e. topic prefix for camera stream and camera info",
        ),
        DeclareLaunchArgument(
            "debayer_image",
            default_value="false",
            description="Debayer the images (supplied in $input_camera_name/image_raw)",
        ),
        DeclareLaunchArgument(
            "lidar_topic",
            default_value="/front_lidar/points",
            description="Topic containing the point cloud from the lidar",
        ),
        DeclareLaunchArgument(
            "object_detection_classes",
            default_value="[11, 24, 25, 39, 74]",
            description="List of the ids of classes for detection (COCO dataset)",
        ),
        DeclareLaunchArgument(
            "model_dir_path",
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("object_detection"),
                    "models",
                ]
            ),
            description="path to the yolo model directory",
        ),
        DeclareLaunchArgument(
            "model", default_value="yolov5l6", description="yolo model name"
        ),
        DeclareLaunchArgument(
            "log_detections",
            default_value="true",
            description="Write a JSONL detection log for post-mission SLAM fusion",
            choices=["true", "false"],
        ),
        DeclareLaunchArgument(
            "log_file_path",
            default_value="",
            description="Path for the JSONL log (empty = ~/object_detections_<timestamp>.jsonl)",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use /clock (bag time) instead of wall time. MUST be true "
            "when replaying a bag, or the deferred map-frame fusion ages every "
            "detection out and no global object map is written.",
            choices=["true", "false"],
        ),
    ]

    # Shared tuning params (algorithm/clustering/output topics). Single source of
    # truth; deployment-specific params below override it.
    config_file = PathJoinSubstitution(
        [FindPackageShare("object_detection"), "config", "object_detection.yaml"]
    )

    # Object detection node
    object_detection_group = GroupAction(
        [
            Node(
                package="object_detection",
                executable="object_detection_node.py",
                name="object_detector",
                output="screen",
                parameters=[
                    config_file,
                    # --- deployment-specific (override the shared config) ---
                    {
                        "camera_topic": PathJoinSubstitution(
                            [LaunchConfiguration("input_camera_name"), "image_raw"]
                        )
                    },
                    {
                        "camera_info_topic": PathJoinSubstitution(
                            [LaunchConfiguration("input_camera_name"), "camera_info"]
                        )
                    },
                    {"lidar_topic": LaunchConfiguration("lidar_topic")},
                    {"model": LaunchConfiguration("model")},
                    {"model_dir_path": LaunchConfiguration("model_dir_path")},
                    {"device": "cpu"},
                    {"iou": 0.0},  # sim override; confident/iou baseline in config
                    {"classes": LaunchConfiguration("object_detection_classes")},
                    {"log_detections": PythonExpression(
                        ["'", LaunchConfiguration("log_detections"), "' == 'true'"]
                    )},
                    {"log_file_path": LaunchConfiguration("log_file_path")},
                    {"use_sim_time": PythonExpression(
                        ["'", LaunchConfiguration("use_sim_time"), "' == 'true'"]
                    )},
                ],
            )
        ]
    )

    return LaunchDescription(
        declared_arguments + [object_detection_group]
    )
