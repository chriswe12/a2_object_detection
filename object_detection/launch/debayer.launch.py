from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    EnvironmentVariable,
    PathJoinSubstitution,
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
            default_value="local",
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
            default_value="/rgb_camera",
            description="Name of the camera, i.e. topic prefix for camera stream and camera info",
        ),
        DeclareLaunchArgument(
            "debayer_image",
            default_value="true",
            description="Debayer the images (supplied in $input_camera_name/image_raw)",
        ),
        DeclareLaunchArgument(
            "lidar_topic",
            default_value="/rslidar/points",
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
    ]

    # Debayer the image (conditionally included)
    debayer_image_group = GroupAction(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("object_detection"),
                            "launch",
                            "debayer.launch.py",
                        ]
                    )
                ),
                condition=IfCondition(LaunchConfiguration("debayer_image")),
                launch_arguments={
                    "input_camera_name": LaunchConfiguration("input_camera_name")
                },
            )
        ]
    )

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
                    {"device": "0" if LaunchConfiguration("gpu") != "off" else "cpu"},
                    {"classes": LaunchConfiguration("object_detection_classes")},
                ],
            )
        ]
    )

    return LaunchDescription(
        declared_arguments + [debayer_image_group, object_detection_group]
    )