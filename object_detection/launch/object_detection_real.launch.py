from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    EnvironmentVariable,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare all launch arguments
    declared_arguments = [
        DeclareLaunchArgument(
            "gpu",
            default_value="off",
            description="Run on GPU? Options: 'local', 'remote', 'off' (default, CPU)",
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
            default_value="false",
            description="Write a JSONL detection log for post-mission SLAM fusion",
            choices=["true", "false"],
        ),
        DeclareLaunchArgument(
            "log_file_path",
            default_value="",
            description="Path for the JSONL log (empty = ~/object_detections_<timestamp>.jsonl)",
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
            launch_arguments=[
                ("input_camera_name", LaunchConfiguration("input_camera_name"))
            ],
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
                    {"device": PythonExpression(
                        ["'cpu' if '", LaunchConfiguration("gpu"), "' == 'off' else '0'"]
                    )},
                    {"classes": LaunchConfiguration("object_detection_classes")},
                    {"log_detections": PythonExpression(
                        ["'", LaunchConfiguration("log_detections"), "' == 'true'"]
                    )},
                    {"log_file_path": LaunchConfiguration("log_file_path")},
                ],
            )
        ]
    )

    return LaunchDescription(
        declared_arguments + [debayer_image_group, object_detection_group]
    )