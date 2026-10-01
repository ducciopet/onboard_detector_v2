#!/usr/bin/env python3

"""PointPillars 3D object detection on raw /velodyne_points — the detection
phase's first model, using OpenPCDet's own PointPillars implementation +
its official nuScenes-pretrained checkpoint (PointPillar-MultiHead,
cbgs_pp_multihead.yaml, 10 classes: car, truck, construction_vehicle, bus,
trailer, barrier, motorcycle, bicycle, pedestrian, traffic_cone).

Why a vendored/patched OpenPCDet instead of pip-installing it normally: this
container's torch (2.14.0+cu130) and numpy (2.5.3, pinned for ultralytics)
are both far newer than anything OpenPCDet or its `spconv` dependency has
ever targeted — there is no spconv wheel for CUDA 13.x at all, and stock
OpenPCDet uses numpy aliases (np.float/np.int/np.bool) removed in numpy>=1.24.
PointPillars itself, unlike SECOND/PV-RCNN/CenterPoint etc. in the same
OpenPCDet model zoo, never actually needs spconv (no BACKBONE_3D in its own
config — see cbgs_pp_multihead.yaml: only VFE -> MAP_TO_BEV -> BACKBONE_2D ->
DENSE_HEAD, all plain dense ops), so the fix applied to the vendored copy at
/home/ros/thirdparty/OpenPCDet is: (1) guard every spconv/pointnet2/dsvt/
bev_pool import in the models package's aggregator __init__.py files with
try/except, so only architectures whose ops actually built survive into each
registry — PointPillar/PillarVFE/PointPillarScatter/BaseBEVBackbone/
AnchorHeadMulti all import cleanly this way; (2) patch the handful of
np.float/np.int/np.bool aliases for numpy 2.x; (3) trim setup.py's
ext_modules down to just iou3d_nms_cuda (rotated-box NMS) and
roiaware_pool3d_cuda (needed transitively by pcdet/utils/box_utils.py's own
import, regardless of model) — both small, self-contained CUDAExtensions
that compiled cleanly against this system's torch/CUDA, unlike spconv or the
pointnet2/roipoint_pool3d ops PointPillars never calls. Checkpoint loads with
an EXACT (strict=True) match against this patched model — see this session's
own validation script, not reproduced here.

Voxelization (raw points -> pillars): OpenPCDet's own VoxelGeneratorWrapper
(pcdet/datasets/processor/data_processor.py) ALSO needs spconv (its CPU
binning goes through spconv.utils.Point2VoxelCPU3d / cumm), so it can't be
reused either — points_to_pillars() below is a from-scratch, vectorized
PyTorch reimplementation of the same "pillar" concept the nuScenes config
describes (VOXEL_SIZE [0.2, 0.2, 8.0]: a flat 2D grid in XY, one bin spanning
the ENTIRE Z range per cell — that's what makes it a "pillar" and not a
voxel), same MAX_POINTS_PER_VOXEL/MAX_NUMBER_OF_VOXELS truncation semantics,
not a bit-exact port of the reference scan-order behavior (see its own
docstring for the exact deviation).

Why raw /velodyne_points, not a fused/preprocessed cloud: preprocessing_node.cpp
used to also publish a fused_cloud (LiDAR + every camera's depth, voxel-merged,
floor/wall-cropped, XYZ-only) meant as this node's input — empirically
compared against raw /velodyne_points this session and dropped entirely
(removed from preprocessing_node.cpp) once the comparison was conclusive: raw
LiDAR with its REAL intensity+per-point-time (use_point_features true, see
below) produced far more, and far more plausible, detections (e.g.
human-scale "pedestrian" boxes landing on real point clusters) than
fused_cloud's zero-padded features ever did — likely some combination of real
intensity/time, a much denser single-sweep cloud (~28k points vs
fused_cloud's ~5-7k after crop+voxel-merge), and a more stable rigid frame
transform (static TF vs a drifting odometry pose, see "Frame handling" below).

Known accuracy caveat, NOT resolved by switching off fused_cloud: this
checkpoint was trained on 5-feature nuScenes points (x, y, z, intensity,
timestamp) accumulated across MULTIPLE sweeps with meaningful per-point
"time since keyframe" values — feeding it a single sweep (points_to_pillars'
point_features here is real intensity + per-point offset WITHIN this one
sweep, not nuScenes' own multi-sweep semantics) is still out-of-distribution
for that specific training assumption, on top of the street-vehicle class
set not matching this robot's actual environment. Treat detections as a
"does the pipeline work, and work reasonably" validation, not yet a tuned/
trusted detector — see this session's own PointPillars-input research for
the fuller picture (KITTI single-sweep checkpoint, fine-tuning, ...).

Frame handling: /velodyne_points arrives in a FIXED sensor frame (velodyne),
but the checkpoint was trained on EGO-CENTERED point clouds (nuScenes' own
POINT_CLOUD_RANGE is symmetric around the sensor, [-51.2, 51.2] on both X/Y)
— anchors are placed in a grid around the origin, so points need to land in
base_frame_ (ego), not the sensor's own frame, before voxelizing.
cloud_frame_mode='static_tf' (the default — see cfg/<env>/pointpillars.yaml)
resolves base_frame_ -> the cloud's own frame_id ONCE via TF and caches it
forever (rigid mount, see _resolve_static_base_sensor_tf) — this TF IS
reliably broadcast, unlike global_frame_ -> base_frame_ (confirmed
empirically this session: a tf_buffer.lookup_transform('base_link', 'odom',
...) here returned "'odom' passed to lookupTransform argument source_frame
does not exist" continuously while the pipeline was otherwise healthy).
cloud_frame_mode='global_pose' is the alternative for a cloud that instead
arrives already in global_frame_ (odom) — e.g. a future re-introduced fused/
accumulated cloud — using the cached odom_topic pose instead of a TF lookup,
same reasoning as global_frame_ -> base_frame_'s own unreliability above.
Either way, detections are transformed back into global_frame_ before
publishing — every OTHER output in this package (GSeg3D,
semantic_ground_points) is already published in global_frame_, so detections
stay consistent with that convention rather than introducing a third frame
into rviz. odom_topic (nav_msgs/Odometry) is therefore always subscribed,
regardless of cloud_frame_mode — needed for this output-side transform even
when cloud_frame_mode='static_tf' doesn't need it for the input side.

Single process, not one-per-camera: unlike the YOLO nodes (one GPU-bound
model instance per camera, for real parallelism across independent camera
streams — see yolo_seg_track_detector.py's own header), PointPillars
operates on ONE cloud covering the whole scene at once, so there is exactly
one of these regardless of camera_names.

use_point_features=true reads REAL intensity + per-point time (both present
on the raw Velodyne driver's own PointCloud2 fields) into voxel features 3/4
instead of zero-padding — see points_to_pillars' point_features argument.
Velodyne intensity is already a 0-255 calibrated-reflectivity float, matching
nuScenes' own raw units, so no rescaling is applied.
"""

import os
import sys
import threading
import time
from collections import deque

import numpy as np
import torch
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException
from vision_msgs.msg import BoundingBox3D, Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

try:
    import sensor_msgs_py.point_cloud2 as pc2
except ImportError:
    pc2 = None

device = "cuda" if torch.cuda.is_available() else "cpu"


def _quat_from_yaw(yaw):
    half = yaw * 0.5
    return (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))


def _odom_to_matrix(odom: Odometry):
    """4x4 homogeneous T_global_base (numpy) from a nav_msgs/Odometry — same
    quantity preprocessing_node.cpp's own odomToGlobalBase() computes, same
    reasoning for using the message directly instead of TF (see file
    header)."""
    p = odom.pose.pose.position
    q = odom.pose.pose.orientation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = [p.x, p.y, p.z]
    return M


def _tf_to_matrix(transform):
    """4x4 homogeneous transform (numpy) from a geometry_msgs/TransformStamped
    — used only for cloud_frame_mode='static_tf' (see file header): a RIGID,
    one-time lookup, unlike the odom-pose-based path above."""
    t = transform.transform.translation
    q = transform.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = [t.x, t.y, t.z]
    return M


def points_to_pillars(points_xyz, point_cloud_range, voxel_size, grid_size,
                       max_points_per_voxel, max_voxels, device, point_features=None):
    """Group points_xyz (N,3 torch float tensor, ego frame) into pillars —
    see file header for why this can't just reuse OpenPCDet's own
    VoxelGeneratorWrapper (also spconv-dependent).

    Returns (voxels [M, max_points_per_voxel, 5], voxel_num_points [M],
    voxel_coords [M, 4] as [batch_idx, z=0, y, x]) — the exact tensors
    PillarVFE/PointPillarScatter expect. point_features, when given, is an
    (N, 2) tensor of [intensity, timestamp] per point (same N/order as
    points_xyz) copied into voxel features 3/4; left zero-padded when None
    (use_point_features=false, or a cloud_topic without those fields — see
    file header). M <= max_voxels; a voxel's own point list is truncated at
    max_points_per_voxel, matching MAX_POINTS_PER_VOXEL/MAX_NUMBER_OF_VOXELS
    in cbgs_pp_multihead.yaml's own DATA_PROCESSOR config.
    """
    pc_range = torch.as_tensor(point_cloud_range, dtype=torch.float32, device=device)
    vsize = torch.as_tensor(voxel_size, dtype=torch.float32, device=device)
    gsize = torch.as_tensor(grid_size, dtype=torch.int64, device=device)

    in_range = (
        (points_xyz[:, 0] >= pc_range[0]) & (points_xyz[:, 0] < pc_range[3]) &
        (points_xyz[:, 1] >= pc_range[1]) & (points_xyz[:, 1] < pc_range[4]) &
        (points_xyz[:, 2] >= pc_range[2]) & (points_xyz[:, 2] < pc_range[5])
    )
    pts = points_xyz[in_range]
    feats = point_features[in_range] if point_features is not None else None
    if pts.shape[0] == 0:
        empty = torch.zeros((0, max_points_per_voxel, 5), dtype=torch.float32, device=device)
        return empty, torch.zeros((0,), dtype=torch.int32, device=device), \
            torch.zeros((0, 4), dtype=torch.int32, device=device)

    ix = torch.floor((pts[:, 0] - pc_range[0]) / vsize[0]).long().clamp_(0, gsize[0].item() - 1)
    iy = torch.floor((pts[:, 1] - pc_range[1]) / vsize[1]).long().clamp_(0, gsize[1].item() - 1)
    flat_id = iy * gsize[0] + ix

    unique_ids, inverse, counts = torch.unique(flat_id, return_inverse=True, return_counts=True)
    num_voxels = unique_ids.shape[0]
    if num_voxels > max_voxels:
        # Deterministic truncation (not the reference's scan-order-first
        # behavior — see file header) — keep the max_voxels most-populated
        # cells, on the theory that a near-empty pillar contributes least.
        keep_order = torch.argsort(counts, descending=True)[:max_voxels]
        keep_mask = torch.zeros(num_voxels, dtype=torch.bool, device=device)
        keep_mask[keep_order] = True
        point_keep = keep_mask[inverse]
        pts = pts[point_keep]
        if feats is not None:
            feats = feats[point_keep]
        inverse = inverse[point_keep]
        # Remap surviving voxel ids to a dense [0, max_voxels) range.
        remap = torch.full((num_voxels,), -1, dtype=torch.long, device=device)
        remap[keep_order] = torch.arange(max_voxels, device=device)
        inverse = remap[inverse]
        unique_ids = unique_ids[keep_order]
        num_voxels = max_voxels

    # Per-point slot within its own voxel: sort points by voxel id (stable),
    # then a reset-on-change running counter gives each point's 0-based
    # position among points sharing the same voxel — i.e. index within each
    # contiguous run of equal sorted_voxel values.
    order = torch.argsort(inverse, stable=True)
    sorted_voxel = inverse[order]
    same_as_prev = torch.zeros_like(sorted_voxel, dtype=torch.bool)
    same_as_prev[1:] = sorted_voxel[1:] == sorted_voxel[:-1]
    reset_points = (~same_as_prev).nonzero(as_tuple=True)[0]

    group_start = torch.zeros_like(sorted_voxel)
    group_start[reset_points] = 1
    group_id = torch.cumsum(group_start, dim=0) - 1
    run_start_pos = torch.zeros(group_id.max().item() + 1, dtype=torch.long, device=device)
    run_start_pos[group_id[reset_points]] = reset_points
    # first index of each point's own run:
    first_idx_of_point = run_start_pos[group_id]
    slot_sorted = torch.arange(sorted_voxel.shape[0], device=device) - first_idx_of_point

    slot = torch.empty_like(slot_sorted)
    slot[order] = slot_sorted
    valid = slot < max_points_per_voxel

    voxels = torch.zeros((num_voxels, max_points_per_voxel, 5), dtype=torch.float32, device=device)
    v_idx = inverse[valid]
    s_idx = slot[valid]
    voxels[v_idx, s_idx, 0:3] = pts[valid]
    if feats is not None:
        voxels[v_idx, s_idx, 3:5] = feats[valid]
    # else: voxels[..., 3] (intensity) / voxels[..., 4] (timestamp) left at 0.

    voxel_num_points = torch.zeros(num_voxels, dtype=torch.int32, device=device)
    voxel_num_points.scatter_add_(0, v_idx, torch.ones_like(v_idx, dtype=torch.int32))

    gy = torch.div(unique_ids, gsize[0], rounding_mode='floor')
    gx = unique_ids - gy * gsize[0]
    batch_idx = torch.zeros((num_voxels,), dtype=torch.int32, device=device)
    gz = torch.zeros((num_voxels,), dtype=torch.int32, device=device)
    voxel_coords = torch.stack([batch_idx, gz, gy.int(), gx.int()], dim=1)

    return voxels, voxel_num_points, voxel_coords


class PointPillarsDetector(Node):
    def __init__(self):
        super().__init__('pointpillars_detector')
        self.get_logger().info("[onboardDetectorV2]: pointpillars detector init...")

        self.cloud_topic = str(self.declare_parameter('cloud_topic', '/velodyne_points').value)
        self.odom_topic = str(self.declare_parameter('odom_topic', '/odometry/filtered').value)
        # See file header's own "Frame handling" section. 'static_tf'
        # (default): cloud_topic is in a fixed sensor frame (raw
        # /velodyne_points), transformed via a ONE-TIME TF lookup.
        # 'global_pose': cloud_topic is already in global_frame_, transformed
        # via the cached odom_topic pose instead — for a cloud that isn't a
        # raw sensor topic (e.g. some future re-introduced fused cloud).
        self.cloud_frame_mode = str(self.declare_parameter('cloud_frame_mode', 'static_tf').value)
        # Read real intensity/time fields off cloud_topic's own PointCloud2
        # (only meaningful for a raw sensor topic that actually has them,
        # e.g. /velodyne_points) instead of zero-padding voxel features 3/4.
        self.use_point_features = bool(self.declare_parameter('use_point_features', True).value)
        self.detections_topic = str(self.declare_parameter('detections_topic', '').value)
        self.markers_topic = str(self.declare_parameter('markers_topic', '').value)
        self.inference_time_topic = str(self.declare_parameter('inference_time_topic', '').value)
        self.global_frame = str(self.declare_parameter('global_frame', 'odom').value)
        self.base_frame = str(self.declare_parameter('base_frame', 'base_link').value)

        self.openpcdet_path = str(self.declare_parameter('openpcdet_path', '/home/ros/thirdparty/OpenPCDet').value)
        self.config_path = str(self.declare_parameter(
            'config_path', 'tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml').value)
        self.checkpoint_path = str(self.declare_parameter(
            'checkpoint_path', '/home/ros/thirdparty/checkpoints/pointpillar_nuscenes_cbgs_multihead.pth').value)
        self.score_thresh = float(self.declare_parameter('score_thresh', 0.3).value)
        self.max_voxels = int(self.declare_parameter('max_voxels', 30000).value)
        self.queue_size = int(self.declare_parameter('queue_size', 5).value)
        self.stats_log_period_sec = float(self.declare_parameter('stats_log_period_sec', 5.0).value)

        missing = [
            name for name, val in [
                ('detections_topic', self.detections_topic),
                ('markers_topic', self.markers_topic),
                ('inference_time_topic', self.inference_time_topic),
            ] if not val
        ]
        if missing:
            self.get_logger().error('Missing required parameters in YAML: ' + ', '.join(missing))
            raise RuntimeError('PointPillars detector parameter initialization failed')

        if pc2 is None:
            raise RuntimeError('sensor_msgs_py.point_cloud2 not importable — cannot decode cloud_topic')

        sys.path.insert(0, self.openpcdet_path)
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.models import build_network

        cfg_path_abs = os.path.join(self.openpcdet_path, self.config_path)
        # OpenPCDet's own cfg_from_yaml_file resolves _BASE_CONFIG_ relative
        # to the CURRENT WORKING DIRECTORY (its own tools/train.py is always
        # run from inside tools/) — replicate that here rather than patch
        # the vendored file.
        prev_cwd = os.getcwd()
        os.chdir(os.path.join(self.openpcdet_path, 'tools'))
        try:
            cfg_from_yaml_file(cfg_path_abs, cfg)
        finally:
            os.chdir(prev_cwd)

        self.class_names = list(cfg.CLASS_NAMES)
        self.point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
        voxel_stage = next(d for d in cfg.DATA_CONFIG.DATA_PROCESSOR if d.NAME == 'transform_points_to_voxels')
        self.voxel_size = np.array(voxel_stage.VOXEL_SIZE, dtype=np.float32)
        self.grid_size = np.round(
            (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size).astype(np.int64)
        self.max_points_per_voxel = int(voxel_stage.MAX_POINTS_PER_VOXEL)

        from types import SimpleNamespace
        dataset_stub = SimpleNamespace(
            class_names=self.class_names,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            depth_downsample_factor=None,
            point_feature_encoder=SimpleNamespace(num_point_features=5),
        )
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(self.class_names), dataset=dataset_stub)
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'], strict=True)
        self.model.to(device)
        self.model.eval()
        self.get_logger().info(
            f"PointPillars configured: device={device}, classes={self.class_names}, "
            f"point_cloud_range={self.point_cloud_range.tolist()}, voxel_size={self.voxel_size.tolist()}, "
            f"grid_size={self.grid_size.tolist()}, checkpoint={self.checkpoint_path}"
        )

        self.det_pub = self.create_publisher(Detection3DArray, self.detections_topic, self.queue_size)
        self.marker_pub = self.create_publisher(MarkerArray, self.markers_topic, self.queue_size)
        self.time_pub = self.create_publisher(Float64, self.inference_time_topic, 1)

        self.lock = threading.Lock()
        self.new_cloud_event = threading.Event()
        self.pending_cloud_msg = None
        self.first_logged = False
        self.first_odom_logged = False
        self.inference_times = deque(maxlen=200)
        self.frames_received = 0
        self.frames_dropped = 0
        self.last_stats_log = time.time()

        # Latest T_global_base, cached from odom_topic — see file header on
        # why this is an Odometry-message subscription, not a TF lookup.
        # has_odom_ guards the very first cycle(s) before anything arrives.
        # Needed regardless of cloud_frame_mode: detections are always
        # published in global_frame_ (see _process_cloud/_publish_detections).
        self.T_global_base_latest = np.eye(4, dtype=np.float64)
        self.has_odom = False

        # cloud_frame_mode='static_tf' only: T_base_sensor is resolved ONCE
        # (rigid mount — see file header) on the first cloud message and
        # cached forever after, keyed by that message's own frame_id so a
        # frame_id change (shouldn't happen mid-run) re-triggers the lookup.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.T_base_sensor = None
        self.T_base_sensor_frame_id = None

        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, self.queue_size)
        self.create_subscription(PointCloud2, self.cloud_topic, self._cloud_callback, self.queue_size)

        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, name='pointpillars_worker', daemon=True)
        self.worker_thread.start()

    def destroy_node(self):
        self._stop_event.set()
        self.new_cloud_event.set()
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=2.0)
        super().destroy_node()

    def _odom_callback(self, msg):
        # No lock: single 8-byte-aligned numpy array assignment from one
        # publisher's callback, read by the worker thread — same "latest
        # wins, staleness is fine" tolerance preprocessing_node.cpp's own
        # T_global_base_latest_ has (see file header).
        self.T_global_base_latest = _odom_to_matrix(msg)
        self.has_odom = True
        if not self.first_odom_logged:
            if msg.child_frame_id and msg.child_frame_id != self.base_frame:
                self.get_logger().warning(
                    f"odom_topic '{self.odom_topic}' child_frame_id is '{msg.child_frame_id}', "
                    f"not base_frame '{self.base_frame}' — points will be transformed as if this "
                    "pose WERE base_frame's; fix base_frame if that's wrong.")
            self.get_logger().info(f"First odometry received on {self.odom_topic}")
            self.first_odom_logged = True

    def _cloud_callback(self, msg):
        with self.lock:
            self.frames_received += 1
            if self.pending_cloud_msg is not None:
                self.frames_dropped += 1
            self.pending_cloud_msg = msg
            self.new_cloud_event.set()
        if not self.first_logged:
            self.get_logger().info(f"First cloud received on {self.cloud_topic} ({msg.width} points, frame={msg.header.frame_id})")
            self.first_logged = True

    def _worker_loop(self):
        while not self._stop_event.is_set():
            if not self.new_cloud_event.wait(timeout=1.0):
                continue
            with self.lock:
                msg = self.pending_cloud_msg
                self.pending_cloud_msg = None
                self.new_cloud_event.clear()
            if msg is None or self._stop_event.is_set():
                continue
            self._process_cloud(msg)

    def _maybe_log_stats(self):
        now = time.time()
        if now - self.last_stats_log < self.stats_log_period_sec:
            return
        with self.lock:
            received, dropped = self.frames_received, self.frames_dropped
        times = list(self.inference_times)
        self.last_stats_log = now
        if not times:
            return
        avg_ms = 1000.0 * sum(times) / len(times)
        max_ms = 1000.0 * max(times)
        drop_rate = (100.0 * dropped / received) if received else 0.0
        self.get_logger().info(
            f"inference: avg={avg_ms:.1f}ms max={max_ms:.1f}ms | "
            f"frames received={received} dropped={dropped} ({drop_rate:.1f}%)"
        )

    def _resolve_static_base_sensor_tf(self, frame_id):
        """cloud_frame_mode='static_tf' only — resolve+cache base_frame_ ->
        frame_id ONCE (rigid mount, see file header); returns the cached
        matrix on every later call for the same frame_id, or None while
        still unavailable/on a genuine frame_id change."""
        if self.T_base_sensor is not None and self.T_base_sensor_frame_id == frame_id:
            return self.T_base_sensor
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, frame_id, Time())
        except (LookupException, ExtrapolationException) as ex:
            self.get_logger().warning(f"TF {frame_id} -> {self.base_frame} unavailable: {ex}")
            return None
        self.T_base_sensor = _tf_to_matrix(tf)
        self.T_base_sensor_frame_id = frame_id
        self.get_logger().info(f"Cached static TF {self.base_frame} -> {frame_id}")
        return self.T_base_sensor

    def _process_cloud(self, msg):
        start_time = time.time()
        try:
            if self.use_point_features:
                arr = pc2.read_points(msg, field_names=('x', 'y', 'z', 'intensity', 'time'), skip_nans=True)
                points_np = np.stack([arr['x'], arr['y'], arr['z']], axis=-1).astype(np.float32)
                feats_np = np.stack([arr['intensity'], arr['time']], axis=-1).astype(np.float32)
            else:
                arr = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
                points_np = np.stack([arr['x'], arr['y'], arr['z']], axis=-1).astype(np.float32)
                feats_np = None
            if points_np.shape[0] == 0:
                return

            # Always needed: detections are published in global_frame_
            # regardless of cloud_frame_mode — see _publish_detections.
            if not self.has_odom:
                self.get_logger().warning(f"No odometry received yet on {self.odom_topic} — skipping frame")
                return
            T_global_base = self.T_global_base_latest

            # Transform cloud points into base_frame_ (ego) before
            # voxelizing, since the model's anchors assume an ego-centered
            # point cloud — see file header for both branches below.
            if self.cloud_frame_mode == 'static_tf':
                T_base_source = self._resolve_static_base_sensor_tf(msg.header.frame_id)
                if T_base_source is None:
                    return
            else:
                T_base_source = np.linalg.inv(T_global_base)

            points_h = np.concatenate([points_np, np.ones((points_np.shape[0], 1), dtype=np.float32)], axis=1)
            points_base = (T_base_source @ points_h.T).T[:, :3].astype(np.float32)

            points_t = torch.from_numpy(points_base).to(device)
            feats_t = torch.from_numpy(feats_np).to(device) if feats_np is not None else None
            voxels, voxel_num_points, voxel_coords = points_to_pillars(
                points_t, self.point_cloud_range, self.voxel_size, self.grid_size,
                self.max_points_per_voxel, self.max_voxels, device, point_features=feats_t)

            if voxels.shape[0] == 0:
                self.get_logger().warning("No points inside point_cloud_range this frame — skipping inference")
                return

            batch_dict = {
                'batch_size': 1,
                'voxels': voxels,
                'voxel_num_points': voxel_num_points,
                'voxel_coords': voxel_coords,
            }
            with torch.no_grad():
                pred_dicts, _ = self.model.forward(batch_dict)

            # Predictions come back in base_frame_ (ego) — same frame the
            # input points were transformed into above; transform box
            # centers/headings back into global_frame_ (T_global_base,
            # already computed above) for publishing, so rviz sees them in
            # the SAME fixed frame as semantic_ground_points/GSeg3D outputs.
            self._publish_detections(pred_dicts[0], msg.header, T_global_base)
        except Exception:
            import traceback
            self.get_logger().error(f"Error in inference:\n{traceback.format_exc()}")
        elapsed = time.time() - start_time
        time_msg = Float64()
        time_msg.data = float(elapsed)
        self.time_pub.publish(time_msg)
        self.inference_times.append(elapsed)
        self._maybe_log_stats()

    def _publish_detections(self, pred_dict, header, T_global_base):
        boxes = pred_dict['pred_boxes'].detach().cpu().numpy()
        scores = pred_dict['pred_scores'].detach().cpu().numpy()
        labels = pred_dict['pred_labels'].detach().cpu().numpy()

        keep = scores >= self.score_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        out_header = header
        out_header.frame_id = self.global_frame

        det_array = Detection3DArray()
        det_array.header = out_header

        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.header = out_header
        # Empty ns (not 'pointpillars_bbox') — DELETEALL only clears markers
        # in the SAME namespace as the marker carrying it, and this array
        # publishes two ('pointpillars_bbox' CUBEs + 'pointpillars_label'
        # TEXT_VIEW_FACINGs); a namespaced DELETEALL here would leave stale
        # labels behind whenever the detection count drops between frames.
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for i in range(boxes.shape[0]):
            x, y, z, dx, dy, dz, heading = boxes[i, :7]
            p_base = np.array([x, y, z, 1.0], dtype=np.float64)
            p_global = T_global_base @ p_base
            yaw_base = float(heading)
            # Only the rotation part of T_global_base affects heading.
            yaw_global = yaw_base + float(np.arctan2(T_global_base[1, 0], T_global_base[0, 0]))
            qx, qy, qz, qw = _quat_from_yaw(yaw_global)
            class_id = self.class_names[int(labels[i]) - 1] if 1 <= int(labels[i]) <= len(self.class_names) \
                else str(int(labels[i]))

            det = Detection3D()
            det.header = out_header
            det.id = f"{i}"
            bbox = BoundingBox3D()
            bbox.center.position.x = float(p_global[0])
            bbox.center.position.y = float(p_global[1])
            bbox.center.position.z = float(p_global[2])
            bbox.center.orientation.x = qx
            bbox.center.orientation.y = qy
            bbox.center.orientation.z = qz
            bbox.center.orientation.w = qw
            bbox.size.x = float(dx)
            bbox.size.y = float(dy)
            bbox.size.z = float(dz)
            det.bbox = bbox
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = class_id
            hyp.hypothesis.score = float(scores[i])
            det.results.append(hyp)
            det_array.detections.append(det)

            m = Marker()
            m.header = out_header
            m.ns = 'pointpillars_bbox'
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(p_global[0])
            m.pose.position.y = float(p_global[1])
            m.pose.position.z = float(p_global[2])
            m.pose.orientation.x = qx
            m.pose.orientation.y = qy
            m.pose.orientation.z = qz
            m.pose.orientation.w = qw
            m.scale.x = float(dx)
            m.scale.y = float(dy)
            m.scale.z = float(dz)
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 0.5
            m.lifetime = Duration(seconds=0.5).to_msg()
            marker_array.markers.append(m)

            text = Marker()
            text.header = out_header
            text.ns = 'pointpillars_label'
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(p_global[0])
            text.pose.position.y = float(p_global[1])
            text.pose.position.z = float(p_global[2]) + float(dz) / 2.0 + 0.2
            text.scale.z = 0.3
            text.color.r, text.color.g, text.color.b, text.color.a = 1.0, 1.0, 1.0, 1.0
            text.text = f"{class_id} {scores[i]:.2f}"
            text.lifetime = Duration(seconds=0.5).to_msg()
            marker_array.markers.append(text)

        self.det_pub.publish(det_array)
        self.marker_pub.publish(marker_array)
