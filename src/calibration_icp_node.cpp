#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/cache.h>
#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/exceptions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

// Depth<->color factory extrinsic message — needed ONLY when depth_cloud_topic_
// is set (native point-cloud mode, see that param's own comment): a native
// cloud arrives in the depth sensor's own optical frame, not the color
// camera's, and this is what corrects for that. Same message type
// preprocessing_node.cpp already uses for the identical purpose — see that
// file's own NOTE on the one part of this unverified against a real build
// (rotation assumed column-major, librealsense's rs2_extrinsics convention).
#include <realsense2_camera_msgs/msg/extrinsics.hpp>

// gtsam_points-based registration (replaces Open3D). This is the same
// point-cloud registration stack already required by the glim submodule
// (gtsam + gtsam_points), so no extra dependency is introduced.
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam_points/config.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/util/parallelism.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/util/covariance_estimation.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/factors/integrated_icp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

class CalibrationICPNode : public rclcpp::Node {
public:
    CalibrationICPNode() : Node("calibration_icp_node") {
        depth_topic_ = this->declare_parameter(
            "depth_topic", std::string("/front_camera/camera/depth/image_rect_raw"));
        // image_transport plugin used to (de)compress depth_topic_ on the wire.
        // "compressedDepth" matches the live camera/typical recorded bags; bags
        // recorded with the RealSense wrapper's zstd option (e.g. validation_lab,
        // which only has .../aligned_depth_to_color/image_raw/zstd, no
        // .../image_rect_raw/compressedDepth at all) need "zstd" here instead —
        // and depth_topic itself pointed at the matching base topic.
        depth_transport_ = this->declare_parameter("depth_transport", std::string("compressedDepth"));

        // Empty (default) = decode+deproject depth_topic_ ourselves, exactly
        // as before. Non-empty = skip depth_topic_/depth_transport_ entirely
        // and read an already-deprojected PointCloud2 straight from this
        // topic instead (e.g. a RealSense driver's own
        // "<name>/camera/depth/color/points") — mirrors
        // preprocessing_node.cpp's aligned_depth_cloud_topic/
        // onNativeDepthCloud() (see that file's own comments); run_detector.
        // launch.py forwards each camera's own preprocessing.yaml choice
        // here too, so setting it once there is enough to cover every node.
        depth_cloud_topic_ = this->declare_parameter("depth_cloud_topic", std::string(""));
        use_native_cloud_ = !depth_cloud_topic_.empty();

        // Native-cloud mode only: the RealSense factory depth<->color
        // extrinsic, needed to bring a native cloud (arrives in the DEPTH
        // sensor's own optical frame) into the same color-frame-equivalent
        // convention depthToPointCloud()'s deprojection implicitly produces
        // — see buildDepthFrameFromCloud()'s own comment for why this
        // matters here specifically (unlike preprocessing_node.cpp's
        // degraded-but-tolerable fallback, a systematically wrong frame here
        // would bias the calibration result itself).
        depth_to_color_extrinsics_topic_ = this->declare_parameter(
            "depth_to_color_extrinsics_topic", std::string("/front_camera/camera/extrinsics/depth_to_color"));

        // Native-cloud mode only: a PointCloud2 carries no image resolution,
        // but extractLidarOverlapWithDepthFov()'s pixel-space FOV-margin
        // test needs one. 0 (default) = derive it from depth_intrinsics'
        // own principal point (2*cx, 2*cy — always close to the real sensor
        // resolution, and overlap_fov_margin_px_ already tolerates a
        // ~dozen-pixel margin of error); set explicitly only if that
        // approximation proves too coarse for a given camera.
        depth_image_width_param_ = this->declare_parameter("depth_image_width", 0);
        depth_image_height_param_ = this->declare_parameter("depth_image_height", 0);

        lidar_topic_ = this->declare_parameter(
            "velodyne_topic", std::string("/velodyne_points"));

        // Starting values only — overwritten live the moment a message
        // arrives on camera_info_topic_ below (K[0]/K[4]/K[2]/K[5] =
        // fx/fy/cx/cy), same "live source preferred, static param is only
        // the fallback/bootstrap" pattern already used for the native depth
        // cloud and its depth<->color extrinsic above. Kept declared (not
        // removed) so the node still starts up usably if camera_info_topic_
        // is misconfigured or never publishes.
        std::vector<double> depth_intrinsics = this->declare_parameter(
            "depth_intrinsics",
            std::vector<double>{436.9647521972656, 436.9647521972656, 431.9205627441406, 240.13380432128906});
        camera_info_topic_ = this->declare_parameter(
            "camera_info_topic", std::string("/front_camera/camera/color/camera_info"));

        depth_scale_ = this->declare_parameter("depth_scale_factor", 1000.0);
        depth_min_ = this->declare_parameter("depth_min_value", 0.5);
        depth_max_ = this->declare_parameter("depth_max_value", 5.0);
        depth_skip_ = this->declare_parameter("depth_skip_pixel", 2);

        lidar_frame_ = this->declare_parameter("lidar_frame", std::string("velodyne"));
        camera_frame_ = this->declare_parameter("urdf_camera_frame", std::string("camera"));
        camera_frame_initial_guess_ = this->declare_parameter("camera_frame_initial_guess", std::string("camera_initial_guess"));
        refined_camera_frame_ = this->declare_parameter("refined_camera_frame", std::string("camera_refined"));

        icp_max_corr_dist_ = this->declare_parameter("icp_max_correspondence_distance", 0.2);
        icp_max_iter_ = this->declare_parameter("icp_max_iteration", 100);
        icp_voxel_size_ = this->declare_parameter("icp_voxel_size", 0.05);
        icp_use_point_to_plane_ = this->declare_parameter("icp_use_point_to_plane", true);
        icp_normal_radius_ = this->declare_parameter("icp_normal_radius", 0.15);
        icp_normal_max_nn_ = this->declare_parameter("icp_normal_max_nn", 30);
        icp_num_threads_ = this->declare_parameter("icp_num_threads", 4);
        min_overlap_points_ = this->declare_parameter("min_overlap_points", 100);
        overlap_fov_margin_px_ = this->declare_parameter("overlap_fov_margin_px", 120);

        fx_ = depth_intrinsics[0];
        fy_ = depth_intrinsics[1];
        cx_ = depth_intrinsics[2];
        cy_ = depth_intrinsics[3];

        icp_max_runs_ = this->declare_parameter("icp_max_runs", 10);
        icp_fitness_threshold_ = this->declare_parameter("icp_fitness_threshold", 0.5);
        icp_timeout_sec_ = this->declare_parameter("icp_timeout_sec", 10.0);

        // 0 (default) = one-shot calibration, exactly as before: the node runs
        // icp_max_runs_ scenes once, publishes a single static TF and is done.
        // > 0 = after that initial one-shot calibration has unblocked the
        // detector, keep re-running the registration every N seconds on the
        // latest synced lidar/depth pair and replace the published TF only
        // when the new result is VALID (fitness above threshold); a DISCARDED
        // result leaves the previously published TF untouched.
        recalibration_period_sec_ = this->declare_parameter("recalibration_period_sec", 0.0);
        periodic_mode_ = recalibration_period_sec_ > 0.0;

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this, false);
        // Always created now (used to be periodic_mode_ only, with a
        // StaticTransformBroadcaster for the one-shot case instead) — see
        // the heartbeat timer below for why publishRefinedTF no longer uses
        // /tf_static at all.
        dynamic_tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        rclcpp::QoS qos(1);
        qos.reliable();
        qos.transient_local();

        pub_lidar_overlap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/calibration/velodyne_points", qos);
        pub_depth_aligned_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/calibration/depth_cloud_in_velodyne", qos);

        sub_lidar_.subscribe(this, lidar_topic_);
        // message_filters::Synchronizer<ApproximateTime<...>> was tried here
        // first (the "textbook" way to pair these two filters), but it never
        // fired a single match. Root cause: for at least the zstd
        // image_transport plugin (used by bags/*_validation_lab), the
        // decoded sensor_msgs::msg::Image's header.stamp comes back zeroed —
        // the decoder doesn't propagate the compressed wire message's real
        // stamp — so ApproximateTime was comparing lidar's real timestamps
        // against a frozen epoch-0 depth timestamp and could never match.
        // Synced manually instead: cache recent lidar scans, and on every
        // depth frame (timestamped by arrival time via onDepthMsg, not its
        // unreliable header.stamp) pick whichever cached scan is closest.
        lidar_cache_ = std::make_shared<message_filters::Cache<sensor_msgs::msg::PointCloud2>>(sub_lidar_, 30);

        // Live intrinsics, in both depth-source modes (extractLidarOverlapWithDepthFov's
        // pixel-space FOV test uses fx_/fy_/cx_/cy_ regardless of source; the
        // native-cloud path additionally falls back on cx_/cy_ for its own
        // image-size guess — see depth_image_width_param_'s comment above).
        // depth_intrinsics (already read into fx_/fy_/cx_/cy_ above) is only
        // the bootstrap value until the first message here arrives.
        sub_camera_info_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic_, rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg) { onCameraInfo(msg); });

        // depth_transport_ (compressedDepth/zstd/raw/...) instead of the raw
        // image_rect_raw wire format: the matching image_transport plugin
        // (installed alongside image_transport_plugins, see docker/Dockerfile)
        // transparently decompresses on receipt, so depth_msg below is still a
        // normal, uncompressed sensor_msgs::msg::Image — nothing downstream
        // changes based on which transport is actually in use.
        //
        // use_native_cloud_ skips all of this (no image, no transport, no
        // decoding) in favor of subscribing depth_cloud_topic_ directly —
        // see that param's own comment above.
        if (use_native_cloud_) {
            sub_depth_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                depth_cloud_topic_, rclcpp::SensorDataQoS(),
                [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) { onDepthCloudMsg(msg); });
            // transient_local (not the plain volatile QoS(1) this used to
            // be): the RealSense driver (and bags/*_validation_lab, which
            // recorded this topic with the SAME offered QoS — see its own
            // metadata.yaml, durability: transient_local, message_count: 1)
            // only ever publishes this exactly once — it's a latched,
            // "camera_info"-style static fact, not a stream. A volatile
            // subscriber only receives it if already connected at that one
            // instant; against `ros2 bag play --loop` (which replays each
            // topic's own recorded QoS, so its own publisher is transient_local
            // too) that meant waiting up to a full loop (~90s+ in this bag,
            // since the message happens to sit late in the recording) every
            // time this node started after the bag. transient_local here
            // instead gets the last-sent value immediately from the
            // publisher's own durability cache regardless of when this node
            // subscribes — after the very first loop has sent it once, every
            // later (re)start of this node picks it up instantly.
            sub_depth_to_color_extrinsics_ = this->create_subscription<realsense2_camera_msgs::msg::Extrinsics>(
                depth_to_color_extrinsics_topic_, rclcpp::QoS(1).transient_local(),
                [this](const realsense2_camera_msgs::msg::Extrinsics::ConstSharedPtr& msg) { onDepthToColorExtrinsics(msg); });
        } else {
            sub_depth_.subscribe(this, depth_topic_, depth_transport_, rmw_qos_profile_sensor_data);
            sub_depth_.registerCallback(
                std::function<void(const sensor_msgs::msg::Image::ConstSharedPtr&)>(
                    [this](const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) {
                        onDepthMsg(depth_msg);
                    }));
        }

        is_indoor_ = this->declare_parameter("is_indoor", false);

        // Outdoor only: if ICP calibration does not complete within 3 seconds,
        // fall back to publishing the initial-guess TF as camera_refined so the
        // detector can start without waiting for a valid ICP result.
        // If the TF lookup fails (e.g. clock not yet running with use_sim_time),
        // retry every second until it succeeds.
        if (!is_indoor_) {
            timeout_timer_ = this->create_wall_timer(
                std::chrono::duration<double>(icp_timeout_sec_),
                [this]() {
                    if (processed_) {
                        timeout_timer_->cancel();
                        return;
                    }
                    builtin_interfaces::msg::Time now_stamp;
                    const int64_t ns = this->now().nanoseconds();
                    now_stamp.sec    = static_cast<int32_t>(ns / 1000000000LL);
                    now_stamp.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
                    if (!fetchInitialGuessFromTF(now_stamp)) {
                        RCLCPP_WARN(this->get_logger(),
                            "Fallback TF lookup failed for '%s -> %s', retrying in 1 s...",
                            lidar_frame_.c_str(), camera_frame_initial_guess_.c_str());
                        timeout_timer_->cancel();
                        timeout_timer_ = this->create_wall_timer(
                            std::chrono::seconds(1),
                            [this]() {
                                if (processed_) { timeout_timer_->cancel(); return; }
                                builtin_interfaces::msg::Time now_stamp2;
                                const int64_t ns2 = this->now().nanoseconds();
                                now_stamp2.sec    = static_cast<int32_t>(ns2 / 1000000000LL);
                                now_stamp2.nanosec = static_cast<uint32_t>(ns2 % 1000000000LL);
                                if (!fetchInitialGuessFromTF(now_stamp2)) {
                                    RCLCPP_WARN(this->get_logger(),
                                        "Fallback TF lookup still failing for '%s -> %s', retrying...",
                                        lidar_frame_.c_str(), camera_frame_initial_guess_.c_str());
                                    return;
                                }
                                timeout_timer_->cancel();
                                publishOnTimeout(now_stamp2);
                                processed_ = true;
                            });
                        return;
                    }
                    timeout_timer_->cancel();
                    publishOnTimeout(now_stamp);
                    processed_ = true;
                });
        }

        // recalibration_timer_ (re-runs ICP and may replace the published TF)
        // stays periodic_mode_-only. tf_heartbeat_timer_ (re-publishes
        // whatever the last good result was, unchanged, on the *dynamic* /tf
        // topic) does NOT: it runs in one-shot mode too now. Reason: a
        // one-shot calibration used to publish camera_refined exactly once,
        // on /tf_static. That's fine against a plain bag replay, but against
        // `ros2 bag play --loop` every loop restart is a clock jump
        // backwards, which tf2_ros::Buffer treats as cause to clear itself —
        // and since nothing ever re-sends that /tf_static message, any
        // listener whose buffer gets cleared loses camera_refined
        // *permanently*, even though nothing about the calibration actually
        // changed. The 500ms heartbeat below (already built for periodic
        // mode, see onTfHeartbeat()) self-heals that within one tick instead
        // — same TF value, just re-asserted continuously instead of once.
        if (periodic_mode_) {
            recalibration_timer_ = this->create_wall_timer(
                std::chrono::duration<double>(recalibration_period_sec_),
                [this]() { onRecalibrationTimer(); });
        }
        tf_heartbeat_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            [this]() { onTfHeartbeat(); });

        RCLCPP_INFO(this->get_logger(), "Calibration ICP node ready (%s)", periodic_mode_ ? "periodic" : "one-shot");
        RCLCPP_INFO(this->get_logger(), "  depth source: %s",
            use_native_cloud_
                ? (std::string("native point cloud (") + depth_cloud_topic_ + ")").c_str()
                : (std::string("deprojected image (") + depth_topic_ + ", transport=" + depth_transport_ + ")").c_str());
        RCLCPP_INFO(this->get_logger(), "  lidar topic: %s", lidar_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  intrinsics: bootstrap fx=%.3f fy=%.3f cx=%.3f cy=%.3f, live from %s",
            fx_, fy_, cx_, cy_, camera_info_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  initial TF from: %s -> %s", lidar_frame_.c_str(), camera_frame_.c_str());
        RCLCPP_INFO(this->get_logger(), "  refined TF target child: %s", refined_camera_frame_.c_str());
        RCLCPP_INFO(this->get_logger(), "  overlap FOV margin: %d px", overlap_fov_margin_px_);
        RCLCPP_INFO(this->get_logger(), "  is_indoor: %s%s", is_indoor_ ? "true" : "false",
            is_indoor_ ? "" : (std::string(" (fallback after ") + std::to_string(static_cast<int>(icp_timeout_sec_)) + " s: average if any valid results, else initial guess)").c_str());
        if (periodic_mode_) {
            RCLCPP_INFO(this->get_logger(), "  periodic recalibration: every %.1f s after the initial one-shot calibration", recalibration_period_sec_);
        } else {
            RCLCPP_INFO(this->get_logger(), "  periodic recalibration: disabled (recalibration_period_sec=0 -> one-shot only)");
        }
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_lidar_overlap_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_depth_aligned_;

    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_lidar_;
    image_transport::SubscriberFilter sub_depth_;                       // fallback path (use_native_cloud_ == false)
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_depth_cloud_;  // native path
    rclcpp::Subscription<realsense2_camera_msgs::msg::Extrinsics>::SharedPtr sub_depth_to_color_extrinsics_;  // native path only
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_camera_info_;
    std::shared_ptr<message_filters::Cache<sensor_msgs::msg::PointCloud2>> lidar_cache_;
    // Max allowed |lidar.stamp - depth.stamp| for a manually-synced pair (see
    // onDepthMsg). Generous relative to the lidar's ~10 Hz period so brief
    // rate hiccups don't stall calibration, but still tight enough to keep
    // the two clouds visually consistent.
    static constexpr double kMaxSyncDeltaSec = 0.15;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> dynamic_tf_broadcaster_;

    std::string depth_topic_;
    std::string depth_transport_;
    std::string camera_info_topic_;
    std::string depth_cloud_topic_;
    bool use_native_cloud_{false};
    std::string depth_to_color_extrinsics_topic_;
    bool has_depth_color_matrix_{false};
    // p_coloropt = T_color_depth_ * p_depthopt — the RealSense
    // depth_to_color extrinsic message's own R,t, same convention as
    // preprocessing_node.cpp's CameraStream::T_color_depth (see that file's
    // comment on why this is NOT inverted); cached from
    // onDepthToColorExtrinsics() below.
    Eigen::Matrix4d T_color_depth_{Eigen::Matrix4d::Identity()};
    int depth_image_width_param_{0};
    int depth_image_height_param_{0};
    std::string lidar_topic_;
    std::string lidar_frame_;
    std::string camera_frame_;
    std::string camera_frame_initial_guess_;
    std::string refined_camera_frame_;

    double fx_{0.0};
    double fy_{0.0};
    double cx_{0.0};
    double cy_{0.0};
    double depth_scale_{1000.0};

    void finalizeCalibration();
    void publishOnTimeout(const builtin_interfaces::msg::Time& now_stamp);
    double depth_min_{0.5};
    double depth_max_{5.0};
    int depth_skip_{2};

    double icp_max_corr_dist_{0.2};
    int icp_max_iter_{100};
    double icp_voxel_size_{0.05};
    bool icp_use_point_to_plane_{true};
    double icp_normal_radius_{0.15};  // unused by the gtsam_points backend, kept for config parity
    int icp_normal_max_nn_{30};       // unused by the gtsam_points backend, kept for config parity
    int icp_num_threads_{4};
    int min_overlap_points_{100};
    int overlap_fov_margin_px_{120};

    int icp_max_runs_{10};
    double icp_fitness_threshold_{0.5};
    double icp_timeout_sec_{10.0};

    bool is_indoor_{false};
    rclcpp::TimerBase::SharedPtr timeout_timer_;

    double recalibration_period_sec_{0.0};
    bool periodic_mode_{false};
    rclcpp::TimerBase::SharedPtr recalibration_timer_;
    rclcpp::TimerBase::SharedPtr tf_heartbeat_timer_;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr last_synced_lidar_msg_;
    bool has_published_once_{false};

    // Depth data, normalized to a single representation regardless of
    // source (image-deprojected or native-cloud) — see the DepthFrame
    // struct and buildDepthFrameFrom{Image,Cloud}() below. Replaces the
    // old last_synced_depth_msg_ (Image::ConstSharedPtr): periodic mode
    // needs to re-run runCalibration() on "whatever the latest synced pair
    // was", and since a DepthFrame is already exactly what runCalibration
    // needs, there's no reason to keep the raw source message around
    // instead (nor, for the native-cloud path, any single message TYPE to
    // keep — it's built once per arrival either way, see onData below).
    struct DepthFrame {
        gtsam_points::PointCloudCPU::Ptr camera_cloud;  // color-optical-frame-equivalent convention, like depthToPointCloud()'s own output
        uint32_t width{0};
        uint32_t height{0};
        builtin_interfaces::msg::Time stamp;
        std::string frame_id;
        bool valid{false};
        std::string error;
    };
    DepthFrame last_synced_depth_frame_;
    bool has_last_synced_depth_frame_{false};

    bool processed_{false};
    Eigen::Matrix4d T_lidar_camera_init_{Eigen::Matrix4d::Identity()};
    Eigen::Matrix4d T_lidar_camera_last_good_{Eigen::Matrix4d::Identity()};

    std::vector<Eigen::Matrix4d> all_valid_transforms_;
    std::vector<Eigen::Vector3d> all_translations_;
    std::vector<Eigen::Vector3d> all_rpy_angles_;
    int valid_count_ = 0;
    int scene_count_ = 0;
    sensor_msgs::msg::PointCloud2 last_lidar_msg_out_;
    pcl::PointCloud<pcl::PointXYZ> last_lidar_cloud_;
    sensor_msgs::msg::PointCloud2 last_depth_msg_out_;
    gtsam_points::PointCloudCPU::Ptr last_depth_cloud_camera_;
    // Actual depth *image* dimensions (e.g. 1280x720), captured from
    // depth_msg->width/height in runCalibration. Do NOT reuse
    // last_depth_msg_out_.width/height for this in finalizeCalibration:
    // that message is the already-converted debug PointCloud2, whose width
    // is the point count (unorganized cloud, height=1) — passing that into
    // extractLidarOverlapWithDepthFov as the image size collapses the FOV
    // pixel-bounds check to a ~1px-tall strip and silently discards nearly
    // all otherwise-valid lidar points.
    uint32_t last_depth_image_width_ = 0;
    uint32_t last_depth_image_height_ = 0;
    builtin_interfaces::msg::Time last_stamp_;
    std::string last_status_message_;

    static Eigen::Matrix4d transformMsgToEigen(const geometry_msgs::msg::Transform& tf_msg) {
        Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
        Eigen::Quaterniond q(tf_msg.rotation.w, tf_msg.rotation.x, tf_msg.rotation.y, tf_msg.rotation.z);
        m.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
        m(0, 3) = tf_msg.translation.x;
        m(1, 3) = tf_msg.translation.y;
        m(2, 3) = tf_msg.translation.z;
        return m;
    }

    static geometry_msgs::msg::Transform eigenToTransformMsg(const Eigen::Matrix4d& m) {
        geometry_msgs::msg::Transform out;
        Eigen::Quaterniond q(m.block<3, 3>(0, 0));
        out.translation.x = m(0, 3);
        out.translation.y = m(1, 3);
        out.translation.z = m(2, 3);
        out.rotation.w = q.w();
        out.rotation.x = q.x();
        out.rotation.y = q.y();
        out.rotation.z = q.z();
        return out;
    }

    static Eigen::Vector3d rotationToRPY(const Eigen::Matrix3d& rot) {
        Eigen::Quaterniond q(rot);
        tf2::Quaternion tf_q(q.x(), q.y(), q.z(), q.w());
        tf2::Matrix3x3 tf_m(tf_q);
        double roll = 0.0;
        double pitch = 0.0;
        double yaw = 0.0;
        tf_m.getRPY(roll, pitch, yaw);
        return Eigen::Vector3d(roll, pitch, yaw);
    }

    bool fetchInitialGuessFromTF(const builtin_interfaces::msg::Time& stamp) {
        try {
            auto tf_stamped = tf_buffer_->lookupTransform(
                lidar_frame_, camera_frame_initial_guess_, rclcpp::Time(stamp), std::chrono::milliseconds(200));
            T_lidar_camera_init_ = transformMsgToEigen(tf_stamped.transform);
            return true;
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN(this->get_logger(), "TF lookup at stamp failed: %s. Trying latest TF.", ex.what());
        }

        try {
            auto tf_stamped = tf_buffer_->lookupTransform(
                lidar_frame_, camera_frame_initial_guess_, tf2::TimePointZero, std::chrono::milliseconds(200));
            T_lidar_camera_init_ = transformMsgToEigen(tf_stamped.transform);
            return true;
        } catch (const tf2::TransformException& ex) {
            RCLCPP_ERROR(this->get_logger(), "Failed to read initial guess from /tf: %s", ex.what());
            return false;
        }
    }

    gtsam_points::PointCloudCPU::Ptr depthToPointCloud(const cv::Mat& depth_image) const {
        std::vector<Eigen::Vector4d> pts;
        pts.reserve((depth_image.rows / depth_skip_) * (depth_image.cols / depth_skip_));

        const bool is_u16 = depth_image.type() == CV_16UC1;
        const bool is_f32 = depth_image.type() == CV_32FC1;
        if (!is_u16 && !is_f32) {
            RCLCPP_ERROR(this->get_logger(), "Unsupported depth image encoding (need 16UC1 or 32FC1)");
            return std::make_shared<gtsam_points::PointCloudCPU>();
        }

        for (int y = 0; y < depth_image.rows; y += depth_skip_) {
            for (int x = 0; x < depth_image.cols; x += depth_skip_) {
                double z = 0.0;
                if (is_u16) {
                    const uint16_t d = depth_image.at<uint16_t>(y, x);
                    if (d == 0) {
                        continue;
                    }
                    z = static_cast<double>(d) / depth_scale_;
                } else {
                    const float d = depth_image.at<float>(y, x);
                    if (!std::isfinite(d) || d <= 0.0f) {
                        continue;
                    }
                    z = static_cast<double>(d);
                }

                if (z < depth_min_ || z > depth_max_) {
                    continue;
                }

                const double X = (static_cast<double>(x) - cx_) * z / fx_;
                const double Y = (static_cast<double>(y) - cy_) * z / fy_;
                pts.emplace_back(X, Y, z, 1.0);
            }
        }

        return std::make_shared<gtsam_points::PointCloudCPU>(pts);
    }

    // Fallback path: decode + deproject depth_topic_ ourselves, exactly as
    // this file always did before use_native_cloud_ existed.
    DepthFrame buildDepthFrameFromImage(const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) const {
        DepthFrame f;
        f.stamp = depth_msg->header.stamp;
        f.frame_id = depth_msg->header.frame_id;
        f.width = depth_msg->width;
        f.height = depth_msg->height;
        try {
            f.camera_cloud = depthToPointCloud(cv_bridge::toCvShare(depth_msg)->image);
            f.valid = true;
        } catch (const std::exception& ex) {
            f.error = std::string("Depth cv_bridge conversion failed: ") + ex.what();
            f.valid = false;
        }
        return f;
    }

    // Native path: parse an already-deprojected PointCloud2 straight from
    // depth_cloud_topic_ instead of decoding+deprojecting an image at all —
    // e.g. a RealSense driver's own "<name>/camera/depth/color/points".
    //
    // Frame-consistency note: unlike preprocessing_node.cpp's own native
    // path (which tolerates a missing T_color_depth_ correction by falling
    // back to a degraded-but-still-roughly-right crop for a few frames —
    // see finishDepthCloud()'s have_calibration_tf gating), that shortcut
    // is NOT acceptable here. This calibration runs once at startup and
    // averages a handful of scenes into the T_lidar_camera it publishes as
    // <name>_refined for the rest of the pipeline's lifetime — if even one
    // of those scenes were calibrated against depth-frame points mistaken
    // for color-frame ones, the whole published TF would be off by the
    // depth<->color extrinsic, silently and permanently. So onDepthCloudMsg
    // (below) simply refuses to start a scene until has_depth_color_matrix_
    // is true, and this function only ever applies the correction, never
    // skips it.
    //
    // p_coloropt = T_color_depth_ * p_depthopt (cached convention, see
    // T_color_depth_'s own comment) — applied per point below to bring native points into the
    // same color-optical-frame-equivalent convention depthToPointCloud()'s
    // deprojection already produces, which is what T_lidar_camera_init_ (and
    // therefore the ICP result built from it) is actually expressed against.
    DepthFrame buildDepthFrameFromCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg) const {
        DepthFrame f;
        f.stamp = cloud_msg->header.stamp;
        f.frame_id = cloud_msg->header.frame_id;
        // No image resolution to read here — see depth_image_width_param_/
        // depth_image_height_param_'s own comment for the fallback used.
        f.width = depth_image_width_param_ > 0
            ? static_cast<uint32_t>(depth_image_width_param_)
            : static_cast<uint32_t>(std::lround(2.0 * cx_));
        f.height = depth_image_height_param_ > 0
            ? static_cast<uint32_t>(depth_image_height_param_)
            : static_cast<uint32_t>(std::lround(2.0 * cy_));

        pcl::PointCloud<pcl::PointXYZ> raw;
        pcl::fromROSMsg(*cloud_msg, raw);
        std::vector<Eigen::Vector4d> pts;
        pts.reserve(raw.points.size());
        for (const auto& pt : raw.points) {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
                continue;
            }
            if (pt.z < depth_min_ || pt.z > depth_max_) {
                continue;
            }
            Eigen::Vector4d p(pt.x, pt.y, pt.z, 1.0);
            if (has_depth_color_matrix_) {
                p = T_color_depth_ * p;
            }
            pts.emplace_back(p);
        }
        f.camera_cloud = std::make_shared<gtsam_points::PointCloudCPU>(pts);
        f.valid = true;
        return f;
    }

    // Native path only: caches the RealSense factory depth<->color
    // extrinsic (see T_color_depth_'s own comment). Mirrors
    // preprocessing_node.cpp's onDepthToColorExtrinsics() — same message,
    // same math — minus the static-TF broadcast, which that node already
    // does; this one only needs the matrix.
    void onDepthToColorExtrinsics(const realsense2_camera_msgs::msg::Extrinsics::ConstSharedPtr& msg) {
        if (has_depth_color_matrix_) {
            return;  // static, only needs to be cached once
        }
        Eigen::Matrix3d R;
        R << msg->rotation[0], msg->rotation[3], msg->rotation[6],
             msg->rotation[1], msg->rotation[4], msg->rotation[7],
             msg->rotation[2], msg->rotation[5], msg->rotation[8];
        Eigen::Quaterniond q(R);
        q.normalize();
        T_color_depth_.setIdentity();
        T_color_depth_.block<3, 3>(0, 0) = q.toRotationMatrix();
        T_color_depth_(0, 3) = msg->translation[0];
        T_color_depth_(1, 3) = msg->translation[1];
        T_color_depth_(2, 3) = msg->translation[2];
        has_depth_color_matrix_ = true;
        RCLCPP_INFO(this->get_logger(), "Cached depth<->color extrinsics from %s",
                    depth_to_color_extrinsics_topic_.c_str());
    }

    // fx_/fy_/cx_/cy_ start out at whatever depth_intrinsics declared (see
    // that param's own comment) and are overwritten here on every message —
    // camera_info_topic_ should be the COLOR camera's own info topic (K is
    // in pixel units against that camera's resolution), matching what
    // depth_topic_ (an aligned-to-color stream) and, when native-cloud mode
    // applies its T_color_depth_ correction, the native path too both
    // ultimately need these against. Not gated to "first message only" —
    // intrinsics are physically fixed for a given camera, so re-applying
    // the same K on every message is a harmless no-op, and this stays
    // correct without extra state if a camera ever republishes a changed K.
    void onCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg) {
        fx_ = msg->k[0];
        fy_ = msg->k[4];
        cx_ = msg->k[2];
        cy_ = msg->k[5];
    }

    gtsam_points::PointCloudCPU::Ptr extractLidarOverlapWithDepthFov(
        const pcl::PointCloud<pcl::PointXYZ>& lidar_cloud,
        uint32_t image_width,
        uint32_t image_height,
        const Eigen::Matrix4d& T_lidar_camera) const {

        std::vector<Eigen::Vector4d> pts;
        pts.reserve(lidar_cloud.points.size());

        const Eigen::Matrix4d T_camera_lidar = T_lidar_camera.inverse();

        for (const auto& pt : lidar_cloud.points) {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
                continue;
            }

            Eigen::Vector4d p_lidar(pt.x, pt.y, pt.z, 1.0);
            Eigen::Vector4d p_camera = T_camera_lidar * p_lidar;

            // Points beyond depth_max_ are intentionally excluded: the depth
            // camera itself cannot measure that far (RealSense aligned-depth
            // range tops out well under 10m), so keeping farther lidar points
            // here would compare them against nothing and only adds noise to
            // ICP (verified empirically: raising depth_max_ to 15m roughly
            // doubled kept points but made every scene fail the fitness
            // threshold instead of improving convergence).
            const double z = p_camera(2);
            if (z <= depth_min_ || z >= depth_max_) {
                continue;
            }

            const double u = fx_ * p_camera(0) / z + cx_;
            const double v = fy_ * p_camera(1) / z + cy_;
            const double min_u = -static_cast<double>(overlap_fov_margin_px_);
            const double min_v = -static_cast<double>(overlap_fov_margin_px_);
            const double max_u = static_cast<double>(image_width) + static_cast<double>(overlap_fov_margin_px_);
            const double max_v = static_cast<double>(image_height) + static_cast<double>(overlap_fov_margin_px_);
            if (u < min_u || v < min_v || u >= max_u || v >= max_v) {
                continue;
            }

            pts.emplace_back(pt.x, pt.y, pt.z, 1.0);
        }

        return std::make_shared<gtsam_points::PointCloudCPU>(pts);
    }

    sensor_msgs::msg::PointCloud2 toROSPointCloud2(
        const gtsam_points::PointCloudCPU& cloud,
        const builtin_interfaces::msg::Time& stamp,
        const std::string& frame,
        uint8_t r,
        uint8_t g,
        uint8_t b) const {

        pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
        pcl_cloud.points.reserve(cloud.size());
        for (size_t i = 0; i < cloud.size(); i++) {
            pcl::PointXYZRGB q;
            q.x = static_cast<float>(cloud.points[i].x());
            q.y = static_cast<float>(cloud.points[i].y());
            q.z = static_cast<float>(cloud.points[i].z());
            q.r = r;
            q.g = g;
            q.b = b;
            pcl_cloud.points.push_back(q);
        }

        pcl_cloud.width = static_cast<uint32_t>(pcl_cloud.points.size());
        pcl_cloud.height = 1;

        sensor_msgs::msg::PointCloud2 out;
        pcl::toROSMsg(pcl_cloud, out);
        out.header.stamp = stamp;
        out.header.frame_id = frame;
        return out;
    }

    // Publishes the refined TF on the dynamic /tf topic (not /tf_static,
    // even in one-shot mode — see the heartbeat timer's comment in the
    // constructor for why: a single static send can be permanently lost
    // from a listener's buffer after a clock-jump-triggered clear, e.g. one
    // caused by `ros2 bag play --loop`). This first send is immediate,
    // same as before; onTfHeartbeat() keeps re-asserting the same value
    // afterward.
    void publishRefinedTF(const Eigen::Matrix4d& T_lidar_camera_refined,
                          const builtin_interfaces::msg::Time& stamp) {
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = stamp;
        tf_msg.header.frame_id = lidar_frame_;
        tf_msg.child_frame_id = refined_camera_frame_;
        tf_msg.transform = eigenToTransformMsg(T_lidar_camera_refined);

        dynamic_tf_broadcaster_->sendTransform(tf_msg);

        T_lidar_camera_last_good_ = T_lidar_camera_refined;
        has_published_once_ = true;

        RCLCPP_INFO(this->get_logger(), "Published refined TF: %s -> %s",
                    lidar_frame_.c_str(), refined_camera_frame_.c_str());
    }

    // Manual lidar/depth sync (see the comment on lidar_cache_'s declaration
    // for why this replaces message_filters::Synchronizer<ApproximateTime>).
    // Runs on every depth frame: picks whichever cached lidar scan is
    // closest in time and, if within kMaxSyncDeltaSec, hands the pair to
    // onData exactly like the synchronizer used to.
    void onDepthMsg(const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) {
        // depth_msg->header.stamp cannot be trusted here: at least the zstd
        // image_transport decoder (used for bags/*_validation_lab) does not
        // propagate the compressed wire message's stamp into the decoded
        // Image — it comes back zeroed, which silently broke every sync
        // approach that compared it against lidar_msg->header.stamp (always
        // ~decades apart). Node-clock time-of-arrival is a good enough stand
        // in: decode/transport latency is well under kMaxSyncDeltaSec.
        const rclcpp::Time depth_time = this->now();
        const auto before = lidar_cache_->getElemBeforeTime(depth_time);
        const auto after = lidar_cache_->getElemAfterTime(depth_time);

        sensor_msgs::msg::PointCloud2::ConstSharedPtr best;
        double best_delta = std::numeric_limits<double>::infinity();
        for (const auto& candidate : {before, after}) {
            if (!candidate) {
                continue;
            }
            const double delta = std::abs((rclcpp::Time(candidate->header.stamp) - depth_time).seconds());
            if (delta < best_delta) {
                best_delta = delta;
                best = candidate;
            }
        }

        if (best && best_delta <= kMaxSyncDeltaSec) {
            onData(best, depth_msg);
        }
    }

    // Native path's sibling of onDepthMsg above — same manual arrival-time
    // sync against lidar_cache_ (a native cloud carries no more trustworthy
    // a stamp guarantee than the image path did, and reusing the same
    // convention here keeps the two paths identical apart from what they
    // each hand to onData). Guarded on has_depth_color_matrix_: see
    // buildDepthFrameFromCloud()'s own comment for why starting a scene
    // before that extrinsic has arrived isn't acceptable here (unlike
    // preprocessing_node.cpp's tolerant degrade-for-a-few-frames approach).
    void onDepthCloudMsg(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg) {
        if (!has_depth_color_matrix_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Native depth cloud received but depth<->color extrinsics not yet available on '%s' "
                "— waiting before starting calibration scenes.", depth_to_color_extrinsics_topic_.c_str());
            return;
        }

        const rclcpp::Time depth_time = this->now();
        const auto before = lidar_cache_->getElemBeforeTime(depth_time);
        const auto after = lidar_cache_->getElemAfterTime(depth_time);

        sensor_msgs::msg::PointCloud2::ConstSharedPtr best;
        double best_delta = std::numeric_limits<double>::infinity();
        for (const auto& candidate : {before, after}) {
            if (!candidate) {
                continue;
            }
            const double delta = std::abs((rclcpp::Time(candidate->header.stamp) - depth_time).seconds());
            if (delta < best_delta) {
                best_delta = delta;
                best = candidate;
            }
        }

        if (best && best_delta <= kMaxSyncDeltaSec) {
            onData(best, cloud_msg);
        }
    }

    // Two thin overloads, one per depth source type, each building the
    // normalized DepthFrame representation and handing off to the actual
    // (source-agnostic) scene-processing logic below. Keeps onDepthMsg's/
    // onDepthCloudMsg's existing "onData(best, depth_msg)"-style call sites
    // unchanged.
    void onData(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg,
                const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) {
        processScene(lidar_msg, buildDepthFrameFromImage(depth_msg));
    }

    void onData(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg,
                const sensor_msgs::msg::PointCloud2::ConstSharedPtr& depth_cloud_msg) {
        processScene(lidar_msg, buildDepthFrameFromCloud(depth_cloud_msg));
    }

    void processScene(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg, const DepthFrame& frame) {
        if (processed_) {
            if (periodic_mode_) {
                // Keep the latest synced pair around for the periodic
                // recalibration timer; the initial one-shot calibration
                // (below) has already completed and unblocked the detector.
                last_synced_lidar_msg_ = lidar_msg;
                last_synced_depth_frame_ = frame;
                has_last_synced_depth_frame_ = true;
            }
            return;
        }

        RCLCPP_INFO(this->get_logger(), "[Step %d/%d]", scene_count_ + 1, icp_max_runs_);
        std::string status_message;
        Eigen::Matrix4d T_result;
        Eigen::Vector3d translation, rpy;
        bool valid = runCalibration(lidar_msg, frame, status_message, T_result, translation, rpy);
        if (valid) {
            all_valid_transforms_.push_back(T_result);
            all_translations_.push_back(translation);
            all_rpy_angles_.push_back(rpy);
            ++valid_count_;
        }
        ++scene_count_;

        // Store last scene's clouds for publishing after all runs. Stamped
        // with this->now() rather than frame.stamp: the latter is
        // unreliable for the image path (see onDepthMsg — at least zstd
        // decodes it to zero); kept unstamped-from-source for the
        // native-cloud path too, for consistency.
        const builtin_interfaces::msg::Time scene_stamp = this->now();
        last_lidar_msg_out_ = toROSPointCloud2(*extractLidarOverlapWithDepthFov(pcl::PointCloud<pcl::PointXYZ>(), frame.width, frame.height, T_lidar_camera_init_), scene_stamp, lidar_frame_, 0, 255, 0);
        // Reuses frame.camera_cloud (already built once, above, inside
        // runCalibration) instead of a second independent decode — the old
        // code here called depthToPointCloud(cv_bridge::toCvShare(...))
        // again from scratch, which was always redundant (identical output)
        // for the image path and has no equivalent at all for the
        // native-cloud path.
        last_depth_msg_out_ = toROSPointCloud2(*frame.camera_cloud, scene_stamp,
            frame.frame_id.empty() ? camera_frame_ : frame.frame_id, 255, 80, 80);
        last_stamp_ = scene_stamp;
        last_status_message_ = status_message;

        if (scene_count_ >= icp_max_runs_) {
            finalizeCalibration();
            if (valid_count_ > 0) {
                processed_ = true;
                if (periodic_mode_) {
                    last_synced_lidar_msg_ = lidar_msg;
                    last_synced_depth_frame_ = frame;
                    has_last_synced_depth_frame_ = true;
                }
            } else {
                scene_count_ = 0;
                valid_count_ = 0;
                all_valid_transforms_.clear();
                all_translations_.clear();
                all_rpy_angles_.clear();
            }
        }
    }

    // Periodic mode only: re-run the registration on the latest synced pair
    // and replace the published TF only if the new result is VALID. A
    // DISCARDED result is logged and the previously published (last known
    // good) TF is left untouched, so a single bad scene mid-mission can't
    // make camera_refined jump.
    void onRecalibrationTimer() {
        if (!last_synced_lidar_msg_ || !has_last_synced_depth_frame_) {
            return;
        }

        std::string status_message;
        Eigen::Matrix4d T_result;
        Eigen::Vector3d translation, rpy;
        const bool valid = runCalibration(last_synced_lidar_msg_, last_synced_depth_frame_, status_message, T_result, translation, rpy);

        if (valid) {
            publishRefinedTF(T_result, last_synced_depth_frame_.stamp);
            RCLCPP_INFO(this->get_logger(), "Periodic recalibration: TF updated.");
        } else {
            RCLCPP_WARN(this->get_logger(), "Periodic recalibration: result %s, keeping previous TF.", status_message.c_str());
        }
    }

    // Runs in both modes now: keep republishing the last known-good TF on
    // /tf (unchanged in one-shot mode — see the constructor's comment) so
    // it stays inside consumers' tf2 buffers even after a clock-jump clears
    // one, and (periodic mode only, incidentally) between two potentially
    // slow recalibration ticks too.
    void onTfHeartbeat() {
        if (!has_published_once_) {
            return;
        }
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = this->now();
        tf_msg.header.frame_id = lidar_frame_;
        tf_msg.child_frame_id = refined_camera_frame_;
        tf_msg.transform = eigenToTransformMsg(T_lidar_camera_last_good_);
        dynamic_tf_broadcaster_->sendTransform(tf_msg);
    }

    bool runCalibration(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg,
                        const DepthFrame& frame,
                        std::string& status_message,
                        Eigen::Matrix4d& T_result,
                        Eigen::Vector3d& translation,
                        Eigen::Vector3d& rpy) {
        if (!frame.valid) {
            status_message = frame.error.empty() ? "Depth frame could not be built" : frame.error;
            RCLCPP_ERROR(this->get_logger(), "%s", status_message.c_str());
            return false;
        }
        if (!fetchInitialGuessFromTF(frame.stamp)) {
            status_message = "Cannot start calibration without initial TF guess velodyne->camera_link";
            RCLCPP_ERROR(this->get_logger(), "%s", status_message.c_str());
            return false;
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*lidar_msg, *lidar_cloud);
        last_lidar_cloud_ = *lidar_cloud;

        auto depth_cloud_camera = frame.camera_cloud;
        last_depth_cloud_camera_ = depth_cloud_camera;
        if (!depth_cloud_camera || depth_cloud_camera->size() == 0) {
            status_message = "Depth point cloud is empty";
            RCLCPP_ERROR(this->get_logger(), "%s", status_message.c_str());
            return false;
        }
        last_depth_image_width_ = frame.width;
        last_depth_image_height_ = frame.height;
        auto lidar_overlap = extractLidarOverlapWithDepthFov(
            *lidar_cloud, frame.width, frame.height, T_lidar_camera_init_);
        if (static_cast<int>(lidar_overlap->size()) < min_overlap_points_) {
            RCLCPP_WARN(this->get_logger(),
                        "LiDAR overlap too small (%d points, min=%d). Using full LiDAR cloud as ICP target.",
                        static_cast<int>(lidar_overlap->size()), min_overlap_points_);
            std::vector<Eigen::Vector4d> pts;
            pts.reserve(lidar_cloud->points.size());
            for (const auto& pt : lidar_cloud->points) {
                if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
                    continue;
                }
                pts.emplace_back(pt.x, pt.y, pt.z, 1.0);
            }
            lidar_overlap = std::make_shared<gtsam_points::PointCloudCPU>(pts);
        }
        auto source = depth_cloud_camera;
        auto target = lidar_overlap;
        if (icp_voxel_size_ > 0.0) {
            auto source_ds = gtsam_points::voxelgrid_sampling(source, icp_voxel_size_, icp_num_threads_);
            auto target_ds = gtsam_points::voxelgrid_sampling(target, icp_voxel_size_, icp_num_threads_);
            if (source_ds->size() > 0) {
                source = source_ds;
            }
            if (target_ds->size() > 0) {
                target = target_ds;
            }
        }

        gtsam::Values values;
        values.insert(0, gtsam::Pose3(T_lidar_camera_init_));
        gtsam::NonlinearFactorGraph graph;
        gtsam::Values optimized;
        double error = 0.0;
        double fitness = 0.0;

        if (icp_use_point_to_plane_) {
            // GICP requires per-point local covariances on both clouds.
            target->add_covs(gtsam_points::estimate_covariances(target->points, target->size()));
            source->add_covs(gtsam_points::estimate_covariances(source->points, source->size()));

            auto factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(gtsam::Pose3::Identity(), 0, target, source);
            factor->set_max_correspondence_distance(icp_max_corr_dist_);
            factor->set_num_threads(icp_num_threads_);
            graph.add(factor);

            gtsam_points::LevenbergMarquardtExtParams lm_params;
            lm_params.setMaxIterations(icp_max_iter_);
            optimized = gtsam_points::LevenbergMarquardtOptimizerExt(graph, values, lm_params).optimize();

            error = factor->error(optimized);
            fitness = factor->inlier_fraction();
        } else {
            auto factor = gtsam::make_shared<gtsam_points::IntegratedICPFactor>(gtsam::Pose3::Identity(), 0, target, source);
            factor->set_max_correspondence_distance(icp_max_corr_dist_);
            factor->set_num_threads(icp_num_threads_);
            graph.add(factor);

            gtsam_points::LevenbergMarquardtExtParams lm_params;
            lm_params.setMaxIterations(icp_max_iter_);
            optimized = gtsam_points::LevenbergMarquardtOptimizerExt(graph, values, lm_params).optimize();

            error = factor->error(optimized);

            // Unlike IntegratedGICPFactor_, IntegratedICPFactor_ doesn't expose
            // an inlier_fraction() accessor (its correspondences are private).
            // Compute the same metric here: the fraction of transformed source
            // points whose nearest target neighbor is within the configured
            // correspondence-distance threshold.
            const Eigen::Matrix4d T_target_source = optimized.at<gtsam::Pose3>(0).matrix();
            gtsam_points::KdTree target_tree(target->points, target->size());
            const double max_sq_dist = icp_max_corr_dist_ * icp_max_corr_dist_;
            size_t inliers = 0;
            for (size_t i = 0; i < source->size(); i++) {
                const Eigen::Vector4d pt = T_target_source * source->points[i];
                size_t index;
                double sq_dist;
                if (target_tree.knn_search(pt.data(), 1, &index, &sq_dist) > 0 && sq_dist <= max_sq_dist) {
                    inliers++;
                }
            }
            fitness = source->size() > 0 ? static_cast<double>(inliers) / source->size() : 0.0;
        }

        if (fitness >= icp_fitness_threshold_) {
            T_result = optimized.at<gtsam::Pose3>(0).matrix();
            translation = T_result.block<3,1>(0,3);
            rpy = rotationToRPY(T_result.block<3,3>(0,0));
            status_message = "VALID";
            RCLCPP_INFO(this->get_logger(),
                        "  fitness=%.4f error=%.4f -> VALID | t=[%.3f, %.3f, %.3f] rpy=[%.3f, %.3f, %.3f] rad",
                        fitness, error,
                        translation(0), translation(1), translation(2),
                        rpy(0), rpy(1), rpy(2));
            return true;
        } else {
            status_message = "DISCARDED";
            RCLCPP_WARN(this->get_logger(),
                        "  fitness=%.4f error=%.4f -> DISCARDED (threshold=%.4f)",
                        fitness, error, icp_fitness_threshold_);
            return false;
        }
    }

};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CalibrationICPNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

void CalibrationICPNode::publishOnTimeout(const builtin_interfaces::msg::Time& now_stamp) {
    if (valid_count_ > 0) {
        RCLCPP_WARN(this->get_logger(),
            "ICP calibration timeout (%.0f s) in outdoor mode. "
            "Using average of %d/%d valid ICP results for '%s'.",
            icp_timeout_sec_, valid_count_, scene_count_, refined_camera_frame_.c_str());
        if (last_stamp_.sec == 0 && last_stamp_.nanosec == 0) {
            last_stamp_ = now_stamp;
        }
        finalizeCalibration();
    } else {
        RCLCPP_WARN(this->get_logger(),
            "ICP calibration timeout (%.0f s) in outdoor mode. "
            "No valid ICP results yet. Falling back to initial-guess TF for '%s'.",
            icp_timeout_sec_, refined_camera_frame_.c_str());
        publishRefinedTF(T_lidar_camera_init_, now_stamp);
    }
}

void CalibrationICPNode::finalizeCalibration() {
    if (valid_count_ == 0) {
        RCLCPP_WARN(this->get_logger(), "No valid ICP results above fitness threshold across all scenes");
        return;
    }
    // Average translation
    Eigen::Vector3d mean_translation = Eigen::Vector3d::Zero();
    for (const auto& t : all_translations_) {
        mean_translation += t;
    }
    mean_translation /= static_cast<double>(valid_count_);
    // Average RPY using mean of sin/cos and atan2
    Eigen::Vector3d mean_rpy = Eigen::Vector3d::Zero();
    for (int i = 0; i < 3; ++i) {
        double sum_sin = 0.0, sum_cos = 0.0;
        for (const auto& rpy : all_rpy_angles_) {
            sum_sin += std::sin(rpy(i));
            sum_cos += std::cos(rpy(i));
        }
        mean_rpy(i) = std::atan2(sum_sin / valid_count_, sum_cos / valid_count_);
    }
    // Build mean transform
    Eigen::Matrix4d T_mean = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d rot_mean;
    rot_mean = (
        Eigen::AngleAxisd(mean_rpy(2), Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(mean_rpy(1), Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(mean_rpy(0), Eigen::Vector3d::UnitX())
    ).toRotationMatrix();
    T_mean.block<3,3>(0,0) = rot_mean;
    T_mean.block<3,1>(0,3) = mean_translation;
    RCLCPP_INFO(this->get_logger(), "==================== CALIBRATION DONE ====================");
    RCLCPP_INFO(this->get_logger(), "Valid scenes: %d/%d", valid_count_, icp_max_runs_);
    RCLCPP_INFO(this->get_logger(), "Translation: [%.6f, %.6f, %.6f] m", mean_translation(0), mean_translation(1), mean_translation(2));
    RCLCPP_INFO(this->get_logger(), "RPY: [%.6f, %.6f, %.6f] rad", mean_rpy(0), mean_rpy(1), mean_rpy(2));
    RCLCPP_INFO(this->get_logger(), "==========================================================");
    // Use the calibrated T_mean here, not T_lidar_camera_init_: this decides
    // which lidar points count as "in the camera FOV" for the published
    // debug cloud, and must match the transform actually used to project the
    // depth cloud below (T_mean too) — otherwise the two clouds are selected
    // with inconsistent extrinsics and barely overlap when they shouldn't.
    auto overlap_cloud = extractLidarOverlapWithDepthFov(last_lidar_cloud_, last_depth_image_width_, last_depth_image_height_, T_mean);
    last_lidar_msg_out_ = toROSPointCloud2(*overlap_cloud, last_depth_msg_out_.header.stamp, lidar_frame_, 0, 255, 0);
    if (last_depth_cloud_camera_ && last_depth_cloud_camera_->size() > 0) {
        // Voxel-downsample before publishing: the raw depth cloud (~150k
        // points from a full-res depth image) dwarfs the lidar overlap cloud
        // (a few thousand points at most) by ~50-100x, which in rviz reads
        // as "the lidar cloud is basically empty" even when every lidar
        // point is there — it's just visually buried under a much denser
        // cloud. Reusing icp_voxel_size_ (already used to downsample both
        // clouds before ICP itself, see below) keeps this consistent with
        // what the registration actually sees and roughly matches the lidar
        // cloud's density, instead of introducing a separate debug-only knob.
        auto depth_cloud_ds = gtsam_points::voxelgrid_sampling(last_depth_cloud_camera_, icp_voxel_size_, icp_num_threads_);
        auto depth_cloud_lidar = gtsam_points::transform(depth_cloud_ds, Eigen::Isometry3d(T_mean));
        auto depth_msg_out = toROSPointCloud2(*depth_cloud_lidar, last_depth_msg_out_.header.stamp, lidar_frame_, 255, 80, 80);
        pub_depth_aligned_->publish(depth_msg_out);
    } else {
        RCLCPP_WARN(this->get_logger(), "No valid last depth cloud to transform and publish.");
    }
    pub_lidar_overlap_->publish(last_lidar_msg_out_);
    publishRefinedTF(T_mean, last_stamp_);
}
