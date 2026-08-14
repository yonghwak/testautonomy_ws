cat > ~/kaboat_ws/src/kaboat_camera/kaboat_camera/camera_node.py << 'EOF'
import math
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import numpy as np


class CameraNode(Node):
    """
    ZED2i 카메라 - 색+모양+거리+좌우판정까지 인식해서 camera/detections(JSON)로 발행.
    각 색의 마스크도 그대로 'camera/mask_R' 등으로 발행 - rosboard로 확인 가능.

    모양 판별: 삼각형/네모/십자는 꼭짓점개수 기반. 원형은 까다롭게 검증
    (circularity + fill_ratio 둘 다 통과해야 함) - 안 그러면 노이즈 덩어리가
    다 원형으로 오판정됨.

    좌/중/우(zone): 화면을 가로로 3등분해서 물체 중심 x좌표 기준으로 판정
    (각도 기반보다 훨씬 직관적이고 중앙 판정 편향 없음).
    """

    COLOR_RANGES = {
        'G': ((35, 80, 60), (85, 255, 255)),
        'B': ((100, 100, 100), (130, 255, 255)),
        'W': ((0, 0, 190), (180, 40, 255)),
        'O': ((13, 120, 70), (20, 255, 255)),
        'Y': ((21, 100, 100), (33, 255, 255)),
    }
    RED_RANGES = [
        ((0, 70, 60), (12, 255, 255)),
        ((168, 70, 60), (180, 255, 255)),
    ]

    HORIZONTAL_FOV_DEG = 100.0
    MIN_PIXEL_COUNT = 1500
    CIRCLE_MIN_CIRCULARITY = 0.78
    CIRCLE_MIN_FILL_RATIO = 0.78

    RGB_TOPIC = '/zed/zed_node/rgb/color/rect/image'
    DEPTH_TOPIC = '/zed/zed_node/depth/depth_registered'

    DEBUG_COLORS_BGR = {
        'R': (0, 0, 255), 'G': (0, 255, 0), 'B': (255, 0, 0), 'W': (255, 255, 255),
        'O': (0, 140, 255), 'Y': (0, 255, 255),
    }

    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('target_color', '')
        self.declare_parameter('target_shape', '')
        self.target_color = self.get_parameter('target_color').value.upper()
        self.target_shape = self.get_parameter('target_shape').value.lower()

        self.bridge = CvBridge()
        self.latest_depth = None

        self.create_subscription(
            Image, self.RGB_TOPIC, self.image_cb, qos_profile_sensor_data)
        self.create_subscription(
            Image, self.DEPTH_TOPIC, self.depth_cb, qos_profile_sensor_data)

        self.detections_pub = self.create_publisher(String, 'camera/detections', 10)
        self.debug_pub = self.create_publisher(Image, 'camera/debug_image', 10)
        self.mask_pubs = {
            c: self.create_publisher(Image, f'camera/mask_{c}', 10)
            for c in ['R', 'G', 'B', 'W', 'O', 'Y']
        }

        self.get_logger().info(
            f'카메라 노드 시작 (등록색={list(self.COLOR_RANGES.keys()) + ["R"]})')
        if self.target_color and self.target_shape:
            self.get_logger().info(f'타겟 지정: {self.target_color}/{self.target_shape}')

    def depth_cb(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception:
            pass

    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'이미지 변환 실패: {e}', throttle_duration_sec=5.0)
            return

        h, w, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        debug_frame = frame.copy()
        detections = []

        red_mask1 = cv2.inRange(hsv, np.array(self.RED_RANGES[0][0]), np.array(self.RED_RANGES[0][1]))
        red_mask2 = cv2.inRange(hsv, np.array(self.RED_RANGES[1][0]), np.array(self.RED_RANGES[1][1]))
        all_masks = {'R': cv2.bitwise_or(red_mask1, red_mask2)}
        for color_name, (lower, upper) in self.COLOR_RANGES.items():
            all_masks[color_name] = cv2.inRange(hsv, np.array(lower), np.array(upper))

        kernel = np.ones((5, 5), np.uint8)
        for color_name, mask in all_masks.items():
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            self.publish_mask(color_name, mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.MIN_PIXEL_COUNT:
                    continue

                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']

                distance = self.get_depth_at(int(cx), int(cy))
                angle = self.pixel_to_angle(cx, w)
                zone = self.pixel_to_zone(cx, w)

                shape = self.classify_shape(contour, area)
                if shape is None:
                    continue  # 노이즈로 보고 아예 detections에 안 넣음

                is_target = bool(
                    self.target_color and self.target_shape
                    and color_name == self.target_color and shape == self.target_shape
                )

                det = {
                    'color': color_name,
                    'angle': round(angle, 4),
                    'zone': zone,
                    'is_target': is_target,
                    'shape': shape,
                    'area': int(area),
                }
                if distance is not None:
                    det['distance'] = round(distance, 3)
                detections.append(det)

                color_bgr = self.DEBUG_COLORS_BGR.get(color_name, (255, 255, 255))
                cv2.drawContours(debug_frame, [contour], -1, color_bgr, 2)
                cv2.drawMarker(debug_frame, (int(cx), int(cy)),
                                (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
                label = f'{color_name}:{shape} {zone} area={int(area)}'
                cv2.putText(debug_frame, label, (int(cx) - 40, int(cy) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

        out = String()
        out.data = json.dumps(detections)
        self.detections_pub.publish(out)
        self.publish_debug_image(debug_frame)

    def classify_shape(self, contour, area):
        """삼각형/네모/십자는 꼭짓점개수, 원형은 circularity+fill_ratio 둘 다 통과해야 인정.
        뭐에도 확실히 안 맞으면 None(노이즈로 보고 무시)."""
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return None

        epsilon = max(0.02 * perimeter, 3.0)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        compactness = area / (perimeter * perimeter)

        if vertices == 3:
            return 'triangle'
        elif vertices == 4:
            return 'triangle' if compactness < 0.055 else 'square'
        elif vertices in (11, 12, 13):
            return 'cross'

        circularity = 4 * math.pi * area / (perimeter * perimeter)
        (_, _), radius = cv2.minEnclosingCircle(contour)
        circle_area = math.pi * radius * radius
        fill_ratio = area / circle_area if circle_area > 0 else 0.0

        if circularity >= self.CIRCLE_MIN_CIRCULARITY and fill_ratio >= self.CIRCLE_MIN_FILL_RATIO:
            return 'circle'
        return None

    def get_depth_at(self, x, y):
        if self.latest_depth is None:
            return None
        h, w = self.latest_depth.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        value = float(self.latest_depth[y, x])
        if math.isnan(value) or math.isinf(value) or value <= 0.0:
            return None
        return value

    def pixel_to_angle(self, centroid_x, image_width):
        half_fov_rad = math.radians(self.HORIZONTAL_FOV_DEG / 2.0)
        normalized = (centroid_x / image_width) - 0.5
        return -normalized * 2.0 * half_fov_rad

    def pixel_to_zone(self, centroid_x, image_width):
        """화면 가로 3등분 기준 좌/중/우 판정."""
        third = image_width / 3.0
        if centroid_x < third:
            return 'left'
        elif centroid_x < third * 2:
            return 'center'
        else:
            return 'right'

    def publish_mask(self, color_name, mask):
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
            self.mask_pubs[color_name].publish(mask_msg)
        except Exception:
            pass

    def publish_debug_image(self, frame):
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
EOF
