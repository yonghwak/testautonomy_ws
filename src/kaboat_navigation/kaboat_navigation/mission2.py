import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data

from kaboat_navigation.field_config import MISSION_TARGETS, TRANSIT_ARRIVAL_RADIUS_M


class Mission2(Node):
    """
    미션 2 - 위치유지.
    대회 규정: 대상 부표 5m 이내에서 5초간 위치를 정지 유지해야 완료. 충돌 시
    패널티. 색상 구분 불필요(최근접 부표를 대상으로 인정).

    흐름 (3단계 - 실측 검증 반영):
      MOVING   : m2s로 GPS 이동
      APPROACH : 라이다로 최근접 클러스터(부표)까지의 거리/방위 보며 접근.
                 목표거리(TARGET_DIST) 이내 들어오면 그 순간 GPS 좌표를
                 앵커로 고정하고 HOLD로 전환 - 라이다는 여기서 역할 끝.
      HOLD     : 라이다 더 이상 안 봄. 고정된 GPS 앵커 기준으로 회전 없이
                 앞/뒤로만 고정된 힘으로 미는 방식으로 유지 - 실측 검증됨.
                 데드밴드 안에서 5초 연속 버티면 성공, m2e로 이동 후 done.

    실측으로 확정된 것들:
    - 회전 넣으면 SOL_COMPUTED가 이동 중 자주 풀려서 헤딩값이 튀고, 그 결과
      배가 좌우로 급하게 흔들리며 오히려 더 밀려나는 현상 확인됨 -> HOLD 단계는
      회전(angular.z) 없이 전/후진만으로 제어.
    - 전진/후진 힘을 다르게 줌(급류가 배를 뒤로 미는 상황이라 후진 보정은 더 약하게).
    - 좌우 밀림은 회전 없이는 원천적으로 못 잡음 -> 규정(5m 이내) 여유를 감안해
      DEADBAND_M을 넉넉하게 잡아서 "그 정도는 유지 성공으로 인정"하는 방식으로 우회.
    """

    MY_MISSION = 'mission_2'

    # ---- APPROACH(라이다) ----
    TARGET_DIST = 5.0            # 이 거리 이내 들어오면 앵커 고정하고 HOLD 전환
    FORWARD_CONE_DEG = 90.0
    LOCK_SWITCH_RADIUS_DEG = 15.0
    APPROACH_CRAWL_SPEED = 0.15  # 부표 안 보일 때 천천히 탐색 전진
    APPROACH_LINEAR_MAX = 0.3
    APPROACH_K_LINEAR = 0.15
    APPROACH_K_ANGULAR = 0.02
    CLUSTER_JUMP_THRESHOLD = 0.3

    # ---- HOLD(GPS, 실측 확정값) ----
    DEADBAND_M = 2.0            # 좌우밀림은 회전없이 못 잡아서 규정(5m) 여유 감안해 넉넉하게
    OUTER_HYSTERESIS_M = 0.5
    HOLD_SECONDS = 5.0
    PUSH_FORWARD_LINEAR = 0.2    # 앵커가 앞쪽일 때
    PUSH_BACKWARD_LINEAR = 0.15  # 앵커가 뒤쪽일 때(급류가 뒤로 밀어줘서 더 약하게)

    def __init__(self):
        super().__init__('mission_2')
        self.active = False
        self.phase = 'MOVING'
        self.current_heading = None
        self.current_lat = None
        self.current_lon = None

        self.locked_target = None   # 라이다 클러스터 (APPROACH 단계용)
        self.anchor_lat = None      # HOLD 단계에서 쓸 GPS 앵커
        self.anchor_lon = None

        self.holding = False
        self.hold_start_time = None
        self.exiting = False

        self.create_subscription(String, 'mission/active', self.active_cb, 10)
        self.create_subscription(String, 'mission/started', self.started_cb, 10)
        self.create_subscription(String, 'kaboat/gps_nav', self.gps_cb, 10)
        self.create_subscription(
            LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)

        self.cmd_pub = self.create_publisher(Twist, 'cmd_mission', 10)
        self.heading_pub = self.create_publisher(Float32, 'goal/heading', 10)
        self.done_pub = self.create_publisher(String, 'mission/done', 10)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            'mission_2(위치유지) 노드 시작 - 라이다로 접근, GPS로 유지(3단계)')

    def active_cb(self, msg):
        self.active = (msg.data == self.MY_MISSION)

    def started_cb(self, msg):
        if msg.data == self.MY_MISSION:
            self.phase = 'MOVING'
            self.locked_target = None
            self.anchor_lat = None
            self.anchor_lon = None
            self.holding = False
            self.hold_start_time = None
            self.exiting = False
            self.get_logger().info('mission_2 시작 - 상태 초기화 (m2s로 이동)')

    def gps_cb(self, msg):
        try:
            for part in msg.data.split(','):
                if part.startswith('lat='):
                    self.current_lat = float(part.split('=')[1])
                elif part.startswith('lon='):
                    self.current_lon = float(part.split('=')[1])
                elif part.startswith('imu_heading='):
                    self.current_heading = float(part.split('=')[1])
        except (ValueError, IndexError):
            pass

    def scan_cb(self, msg):
        # 라이다는 APPROACH 단계에서만 씀 - MOVING/HOLD에서는 무시
        if not self.active or self.exiting or self.phase != 'APPROACH':
            return

        n = len(msg.ranges)
        angles = [msg.angle_min + i * msg.angle_increment for i in range(n)]
        clusters = self.cluster_scan(msg.ranges, angles, msg.range_min, msg.range_max, max_range=10.0)

        cone_rad = math.radians(self.FORWARD_CONE_DEG)
        candidates = [c for c in clusters if abs(c['center_angle']) <= cone_rad]
        if not candidates:
            return

        if self.locked_target is not None:
            prev_bearing_rad = math.radians(self.locked_target['bearing'])
            switch_rad = math.radians(self.LOCK_SWITCH_RADIUS_DEG)
            same_target = [
                c for c in candidates
                if abs(c['center_angle'] - prev_bearing_rad) <= switch_rad
            ]
            chosen = min(same_target, key=lambda c: c['min_range']) if same_target \
                else min(candidates, key=lambda c: c['min_range'])
        else:
            chosen = min(candidates, key=lambda c: c['min_range'])

        self.locked_target = {
            'range': chosen['min_range'],
            'bearing': math.degrees(chosen['center_angle']),
        }

    def control_loop(self):
        if not self.active:
            return

        if self.phase == 'MOVING':
            self.run_moving()
            return

        if self.exiting:
            self.run_exit()
            return

        if self.phase == 'APPROACH':
            self.run_approach()
        elif self.phase == 'HOLD':
            self.run_hold()

    def run_moving(self):
        if self.current_lat is None:
            return

        start_point = MISSION_TARGETS.get('m2s')
        if start_point is None:
            self.get_logger().warn('m2s 좌표 없음 (field_config.py 확인)', throttle_duration_sec=5.0)
            return

        dist = self.distance_m(self.current_lat, self.current_lon, *start_point)
        if dist > TRANSIT_ARRIVAL_RADIUS_M:
            brg = self.bearing_deg(self.current_lat, self.current_lon, *start_point)
            h_msg = Float32()
            h_msg.data = brg
            self.heading_pub.publish(h_msg)
            cmd = Twist()
            cmd.linear.x = 0.2
            self.cmd_pub.publish(cmd)
            return

        self.get_logger().info('m2s 도착 - 라이다 접근(APPROACH) 시작')
        self.phase = 'APPROACH'

    def run_approach(self):
        """라이다로 부표까지 접근. TARGET_DIST 이내 들어오면 그 순간 GPS 좌표를
        앵커로 고정하고 HOLD로 전환 - 이후로는 라이다 값 안 씀."""
        cmd = Twist()

        if self.locked_target is None:
            cmd.linear.x = self.APPROACH_CRAWL_SPEED
            self.cmd_pub.publish(cmd)
            return

        r = self.locked_target['range']
        bearing_rel = self.locked_target['bearing']

        if r <= self.TARGET_DIST:
            if self.current_lat is None:
                self.get_logger().warn('목표거리 도달했지만 GPS 없음 - 앵커 고정 대기', throttle_duration_sec=2.0)
                self.cmd_pub.publish(Twist())
                return
            self.anchor_lat = self.current_lat
            self.anchor_lon = self.current_lon
            self.get_logger().info(
                f'★ 목표거리({self.TARGET_DIST}m) 도달 - GPS 앵커 고정: '
                f'({self.anchor_lat:.8f}, {self.anchor_lon:.8f}) - HOLD 전환 ★')
            self.phase = 'HOLD'
            self.cmd_pub.publish(Twist())
            return

        error = r - self.TARGET_DIST
        cmd.linear.x = max(0.0, min(self.APPROACH_LINEAR_MAX, self.APPROACH_K_LINEAR * error))
        cmd.angular.z = max(-0.5, min(0.5, self.APPROACH_K_ANGULAR * bearing_rel))
        self.cmd_pub.publish(cmd)

    def run_hold(self):
        """실측 검증된 방식: 회전 없이, 앵커가 배 앞쪽에 있으면 전진, 뒤쪽이면 후진.
        전/후진 힘은 서로 다르게(급류 방향 고려). 데드밴드+히스테리시스로
        5초 연속 유지 확인."""
        cmd = Twist()

        if self.current_lat is None or self.current_heading is None or self.anchor_lat is None:
            self.cmd_pub.publish(cmd)
            return

        bearing = self.bearing_deg(self.current_lat, self.current_lon, self.anchor_lat, self.anchor_lon)
        dist = self.distance_m(self.current_lat, self.current_lon, self.anchor_lat, self.anchor_lon)
        heading_error = self.normalize_angle(bearing - self.current_heading)

        outer_limit = self.DEADBAND_M + self.OUTER_HYSTERESIS_M
        if not self.holding:
            if dist <= self.DEADBAND_M:
                self.holding = True
                self.hold_start_time = self.get_clock().now()
                self.get_logger().info('데드밴드 진입 - 5초 유지 타이머 시작')
        else:
            if dist > outer_limit:
                self.holding = False
                self.hold_start_time = None
                self.get_logger().warn(f'{outer_limit}m 밖으로 이탈 - 타이머 리셋')

        if dist <= self.DEADBAND_M:
            cmd.linear.x = 0.0
        elif abs(heading_error) <= 90.0:
            cmd.linear.x = self.PUSH_FORWARD_LINEAR    # 앵커가 앞쪽 -> 전진
        else:
            cmd.linear.x = -self.PUSH_BACKWARD_LINEAR  # 앵커가 뒤쪽 -> 후진
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

        if self.holding and self.hold_start_time is not None:
            elapsed = (self.get_clock().now() - self.hold_start_time).nanoseconds / 1e9
            if elapsed >= self.HOLD_SECONDS:
                self.get_logger().info('mission_2 5초 위치유지 성공')
                self.holding = False
                end_point = MISSION_TARGETS.get('m2e')
                if end_point is not None:
                    self.exiting = True
                    self.get_logger().info('m2e로 이동 시작')
                else:
                    self.finish()

    def run_exit(self):
        point = MISSION_TARGETS.get('m2e')
        if point is not None and self.current_lat is not None:
            dist_to_point = self.distance_m(self.current_lat, self.current_lon, point[0], point[1])
            if dist_to_point > TRANSIT_ARRIVAL_RADIUS_M:
                brg = self.bearing_deg(self.current_lat, self.current_lon, point[0], point[1])
                h_msg = Float32()
                h_msg.data = brg
                self.heading_pub.publish(h_msg)
                cmd = Twist()
                cmd.linear.x = 0.2
                self.cmd_pub.publish(cmd)
                return

        self.finish()

    def finish(self):
        self.get_logger().info('mission_2 전체 완료')
        done = String()
        done.data = self.MY_MISSION
        self.done_pub.publish(done)

    def normalize_angle(self, angle_deg):
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0
        return angle_deg

    def cluster_scan(self, ranges, angles, range_min, range_max, max_range):
        """연속된 유효 거리값들을 클러스터로 묶는다.
        각 클러스터의 대표각도(중앙 인덱스 각도)와 평균거리를 반환."""
        n = len(ranges)
        effective_max = min(range_max, max_range)
        valid = [r if range_min < r < effective_max else None for r in ranges]

        clusters = []
        start_idx = None
        for i in range(n):
            if valid[i] is not None:
                if start_idx is None:
                    start_idx = i
                elif abs(valid[i] - valid[i - 1]) > self.CLUSTER_JUMP_THRESHOLD:
                    clusters.append((start_idx, i - 1))
                    start_idx = i
            else:
                if start_idx is not None:
                    clusters.append((start_idx, i - 1))
                    start_idx = None
        if start_idx is not None:
            clusters.append((start_idx, n - 1))

        result = []
        for s, e in clusters:
            seg = [valid[i] for i in range(s, e + 1) if valid[i] is not None]
            if not seg:
                continue
            center_idx = (s + e) // 2
            result.append({
                'center_angle': angles[center_idx],
                'min_range': sum(seg) / len(seg),
            })
        return result

    @staticmethod
    def bearing_deg(lat1, lon1, lat2, lon2):
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dlambda = math.radians(lon2 - lon1)
        y = math.sin(dlambda) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
        return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

    @staticmethod
    def distance_m(lat1, lon1, lat2, lon2):
        R = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def main(args=None):
    rclpy.init(args=args)
    node = Mission2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
