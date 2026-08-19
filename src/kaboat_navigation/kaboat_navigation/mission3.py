import math
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

from kaboat_navigation.field_config import MISSION_TARGETS, DOCK_SECTORS, MISSION_TARGETS_CONFIG


class Mission3(Node):
    """
    미션 3 - 도킹.
    대회 규정: 좌/중/우 3개 슬롯 중 대회 당일 지정된 색+모양 슬롯 하나에
    정확히 도킹해야 함(색상+모양 둘 다 일치).

    흐름:
      MOVING   : m3s로 이동 (leg 이동)
      SEARCH   : m3s 도착 지점을 앵커로 회전없이 전/후진으로만 밀림 보정하며
                 대기(mission_2 HOLD와 동일 패턴). camera/detections에서
                 TARGET_COLOR+TARGET_SHAPE를 최근 CONFIRM_WINDOW개 프레임 중
                 CONFIRM_COUNT번 이상 잡으면 확정, 평균각도로 좌/중/우 판단.
      APPROACH : 확정된 섹터의 DOCK_SECTORS 좌표로 leg 이동(도착반경 더 타이트).
      DOCKED   : 3초 정지 - 도킹 완료 판정.
      TURNING  : 확정된 섹터 반대쪽(벽에서 먼 쪽)으로 제자리 회전해서 이탈 준비.
      EXIT     : m3e로 leg 이동. 도착하면 done 발행.

    --- 좌표이동(leg) 로직 (mission1.py/mission0.py와 동일 원칙) ---
    REACQUIRE: 정지+SOL확보 대기 -> 방위각/조향 커밋(전진+회전 같이).
    CRUISE   : 커밋값 유지, 거리만 체크. 오버슈트(7초 유예 후 재이탈)시 ALIGN.
    ALIGN    : 전진없이 제자리회전만으로 목표방향 재조준(±15도) 후 CRUISE 복귀.
    도착 후 3초 정지 후 다음 단계.

    --- 위치유지(SEARCH) 로직 (mission_2 HOLD와 동일, 실측 검증) ---
    회전 없이 앵커가 배 앞쪽이면 전진, 뒤쪽이면 후진만. 데드밴드 안이면 정지.
    좌우밀림은 회전없이 못 잡으므로 데드밴드를 넉넉하게(2.0m) 잡아서 우회.

    목표 색/모양은 field_config.py의 MISSION_TARGETS_CONFIG에서 가져옴.
    """

    MY_MISSION = 'mission_3'

    TARGET_COLOR = MISSION_TARGETS_CONFIG['mission_3']['color']
    TARGET_SHAPE = MISSION_TARGETS_CONFIG['mission_3']['shape']

    CONFIRM_WINDOW = 10
    CONFIRM_COUNT = 4
    SECTOR_ANGLE_THRESHOLD = 0.2

    DOCKED_TIME = 3.0

    TURN_SPEED_LINEAR = 0.15
    TURN_SPEED_ANGULAR = 0.5
    TURN_DIRECTION_DEFAULT = 1.0
    TURN_COMPLETE_TOLERANCE_DEG = 15.0

    # ---- leg 이동(MOVING/APPROACH/EXIT 공용) ----
    LEG_ARRIVAL_RADIUS_M = 1.0          # MOVING/EXIT - 존 진입용, 널널
    LEG_APPROACH_ARRIVAL_RADIUS_M = 0.6  # APPROACH - 실제 도킹 슬롯 접근, 타이트
    LEG_OVERSHOOT_MARGIN_M = 0.5
    LEG_MIN_OVERSHOOT_CHECK_SEC = 7.0
    LEG_CRUISE_LINEAR = 0.4
    LEG_APPROACH_LINEAR = 0.25          # APPROACH는 도킹 정밀접근이라 좀 더 저속
    LEG_ANGULAR_MAX = 1.0
    LEG_K_ANGULAR = 0.6
    LEG_ALIGN_TOLERANCE_DEG = 15.0
    LEG_ARRIVAL_PAUSE_SEC = 3.0

    # ---- SEARCH(회전없는 위치유지, mission_2 실측값과 동일) ----
    HOLD_DEADBAND_M = 2.0
    HOLD_PUSH_FORWARD = 0.2
    HOLD_PUSH_BACKWARD = 0.15

    def __init__(self):
        super().__init__('mission_3')

        self.active = False
        self.phase = 'MOVING'   # MOVING / SEARCH / APPROACH / DOCKED / TURNING / EXIT

        self.current_lat = None
        self.current_lon = None
        self.current_heading = None
        self._fix_this_tick = False

        self.hold_lat = None
        self.hold_lon = None

        self.detection_history = []
        self.confirmed_sector = None

        self.docked_start_time = None
        self.entry_heading = None
        self.turn_direction = self.TURN_DIRECTION_DEFAULT

        # leg 이동 상태
        self._leg_state = None
        self._leg_target = None
        self._leg_arrival_radius = self.LEG_ARRIVAL_RADIUS_M
        self._leg_cruise_linear = self.LEG_CRUISE_LINEAR
        self._leg_committed_linear = 0.0
        self._leg_committed_angular = 0.0
        self._leg_min_dist = None
        self._leg_cruise_start = None

        self._pause_active = False
        self._pause_start = None
        self._pause_next_phase = None

        self.create_subscription(String, 'mission/active', self.active_cb, 10)
        self.create_subscription(String, 'mission/started', self.started_cb, 10)
        self.create_subscription(String, 'camera/detections', self.detections_cb, 10)
        self.create_subscription(String, 'kaboat/gps_nav', self.gps_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, 'cmd_mission', 10)
        self.done_pub = self.create_publisher(String, 'mission/done', 10)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('미션3(도킹) 노드 대기 중')

    def active_cb(self, msg):
        self.active = (msg.data == self.MY_MISSION)

    def started_cb(self, msg):
        if msg.data == self.MY_MISSION:
            self.get_logger().info('★ 미션3 시작 (MOVING -> m3s로 이동) ★')
            self.phase = 'MOVING'
            self.hold_lat = None
            self.hold_lon = None
            self.detection_history = []
            self.confirmed_sector = None
            self.docked_start_time = None
            self.entry_heading = None
            self.turn_direction = self.TURN_DIRECTION_DEFAULT
            self._leg_state = None
            self._pause_active = False

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
            return
        if self.current_lat is not None and self.current_heading is not None:
            self._fix_this_tick = True

    def detections_cb(self, msg):
        if not self.active or self.phase != 'SEARCH':
            return
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        match = next(
            (d for d in detections
             if d.get('color') == self.TARGET_COLOR and d.get('shape') == self.TARGET_SHAPE),
            None
        )

        if match is not None:
            self.detection_history.append((True, match['angle']))
        else:
            self.detection_history.append((False, None))

        if len(self.detection_history) > self.CONFIRM_WINDOW:
            self.detection_history.pop(0)

        self.check_confirm()

    def check_confirm(self):
        matched = [angle for is_match, angle in self.detection_history if is_match]
        if len(matched) >= self.CONFIRM_COUNT:
            avg_angle = sum(matched) / len(matched)
            self.confirmed_sector = self.angle_to_sector(avg_angle)
            self.get_logger().info(
                f'목표 확정! ({self.TARGET_COLOR}/{self.TARGET_SHAPE}) '
                f'평균각도={math.degrees(avg_angle):.1f}도 -> 섹터={self.confirmed_sector}')
            self.phase = 'APPROACH'
            self._leg_state = None

    def angle_to_sector(self, angle_rad):
        if angle_rad > self.SECTOR_ANGLE_THRESHOLD:
            return 'left'
        elif angle_rad < -self.SECTOR_ANGLE_THRESHOLD:
            return 'right'
        else:
            return 'center'

    def control_loop(self):
        if not self.active:
            return

        if self._pause_active:
            self._tick_pause()
            self._fix_this_tick = False
            return

        if self.phase == 'MOVING':
            self.run_moving()
        elif self.phase == 'SEARCH':
            self.run_search_hold()
        elif self.phase == 'APPROACH':
            self.run_approach()
        elif self.phase == 'DOCKED':
            self.run_docked()
        elif self.phase == 'TURNING':
            self.run_turning()
        elif self.phase == 'EXIT':
            self.run_exit()

        self._fix_this_tick = False

    def run_moving(self):
        target = MISSION_TARGETS.get('m3s')
        if target is None:
            self.get_logger().warn('m3s 좌표 없음 (field_config.py 확인)', throttle_duration_sec=5.0)
            return
        if self._leg_state is None:
            self._leg_start(*target, arrival_radius=self.LEG_ARRIVAL_RADIUS_M,
                             cruise_linear=self.LEG_CRUISE_LINEAR)
        result = self._leg_tick()
        if result == 'ARRIVED':
            self.get_logger().info('m3s 도착 - 정지 후 SEARCH 시작 (목표 색/모양 탐색+위치고정)')
            self.hold_lat, self.hold_lon = self.current_lat, self.current_lon
            self._start_pause(next_phase='SEARCH')

    def run_search_hold(self):
        """mission_2의 HOLD와 동일 패턴 - 회전 없이 전/후진으로만 밀림 보정."""
        if self.current_lat is None or self.current_heading is None or self.hold_lat is None:
            self.cmd_pub.publish(Twist())
            return

        bearing = self.bearing_deg(self.current_lat, self.current_lon, self.hold_lat, self.hold_lon)
        dist = self.distance_m(self.current_lat, self.current_lon, self.hold_lat, self.hold_lon)
        heading_error = self.normalize_angle_deg(bearing - self.current_heading)

        cmd = Twist()
        if dist <= self.HOLD_DEADBAND_M:
            cmd.linear.x = 0.0
        elif abs(heading_error) <= 90.0:
            cmd.linear.x = self.HOLD_PUSH_FORWARD
        else:
            cmd.linear.x = -self.HOLD_PUSH_BACKWARD
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    def run_approach(self):
        target = DOCK_SECTORS.get(self.confirmed_sector)
        if target is None:
            self.get_logger().warn(
                f'{self.confirmed_sector} 섹터 좌표 없음 (field_config.py 확인)',
                throttle_duration_sec=5.0)
            return
        if self._leg_state is None:
            self._leg_start(*target, arrival_radius=self.LEG_APPROACH_ARRIVAL_RADIUS_M,
                             cruise_linear=self.LEG_APPROACH_LINEAR)
        result = self._leg_tick()
        if result == 'ARRIVED':
            self.get_logger().info(f'{self.confirmed_sector} 섹터 도착 - 도킹 완료 대기(3초)')
            self.cmd_pub.publish(Twist())
            self.phase = 'DOCKED'
            self.docked_start_time = self.get_clock().now()

    def run_docked(self):
        self.cmd_pub.publish(Twist())
        elapsed = (self.get_clock().now() - self.docked_start_time).nanoseconds / 1e9
        if elapsed >= self.DOCKED_TIME:
            self.turn_direction = self.decide_turn_direction()
            self.entry_heading = self.current_heading
            self.get_logger().info(
                f'3초 정지 완료 - 섹터={self.confirmed_sector} → '
                f'회전방향={"시계" if self.turn_direction > 0 else "반시계"}로 이탈 회전 시작')
            self.phase = 'TURNING'

    def decide_turn_direction(self):
        if self.confirmed_sector == 'left':
            return 1.0
        if self.confirmed_sector == 'right':
            return -1.0
        return self.TURN_DIRECTION_DEFAULT

    def run_turning(self):
        if self.current_heading is None or self.entry_heading is None:
            return

        target_heading_deg = (self.entry_heading + 180.0) % 360.0
        diff = self.normalize_angle_deg(target_heading_deg - self.current_heading)

        if abs(diff) <= self.TURN_COMPLETE_TOLERANCE_DEG:
            self.get_logger().info('회전 완료 - 도킹장 이탈 성공, m3e로 이동 시작')
            self.phase = 'EXIT'
            self._leg_state = None
            return

        cmd = Twist()
        cmd.linear.x = self.TURN_SPEED_LINEAR
        cmd.angular.z = self.turn_direction * self.TURN_SPEED_ANGULAR
        self.cmd_pub.publish(cmd)

    def run_exit(self):
        target = MISSION_TARGETS.get('m3e')
        if target is None:
            self.get_logger().warn('m3e 좌표 없음 (field_config.py 확인)', throttle_duration_sec=5.0)
            return
        if self._leg_state is None:
            self._leg_start(*target, arrival_radius=self.LEG_ARRIVAL_RADIUS_M,
                             cruise_linear=self.LEG_CRUISE_LINEAR)
        result = self._leg_tick()
        if result == 'ARRIVED':
            self.get_logger().info('m3e 도달! 미션3 완료')
            self.cmd_pub.publish(Twist())
            done = String()
            done.data = self.MY_MISSION
            self.done_pub.publish(done)

    # ==== 좌표이동(leg) 로직 - mission1.py/mission0.py와 동일 ====

    def _leg_start(self, target_lat, target_lon, arrival_radius, cruise_linear):
        self._leg_target = (target_lat, target_lon)
        self._leg_arrival_radius = arrival_radius
        self._leg_cruise_linear = cruise_linear
        self._leg_state = 'REACQUIRE'
        self._leg_committed_linear = 0.0
        self._leg_committed_angular = 0.0
        self._leg_min_dist = None
        self._leg_cruise_start = None

    def _leg_tick(self):
        if self._leg_state == 'REACQUIRE':
            self.cmd_pub.publish(Twist())
            if not self._fix_this_tick or self.current_lat is None:
                return 'WAITING'
            return self._leg_commit_and_cruise()

        if self._leg_state == 'ALIGN':
            if not self._fix_this_tick or self.current_lat is None:
                self.cmd_pub.publish(Twist())
                return 'WAITING'
            bearing = self.bearing_deg(self.current_lat, self.current_lon, *self._leg_target)
            heading_error = self.normalize_angle_deg(bearing - self.current_heading)
            if abs(heading_error) <= self.LEG_ALIGN_TOLERANCE_DEG:
                self._leg_committed_linear = self._leg_cruise_linear
                self._leg_committed_angular = 0.0
                self._leg_min_dist = None
                self._leg_cruise_start = self.get_clock().now()
                self._leg_state = 'CRUISE'
                return 'MOVING'
            cmd = Twist()
            cmd.angular.z = max(-self.LEG_ANGULAR_MAX,
                                 min(self.LEG_ANGULAR_MAX, self.LEG_K_ANGULAR * math.radians(heading_error)))
            self.cmd_pub.publish(cmd)
            return 'MOVING'

        if self._leg_state == 'CRUISE':
            if self.current_lat is None:
                self._leg_publish_cmd()
                return 'MOVING'
            dist = self.distance_m(self.current_lat, self.current_lon, *self._leg_target)
            if dist <= self._leg_arrival_radius:
                self.cmd_pub.publish(Twist())
                self._leg_state = None
                return 'ARRIVED'
            if self._leg_min_dist is None or dist < self._leg_min_dist:
                self._leg_min_dist = dist
            else:
                elapsed = ((self.get_clock().now() - self._leg_cruise_start).nanoseconds / 1e9
                           if self._leg_cruise_start else 0.0)
                if elapsed >= self.LEG_MIN_OVERSHOOT_CHECK_SEC and dist > self._leg_min_dist + self.LEG_OVERSHOOT_MARGIN_M:
                    self.get_logger().warn(
                        f'오버슈트 감지(최소 {self._leg_min_dist:.2f}m -> {dist:.2f}m) - 정지 후 재정렬')
                    self.cmd_pub.publish(Twist())
                    self._leg_state = 'ALIGN'
                    return 'MOVING'
            self._leg_publish_cmd()
            return 'MOVING'

        return 'WAITING'

    def _leg_commit_and_cruise(self):
        dist = self.distance_m(self.current_lat, self.current_lon, *self._leg_target)
        if dist <= self._leg_arrival_radius:
            self._leg_state = None
            return 'ARRIVED'
        bearing = self.bearing_deg(self.current_lat, self.current_lon, *self._leg_target)
        heading_error = self.normalize_angle_deg(bearing - self.current_heading)
        self._leg_committed_linear = self._leg_cruise_linear
        self._leg_committed_angular = max(-self.LEG_ANGULAR_MAX,
                                           min(self.LEG_ANGULAR_MAX, self.LEG_K_ANGULAR * math.radians(heading_error)))
        self._leg_min_dist = None
        self._leg_cruise_start = self.get_clock().now()
        self._leg_state = 'CRUISE'
        return 'MOVING'

    def _leg_publish_cmd(self):
        cmd = Twist()
        cmd.linear.x = self._leg_committed_linear
        cmd.angular.z = self._leg_committed_angular
        self.cmd_pub.publish(cmd)

    def _start_pause(self, next_phase):
        self.cmd_pub.publish(Twist())
        self._pause_active = True
        self._pause_start = self.get_clock().now()
        self._pause_next_phase = next_phase

    def _tick_pause(self):
        self.cmd_pub.publish(Twist())
        elapsed = (self.get_clock().now() - self._pause_start).nanoseconds / 1e9
        if elapsed >= self.LEG_ARRIVAL_PAUSE_SEC:
            self.phase = self._pause_next_phase
            self._pause_active = False
            self._leg_state = None

    def normalize_angle_deg(self, angle_deg):
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0
        return angle_deg

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
    node = Mission3()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()    ARRIVAL_RADIUS = 1.0
    DOCK_ARRIVAL_RADIUS = 1.0
    FINISH_RADIUS = 1.0
    DOCKED_TIME = 3.0

    CONFIRM_WINDOW = 10
    CONFIRM_COUNT = 4

    SECTOR_ANGLE_THRESHOLD = 0.2

    KP_DIST = 0.3
    KP_ANGLE = 0.5

    CRUISE_SPEED = 0.3
    K_ANGLE = 0.8

    TURN_SPEED_LINEAR = 0.15
    TURN_SPEED_ANGULAR = 0.5
    TURN_DIRECTION_DEFAULT = 1.0    # 가운데 섹터거나 판단불가시 기본값 (1.0=시계, -1.0=반시계)
    TURN_COMPLETE_TOLERANCE_DEG = 15.0

    def __init__(self):
        super().__init__('mission_3')

        self.active = False
        self.phase = 'MOVING'   # MOVING / SEARCH / APPROACH / DOCKED / TURNING / EXIT

        self.current_lat = None
        self.current_lon = None
        self.current_heading = None

        self.hold_lat = None
        self.hold_lon = None

        self.detection_history = []
        self.confirmed_sector = None

        self.docked_start_time = None
        self.entry_heading = None
        self.turn_direction = self.TURN_DIRECTION_DEFAULT

        self.create_subscription(String, 'mission/active', self.active_cb, 10)
        self.create_subscription(String, 'mission/started', self.started_cb, 10)
        self.create_subscription(String, 'camera/detections', self.detections_cb, 10)
        self.create_subscription(String, 'kaboat/gps_nav', self.gps_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, 'cmd_mission', 10)
        self.done_pub = self.create_publisher(String, 'mission/done', 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('미션3(도킹) 노드 대기 중')

    def active_cb(self, msg):
        self.active = (msg.data == self.MY_MISSION)

    def started_cb(self, msg):
        if msg.data == self.MY_MISSION:
            self.get_logger().info('★ 미션3 시작 (MOVING -> m3s로 이동) ★')
            self.phase = 'MOVING'
            self.hold_lat = None
            self.hold_lon = None
            self.detection_history = []
            self.confirmed_sector = None
            self.docked_start_time = None
            self.entry_heading = None
            self.turn_direction = self.TURN_DIRECTION_DEFAULT

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

    def detections_cb(self, msg):
        if not self.active or self.phase != 'SEARCH':
            return
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        match = next(
            (d for d in detections
             if d.get('color') == self.TARGET_COLOR and d.get('shape') == self.TARGET_SHAPE),
            None
        )

        if match is not None:
            self.detection_history.append((True, match['angle']))
        else:
            self.detection_history.append((False, None))

        if len(self.detection_history) > self.CONFIRM_WINDOW:
            self.detection_history.pop(0)

        self.check_confirm()

    def check_confirm(self):
        matched = [angle for is_match, angle in self.detection_history if is_match]
        if len(matched) >= self.CONFIRM_COUNT:
            avg_angle = sum(matched) / len(matched)
            self.confirmed_sector = self.angle_to_sector(avg_angle)
            self.get_logger().info(
                f'목표 확정! ({self.TARGET_COLOR}/{self.TARGET_SHAPE}) '
                f'평균각도={math.degrees(avg_angle):.1f}도 -> 섹터={self.confirmed_sector}')
            self.phase = 'APPROACH'

    def angle_to_sector(self, angle_rad):
        if angle_rad > self.SECTOR_ANGLE_THRESHOLD:
            return 'left'
        elif angle_rad < -self.SECTOR_ANGLE_THRESHOLD:
            return 'right'
        else:
            return 'center'

    def control_loop(self):
        if not self.active:
            return

        if self.phase == 'MOVING':
            self.run_moving()
        elif self.phase == 'SEARCH':
            self.run_search_hold()
        elif self.phase == 'APPROACH':
            self.run_approach()
        elif self.phase == 'DOCKED':
            self.run_docked()
        elif self.phase == 'TURNING':
            self.run_turning()
        elif self.phase == 'EXIT':
            self.run_exit()

    def run_moving(self):
        if self.current_lat is None or self.current_lon is None or self.current_heading is None:
            return
        target = MISSION_TARGETS.get('m3s')
        if target is None:
            self.get_logger().warn('m3s 좌표 없음 (field_config.py 확인)', throttle_duration_sec=5.0)
            return
        target_lat, target_lon = target

        distance = self.distance_m(self.current_lat, self.current_lon, target_lat, target_lon)
        if distance <= self.ARRIVAL_RADIUS:
            self.get_logger().info('m3s 도착 → SEARCH 시작 (목표 색/모양 탐색 + 위치고정)')
            self.hold_lat, self.hold_lon = self.current_lat, self.current_lon
            self.phase = 'SEARCH'
            return

        self.drive_toward_gps(target_lat, target_lon)

    def run_search_hold(self):
        if self.current_lat is None or self.current_lon is None or self.current_heading is None:
            return
        if self.hold_lat is None:
            return

        distance = self.distance_m(self.current_lat, self.current_lon, self.hold_lat, self.hold_lon)
        bearing = self.bearing_deg(self.current_lat, self.current_lon, self.hold_lat, self.hold_lon)
        angle_error_deg = self.normalize_angle_deg(bearing - self.current_heading)

        cmd = Twist()
        cmd.linear.x = max(-0.3, min(0.3, self.KP_DIST * distance))
        cmd.angular.z = max(-0.5, min(0.5, self.KP_ANGLE * math.radians(angle_error_deg)))
        self.cmd_pub.publish(cmd)

    def run_approach(self):
        if self.current_lat is None or self.current_lon is None or self.current_heading is None:
            return
        target = DOCK_SECTORS.get(self.confirmed_sector)
        if target is None:
            self.get_logger().warn(
                f'{self.confirmed_sector} 섹터 좌표 없음 (field_config.py 확인)',
                throttle_duration_sec=5.0)
            return
        target_lat, target_lon = target

        distance = self.distance_m(self.current_lat, self.current_lon, target_lat, target_lon)
        if distance <= self.DOCK_ARRIVAL_RADIUS:
            self.get_logger().info(f'{self.confirmed_sector} 섹터 도착 → 도킹 완료 대기(3초)')
            self.cmd_pub.publish(Twist())
            self.phase = 'DOCKED'
            self.docked_start_time = self.get_clock().now()
            return

        self.drive_toward_gps(target_lat, target_lon)

    def run_docked(self):
        self.cmd_pub.publish(Twist())
        elapsed = (self.get_clock().now() - self.docked_start_time).nanoseconds / 1e9
        if elapsed >= self.DOCKED_TIME:
            self.turn_direction = self.decide_turn_direction()
            self.entry_heading = self.current_heading
            self.get_logger().info(
                f'3초 정지 완료 - 섹터={self.confirmed_sector} → '
                f'회전방향={"시계" if self.turn_direction > 0 else "반시계"}로 이탈 회전 시작')
            self.phase = 'TURNING'

    def decide_turn_direction(self):
        """도킹된 섹터 기준으로 벽에서 먼 쪽으로 회전하도록 방향 결정."""
        if self.confirmed_sector == 'left':
            return 1.0    # 왼쪽 슬롯 -> 시계(오른쪽)로 회전
        if self.confirmed_sector == 'right':
            return -1.0   # 오른쪽 슬롯 -> 반시계(왼쪽)로 회전
        return self.TURN_DIRECTION_DEFAULT

    def run_turning(self):
        if self.current_heading is None or self.entry_heading is None:
            return

        target_heading_deg = (self.entry_heading + 180.0) % 360.0
        diff = self.normalize_angle_deg(target_heading_deg - self.current_heading)

        if abs(diff) <= self.TURN_COMPLETE_TOLERANCE_DEG:
            self.get_logger().info('회전 완료 - 도킹장 이탈 성공, m3e로 이동 시작')
            self.phase = 'EXIT'
            return

        cmd = Twist()
        cmd.linear.x = self.TURN_SPEED_LINEAR
        cmd.angular.z = self.turn_direction * self.TURN_SPEED_ANGULAR
        self.cmd_pub.publish(cmd)

    def run_exit(self):
        if self.current_lat is None or self.current_lon is None or self.current_heading is None:
            return
        target = MISSION_TARGETS.get('m3e')
        if target is None:
            self.get_logger().warn('m3e 좌표 없음 (field_config.py 확인)', throttle_duration_sec=5.0)
            return
        target_lat, target_lon = target

        distance = self.distance_m(self.current_lat, self.current_lon, target_lat, target_lon)
        if distance <= self.FINISH_RADIUS:
            self.get_logger().info('m3e 도달! 미션3 완료')
            self.cmd_pub.publish(Twist())
            done = String()
            done.data = self.MY_MISSION
            self.done_pub.publish(done)
            return

        self.drive_toward_gps(target_lat, target_lon)

    def drive_toward_gps(self, target_lat, target_lon):
        bearing = self.bearing_deg(self.current_lat, self.current_lon, target_lat, target_lon)
        angle_error_deg = self.normalize_angle_deg(bearing - self.current_heading)

        cmd = Twist()
        cmd.linear.x = self.CRUISE_SPEED
        cmd.angular.z = self.angle_to_angular(math.radians(angle_error_deg))
        self.cmd_pub.publish(cmd)

    def angle_to_angular(self, target_angle_rad):
        return max(-1.0, min(1.0, self.K_ANGLE * target_angle_rad))

    def normalize_angle_deg(self, angle_deg):
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0
        return angle_deg

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
    node = Mission3()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
