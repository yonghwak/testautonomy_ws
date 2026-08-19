import math
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

from kaboat_navigation.field_config import MISSION_TARGETS, MISSION_TARGETS_CONFIG


class Mission4(Node):
    """
    미션 4 - 탐색/선회.
    대회 규정: 지정된 색 부표를 중심으로 360도 완전히 선회해야 완료. 회전방향은
    색상별 고정 - 빨강/초록=시계방향(CW), 흰색=반시계방향(CCW). 모양은 무관.

    흐름:
      MOVING   : m4s로 이동 (leg 이동 - GPS 전용 구간에만 적용, 아래 설명)
      SEARCH   : camera/detections에서 TARGET_COLOR가 연속으로 CONFIRM_STREAK
                 프레임 이상 잡히면 확정, APPROACH로 전환.
      APPROACH : 카메라(색+각도+거리)만으로 부표에 접근 - 라이다 안 씀
                 (camera_node.py가 ZED depth로 이미 거리까지 주므로 라이다
                 매칭이 불필요 - circle_test.py로 검증된 방식).
                 거리가 CIRCLE_START_DIST 이내가 되면 CIRCLE로.
      CIRCLE   : 계속 카메라(색+거리)로 궤도반경 유지하며 원선회. imu_heading을
                 부호 포함으로 누적, 540도(1.5바퀴, 안전마진) 채우면 EXIT로.
                 카메라가 순간적으로 놓치면(DETECTION_STALE_TIMEOUT_SEC) 정지.
      EXIT     : m4e로 이동 (leg 이동).

    --- 좌표이동(leg) 로직 (mission0/1/3.py와 동일 원칙) ---
    MOVING/EXIT처럼 "GPS 좌표 하나로 이동"하는 구간에만 적용. SEARCH/APPROACH/
    CIRCLE은 카메라 실시간 추적이라 이 로직과 무관.
      REACQUIRE: 정지+SOL확보 대기 -> 방위각/조향 커밋(전진+회전 같이).
      CRUISE   : 커밋값 유지, 거리만 체크. 오버슈트(7초 유예 후 재이탈)시 ALIGN.
      ALIGN    : 전진없이 제자리회전만으로 목표방향 재조준(±15도) 후 CRUISE 복귀.
      도착 후 3초 정지 후 다음 단계.

    목표색은 field_config.py의 MISSION_TARGETS_CONFIG에서 가져옴.
    """

    MY_MISSION = 'mission_4'

    TARGET_COLOR = MISSION_TARGETS_CONFIG['mission_4']['color']

    CONFIRM_STREAK = 5

    CIRCLE_START_DIST = 3.0
    ORBIT_RADIUS = 3.0
    KP_ORBIT = 0.05
    CIRCLE_BASE_LINEAR = 0.3
    CIRCLE_TURN_RATE = 0.4
    FINISH_DEG = 540.0   # 규정 360도보다 여유있게(1.5바퀴)

    APPROACH_CRUISE_SPEED = 0.3
    APPROACH_K_ANGLE = 0.8

    DETECTION_STALE_TIMEOUT_SEC = 1.0   # 카메라가 이만큼 놓치면 정지

    ROTATE_CW = {'R': True, 'G': True, 'W': False}

    # ---- leg 이동(MOVING/EXIT 공용) ----
    LEG_ARRIVAL_RADIUS_M = 1.0
    LEG_OVERSHOOT_MARGIN_M = 0.5
    LEG_MIN_OVERSHOOT_CHECK_SEC = 7.0
    LEG_CRUISE_LINEAR = 0.4
    LEG_ANGULAR_MAX = 1.0
    LEG_K_ANGULAR = 0.6
    LEG_ALIGN_TOLERANCE_DEG = 15.0
    LEG_ARRIVAL_PAUSE_SEC = 3.0

    def __init__(self):
        super().__init__('mission_4')

        self.active = False
        self.phase = 'MOVING'   # MOVING / SEARCH / APPROACH / CIRCLE / EXIT

        self.current_lat = None
        self.current_lon = None
        self.current_heading = None
        self._fix_this_tick = False

        self.confirm_streak = 0

        self.buoy_angle = None      # rad, camera 기준
        self.buoy_dist = None       # m, camera depth 기준
        self.last_detection_time = None

        self.turn_cw = True
        self.last_heading = None
        self.accumulated_turn_deg = 0.0

        # leg 이동 상태
        self._leg_state = None
        self._leg_target = None
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

        self.get_logger().info('미션4(탐색) 노드 대기 중 - 카메라 단독(라이다 미사용)')

    def active_cb(self, msg):
        self.active = (msg.data == self.MY_MISSION)

    def started_cb(self, msg):
        if msg.data == self.MY_MISSION:
            self.get_logger().info('★ 미션4 시작 (MOVING -> m4s로 이동) ★')
            self.phase = 'MOVING'
            self.confirm_streak = 0
            self.buoy_angle = None
            self.buoy_dist = None
            self.last_detection_time = None
            self.accumulated_turn_deg = 0.0
            self.last_heading = None
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
                    heading = float(part.split('=')[1])
                    self.current_heading = heading
                    if self.phase == 'CIRCLE':
                        self.accumulate_turn(heading)
        except (ValueError, IndexError):
            return
        if self.current_lat is not None and self.current_heading is not None:
            self._fix_this_tick = True

    def accumulate_turn(self, heading):
        if self.last_heading is not None:
            delta = heading - self.last_heading
            if delta > 180:
                delta -= 360
            elif delta < -180:
                delta += 360
            self.accumulated_turn_deg += delta if self.turn_cw else -delta
        self.last_heading = heading

    def detections_cb(self, msg):
        if not self.active:
            return
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if self.phase == 'SEARCH':
            match = next((d for d in detections if d.get('color') == self.TARGET_COLOR), None)
            if match is not None:
                self.confirm_streak += 1
            else:
                self.confirm_streak = 0

            if self.confirm_streak >= self.CONFIRM_STREAK:
                self.get_logger().info(f'목표색({self.TARGET_COLOR}) 확정 → 카메라 접근 시작')
                self.turn_cw = self.ROTATE_CW.get(self.TARGET_COLOR, True)
                self.phase = 'APPROACH'
            return

        if self.phase in ('APPROACH', 'CIRCLE'):
            # 같은 색 중 제일 가까운(거리 있는 것 우선) 것을 부표로 봄
            matches = [d for d in detections if d.get('color') == self.TARGET_COLOR]
            if not matches:
                return
            with_dist = [d for d in matches if d.get('distance') is not None]
            match = min(with_dist, key=lambda d: d['distance']) if with_dist else matches[0]
            self.buoy_angle = match['angle']
            self.buoy_dist = match.get('distance')
            self.last_detection_time = self.get_clock().now()

    def control_loop(self):
        if not self.active:
            return

        if self._pause_active:
            self._tick_pause()
            self._fix_this_tick = False
            return

        if self.phase == 'MOVING':
            self.run_moving()
        elif self.phase == 'APPROACH':
            self.run_approach()
        elif self.phase == 'CIRCLE':
            self.run_circle()
        elif self.phase == 'EXIT':
            self.run_exit()

        self._fix_this_tick = False

    def run_moving(self):
        target = MISSION_TARGETS.get('m4s')
        if target is None:
            self.get_logger().warn('m4s 좌표 없음 (field_config.py 확인)', throttle_duration_sec=5.0)
            return
        if self._leg_state is None:
            self._leg_start(*target)
        result = self._leg_tick()
        if result == 'ARRIVED':
            self.get_logger().info('m4s 도착 - 정지 후 SEARCH 시작 (목표색 확정 대기)')
            self._start_pause(next_phase='SEARCH')

    def _detection_is_fresh(self):
        if self.last_detection_time is None:
            return False
        elapsed = (self.get_clock().now() - self.last_detection_time).nanoseconds / 1e9
        return elapsed <= self.DETECTION_STALE_TIMEOUT_SEC

    def run_approach(self):
        if not self._detection_is_fresh() or self.buoy_angle is None:
            self.cmd_pub.publish(Twist())
            self.get_logger().warn(f'{self.TARGET_COLOR} 부표 안 보임 - 정지', throttle_duration_sec=1.0)
            return

        cmd = Twist()
        cmd.linear.x = self.APPROACH_CRUISE_SPEED
        cmd.angular.z = self.angle_to_angular(self.buoy_angle)
        self.cmd_pub.publish(cmd)

        if self.buoy_dist is not None and self.buoy_dist <= self.CIRCLE_START_DIST:
            self.get_logger().info('목표 부표 근접 → 원선회 시작')
            self.phase = 'CIRCLE'
            self.accumulated_turn_deg = 0.0
            self.last_heading = None

    def run_circle(self):
        if abs(self.accumulated_turn_deg) >= self.FINISH_DEG:
            self.get_logger().info(f'{self.FINISH_DEG:.0f}도 선회 완료 → EXIT 시작 (m4e로 이동)')
            self.cmd_pub.publish(Twist())
            self.phase = 'EXIT'
            self._leg_state = None
            return

        if not self._detection_is_fresh() or self.buoy_dist is None:
            self.cmd_pub.publish(Twist())
            self.get_logger().warn(f'{self.TARGET_COLOR} 부표 안 보임(선회 중) - 정지', throttle_duration_sec=1.0)
            return

        dist_error = self.buoy_dist - self.ORBIT_RADIUS

        cmd = Twist()
        cmd.linear.x = self.CIRCLE_BASE_LINEAR + max(-0.1, min(0.1, self.KP_ORBIT * dist_error))
        cmd.angular.z = self.CIRCLE_TURN_RATE if self.turn_cw else -self.CIRCLE_TURN_RATE
        self.cmd_pub.publish(cmd)

    def run_exit(self):
        target = MISSION_TARGETS.get('m4e')
        if target is None:
            self.get_logger().warn('m4e 좌표 없음 (field_config.py 확인)', throttle_duration_sec=5.0)
            return
        if self._leg_state is None:
            self._leg_start(*target)
        result = self._leg_tick()
        if result == 'ARRIVED':
            self.get_logger().info('m4e 도달! 미션4 완료')
            self.cmd_pub.publish(Twist())
            done = String()
            done.data = self.MY_MISSION
            self.done_pub.publish(done)

    def angle_to_angular(self, target_angle_rad):
        return max(-1.0, min(1.0, self.APPROACH_K_ANGLE * target_angle_rad))

    # ==== 좌표이동(leg) 로직 - mission0/1/3.py와 동일 ====

    def _leg_start(self, target_lat, target_lon):
        self._leg_target = (target_lat, target_lon)
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
                self._leg_committed_linear = self.LEG_CRUISE_LINEAR
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
            if dist <= self.LEG_ARRIVAL_RADIUS_M:
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
        if dist <= self.LEG_ARRIVAL_RADIUS_M:
            self._leg_state = None
            return 'ARRIVED'
        bearing = self.bearing_deg(self.current_lat, self.current_lon, *self._leg_target)
        heading_error = self.normalize_angle_deg(bearing - self.current_heading)
        self._leg_committed_linear = self.LEG_CRUISE_LINEAR
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
    node = Mission4()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
