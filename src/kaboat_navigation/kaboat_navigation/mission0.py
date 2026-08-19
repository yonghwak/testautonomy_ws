import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist

from kaboat_navigation.field_config import MISSION_TARGETS


class Mission0(Node):
    """
    미션 0 - 출발지 -> 미션1 시작 전 장소이동 전용.
    순수 GPS 좌표이동만 함(회피/카메라/라이다 없음) - m0s로 간 다음 m0e로 감,
    도착하면 done 발행.

    좌표이동(leg) 로직은 mission1.py와 동일:
      REACQUIRE: 정지 + SOL 확보 대기. 확보되면 방위각/조향값 한번에 커밋(전진+회전).
      CRUISE   : 커밋된 조향 유지, 좌표 거리만 체크(헤딩 재계산 안 함). 도착하면 ARRIVED.
                 최소거리 대비 다시 멀어지면(7초 유예 후) 오버슈트로 보고 ALIGN.
      ALIGN    : 오버슈트 복구 전용. 전진 없이 제자리 회전만으로 목표방향 재조준
                 (±15도 이내) - 방향 크게 틀어진 채로 계속 밀면 원 그리며 더
                 멀어지는 문제 방지. 맞춰지면 CRUISE 재개.
      도착 후 3초 완전정지 후 다음 leg(또는 완료).
    """

    MY_MISSION = 'mission_0'

    LEG_ARRIVAL_RADIUS_M = 1.0
    LEG_OVERSHOOT_MARGIN_M = 0.5
    LEG_MIN_OVERSHOOT_CHECK_SEC = 7.0
    LEG_CRUISE_LINEAR = 0.4
    LEG_ANGULAR_MAX = 1.0
    LEG_K_ANGULAR = 0.6
    LEG_ALIGN_TOLERANCE_DEG = 15.0
    LEG_ARRIVAL_PAUSE_SEC = 3.0

    def __init__(self):
        super().__init__('mission_0')
        self.active = False
        self.phase = 'M0S'   # M0S -> M0E -> DONE
        self.done_logged = False

        self.current_lat = None
        self.current_lon = None
        self.current_heading = None

        self._leg_state = None
        self._leg_target = None
        self._leg_committed_linear = 0.0
        self._leg_committed_angular = 0.0
        self._leg_min_dist = None
        self._leg_cruise_start = None
        self._fix_this_tick = False

        self._pause_active = False
        self._pause_start = None
        self._pause_next_phase = None

        self.create_subscription(String, 'mission/active', self.active_cb, 10)
        self.create_subscription(String, 'mission/started', self.started_cb, 10)
        self.create_subscription(String, 'kaboat/gps_nav', self.gps_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, 'cmd_mission', 10)
        self.heading_pub = self.create_publisher(Float32, 'goal/heading', 10)
        self.done_pub = self.create_publisher(String, 'mission/done', 10)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('mission_0(장소이동) 노드 시작')

    def active_cb(self, msg):
        self.active = (msg.data == self.MY_MISSION)

    def started_cb(self, msg):
        if msg.data == self.MY_MISSION:
            self.phase = 'M0S'
            self.done_logged = False
            self._leg_state = None
            self._pause_active = False
            self.get_logger().info('mission_0 시작 - 상태 초기화 (m0s로 이동)')

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

    def control_loop(self):
        if not self.active:
            return

        if self._pause_active:
            self._tick_pause()
            self._fix_this_tick = False
            return

        if self.phase == 'M0S':
            self._run_leg('m0s', next_phase='M0E')
        elif self.phase == 'M0E':
            self._run_leg('m0e', next_phase='DONE')

        self._fix_this_tick = False

    def _run_leg(self, target_key, next_phase):
        target = MISSION_TARGETS.get(target_key)
        if target is None:
            self.get_logger().warn(f'{target_key} 좌표 없음 (field_config.py 확인)', throttle_duration_sec=5.0)
            return
        if self._leg_state is None:
            self._leg_start(*target)
        result = self._leg_tick()
        if result == 'ARRIVED':
            self.get_logger().info(f'{target_key} 도착')
            if next_phase == 'DONE':
                self.finish()
            else:
                self._start_pause(next_phase=next_phase)

    def finish(self):
        self.get_logger().info('m0e 도착 - mission_0 완료')
        self.cmd_pub.publish(Twist())
        done = String()
        done.data = self.MY_MISSION
        self.done_pub.publish(done)
        self.done_logged = True
        self.phase = 'DONE'

    # ==== 좌표이동(leg) 로직 - mission1.py와 동일 ====

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
            self.publish_goal_heading(bearing)
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
        self.publish_goal_heading(bearing)
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

    def publish_goal_heading(self, deg):
        h = Float32()
        h.data = deg
        self.heading_pub.publish(h)

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
    node = Mission0()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
