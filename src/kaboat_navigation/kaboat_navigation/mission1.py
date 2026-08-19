import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist

from kaboat_navigation.field_config import MISSION_TARGETS


class Mission1(Node):
    """
    미션 1 - 장애물회피 통과.
    대회 규정: 부표 사이(장애물 구간)를 통과하며 목적지까지 이동. 충돌 시 패널티.

    흐름:
      MOVING : m1s로 이동 (좌표기반 leg 이동, 아래 설명)
      TASK   : m1e로 이동. 도착시 done 발행.

    실제 라이다 회피는 이 노드가 신경쓰지 않음 - avoidance.py + arbiter가 전담.
    이 노드는 목표방향(goal/heading)만 계속 알려주는 역할.

    --- 좌표기반 이동(leg) 로직 ---
    실측 결과 배가 움직이는 동안 SOL_COMPUTED가 자주 풀리고, 정지 상태에서만
    안정적으로 잡힘. 그래서:
      REACQUIRE: 정지 + SOL 확보 대기. 확보되면 그 순간 방위각/조향값을 한번에
                 커밋(전진+회전 같이 나감 - 일반적인 완만한 진입각이면 이걸로 충분).
      CRUISE   : 커밋된 조향 그대로 유지, 좌표 거리만 계속 체크(헤딩 재계산 안 함).
                 도착하면 ARRIVED. 최소거리 대비 다시 멀어지면(7초 유예 후)
                 오버슈트로 보고 ALIGN으로.
      ALIGN    : 오버슈트 복구 전용. 전진 없이 제자리 회전만으로 목표방향을
                 다시 정확히 맞춤(±15도 이내) - 방향이 크게 틀어진 채로 계속
                 밀고 나가면 원을 그리며 더 멀어지는 문제 방지. 맞춰지면 CRUISE로.
      도착 후에는(다음 단계로 넘어가기 전) 3초 완전정지.
    """

    MY_MISSION = 'mission_1'

    # ---- 좌표이동(leg) 공용 파라미터 ----
    LEG_ARRIVAL_RADIUS_M = 1.0
    LEG_OVERSHOOT_MARGIN_M = 0.5
    LEG_MIN_OVERSHOOT_CHECK_SEC = 7.0
    LEG_CRUISE_LINEAR = 0.4
    LEG_ANGULAR_MAX = 1.0
    LEG_K_ANGULAR = 0.6
    LEG_ALIGN_TOLERANCE_DEG = 15.0
    LEG_ARRIVAL_PAUSE_SEC = 3.0

    def __init__(self):
        super().__init__('mission_1')
        self.active = False
        self.phase = 'MOVING'
        self.done_logged = False

        self.current_lat = None
        self.current_lon = None
        self.current_heading = None

        # leg 이동 상태
        self._leg_state = None       # None / 'REACQUIRE' / 'CRUISE' / 'ALIGN'
        self._leg_target = None
        self._leg_arrival_radius = self.LEG_ARRIVAL_RADIUS_M
        self._leg_committed_linear = 0.0
        self._leg_committed_angular = 0.0
        self._leg_min_dist = None
        self._leg_cruise_start = None
        self._fix_this_tick = False  # 이번 tick 사이에 새 SOL fix 들어왔는지

        # 정지 대기(도착 후 3초)
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
        self.get_logger().info('mission_1(장애물회피) 노드 시작')

    def active_cb(self, msg):
        self.active = (msg.data == self.MY_MISSION)

    def started_cb(self, msg):
        if msg.data == self.MY_MISSION:
            self.phase = 'MOVING'
            self.done_logged = False
            self._leg_state = None
            self._pause_active = False
            self.get_logger().info('mission_1 시작 - 상태 초기화')

    def gps_cb(self, msg):
        # kaboat/gps_nav는 gps_node1이 SOL_COMPUTED일 때만 발행 - 여기 도달했다는
        # 것 자체가 이번 fix가 유효했다는 뜻.
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

        if self.phase == 'MOVING':
            self.run_moving()
        elif self.phase == 'TASK':
            self.run_task()

        self._fix_this_tick = False

    def run_moving(self):
        target = MISSION_TARGETS.get('m1s')
        if target is None:
            self.get_logger().warn('m1s 좌표 없음 (field_config.py 확인)', throttle_duration_sec=5.0)
            return
        if self._leg_state is None:
            self._leg_start(*target)
        result = self._leg_tick()
        if result == 'ARRIVED':
            self.get_logger().info('m1s 도착 - 정지 후 m1e로 전환')
            self._start_pause(next_phase='TASK')

    def run_task(self):
        if self.done_logged:
            return
        target = MISSION_TARGETS.get('m1e')
        if target is None:
            self.get_logger().warn('m1e 좌표 없음 (field_config.py 확인)', throttle_duration_sec=5.0)
            return
        if self._leg_state is None:
            self._leg_start(*target)
        result = self._leg_tick()
        if result == 'ARRIVED':
            self.finish()

    def finish(self):
        self.get_logger().info('m1e 도착 - mission_1 완료')
        self.cmd_pub.publish(Twist())
        done = String()
        done.data = self.MY_MISSION
        self.done_pub.publish(done)
        self.done_logged = True

    # ==== 좌표이동(leg) 공용 로직 ====

    def _leg_start(self, target_lat, target_lon, arrival_radius=None):
        self._leg_target = (target_lat, target_lon)
        self._leg_arrival_radius = arrival_radius if arrival_radius is not None else self.LEG_ARRIVAL_RADIUS_M
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
    node = Mission1()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
