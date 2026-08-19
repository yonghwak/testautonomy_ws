import json
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data

from kaboat_navigation.field_config import MISSION_TARGETS


def bearing_deg(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def distance_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def destination_point(lat, lon, bearing_deg_val, dist_m):
    """시작좌표 + 방위각(도) + 거리(m)로 목적지 좌표 계산 (구면 공식)."""
    R = 6371000.0
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    theta = math.radians(bearing_deg_val)
    delta = dist_m / R

    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta) +
        math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lam2 = lam1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2)
    )
    return math.degrees(phi2), math.degrees(lam2)


def cluster_scan(ranges, angles, range_min, range_max, max_range, jump_threshold=0.3):
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
            elif abs(valid[i] - valid[i - 1]) > jump_threshold:
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


def fuse_vision_lidar(candidates, clusters, max_bearing_diff_deg=8.0):
    """카메라 candidate(색상 + bearing_deg 힌트)를 라이다 클러스터와 매칭해서
    range/bearing을 라이다 기준값으로 교체. 매칭되는 클러스터가 없으면
    (라이다로 확인 안 된 값은 신뢰하지 않음) 그 candidate는 버림."""
    fused = []
    for cand in candidates:
        best = None
        best_diff = max_bearing_diff_deg
        for cluster in clusters:
            cluster_deg = math.degrees(cluster['center_angle'])
            diff = abs(cluster_deg - cand['bearing_deg'])
            if diff < best_diff:
                best_diff = diff
                best = cluster
        if best is not None:
            fused.append({
                'color': cand['color'],
                'bearing_deg': math.degrees(best['center_angle']),
                'range_m': best['min_range'],
            })
    return fused


class GatePositionMemory:
    """이미 통과한 게이트의 절대좌표(lat, lon)를 기억해서, 회피기동으로
    돌아섰다가 다시 같은 게이트를 근접후보로 잡아 중복 카운트하는 것을 방지."""

    SAME_GATE_RADIUS_M = 3.0

    def __init__(self):
        self.passed_positions = []

    def is_already_passed(self, lat, lon):
        for plat, plon in self.passed_positions:
            if distance_m(plat, plon, lat, lon) <= self.SAME_GATE_RADIUS_M:
                return True
        return False

    def mark_passed(self, lat, lon):
        self.passed_positions.append((lat, lon))


class Mission5(Node):
    """
    미션 5 - 항로추종/게이트.
    대회 규정: 빨강/초록 부표 쌍(게이트)이 코스를 따라 배치되어 있으며, 이를
    순서대로 통과해야 함. 게이트 이탈/충돌 시 패널티.

    흐름:
      MOVING : m5s로 이동 (leg 이동 - 아래 설명). 도착 후 게이트 탐색 대기.
      TASK   : 전방 콘(±80도) 안 red/green 각각 최근접 1개 검출 -> LiDAR로
               range/bearing 정밀보정(fuse_vision_lidar) -> 중간 방위각/거리를
               게이트 중심으로 삼아 조향(건드리지 않음 - 실시간 추적).
               GPS로 이미 통과한 게이트를 기억해 중복 카운트 방지. 거리가
               최소값을 찍고 다시 멀어지면 통과 판정.
      EXIT   : 게이트가 3초간 안 보이고 1개 이상 통과했으면 m5e로 이동(leg 이동)
               후 done. 실제 장애물회피는 avoidance.py + arbiter가 전담,
               여기선 목표방향(goal/heading)만 계속 발행.

    --- 좌표이동(leg) 로직 (mission0/1/3/4.py와 동일 원칙) ---
    MOVING/EXIT처럼 "GPS 좌표 하나로 이동"하는 구간에만 적용. TASK(게이트
    실시간추적)는 이 로직과 무관 - buoys_cb 그대로 둠.
      REACQUIRE: 정지+SOL확보 대기 -> 방위각/조향 커밋(전진+회전 같이).
      CRUISE   : 커밋값 유지, 거리만 체크. 오버슈트(7초 유예 후 재이탈)시 ALIGN.
      ALIGN    : 전진없이 제자리회전만으로 목표방향 재조준(±15도) 후 CRUISE 복귀.
      도착 후 3초 정지 후 다음 단계.
    (avoidance.py는 goal/heading을 매 tick 재계산 안 해도 이미 기준방향으로만
    쓰므로, MOVING/EXIT가 이동 중 방향을 안 바꿔도 회피 성능엔 영향 없음.)

    camera/detections(camera_node.py 발행, color 'R'/'G'/'B' + angle(rad))를
    게이트 판정용 형식(color 'red'/'green' + bearing_deg(도))으로 변환해서 사용.
    """

    MY_MISSION = 'mission_5'

    FORWARD_CONE_DEG = 80.0
    NO_GATE_TIMEOUT = 3.0

    CAMERA_COLOR_MAP = {'R': 'red', 'G': 'green'}

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
        super().__init__('mission_5')
        self.active = False
        self.phase = 'MOVING'
        self.current_heading = None
        self.current_lat = None
        self.current_lon = None
        self._fix_this_tick = False

        self.gate_count = 0
        self.tracking_min_dist = None
        self.last_gate_seen_time = None
        self.exiting = False
        self.latest_clusters = []
        self.gate_memory = GatePositionMemory()
        self.pending_gate_pos = None

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
        self.create_subscription(String, 'kaboat/gps_nav', self.gps_cb, 10)
        self.create_subscription(String, 'camera/detections', self.buoys_cb, 10)
        self.create_subscription(
            LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)

        self.cmd_pub = self.create_publisher(Twist, 'cmd_mission', 10)
        self.heading_pub = self.create_publisher(Float32, 'goal/heading', 10)
        self.done_pub = self.create_publisher(String, 'mission/done', 10)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('mission_5(항로추종/게이트) 노드 시작')

    def active_cb(self, msg):
        self.active = (msg.data == self.MY_MISSION)

    def started_cb(self, msg):
        if msg.data == self.MY_MISSION:
            self.phase = 'MOVING'
            self.gate_count = 0
            self.tracking_min_dist = None
            self.last_gate_seen_time = None
            self.exiting = False
            self.gate_memory = GatePositionMemory()
            self.pending_gate_pos = None
            self._leg_state = None
            self._pause_active = False
            self.get_logger().info('mission_5 시작 - 상태 초기화')

    def scan_cb(self, msg):
        n = len(msg.ranges)
        angles = [msg.angle_min + i * msg.angle_increment for i in range(n)]
        self.latest_clusters = cluster_scan(
            msg.ranges, angles, msg.range_min, msg.range_max, max_range=15.0)

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

    def buoys_cb(self, msg):
        # 게이트 실시간 추적(TASK) - 그대로 둠, leg 이동과 무관
        if not self.active or self.exiting or self.phase != 'TASK':
            return
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        buoys = []
        for d in detections:
            color = self.CAMERA_COLOR_MAP.get(d.get('color'))
            if color is None:
                continue
            buoys.append({
                'color': color,
                'bearing_deg': math.degrees(d.get('angle', 0.0)),
                'range_m': d.get('distance'),
            })

        cone = self.FORWARD_CONE_DEG
        candidates = [b for b in buoys if abs(b['bearing_deg']) <= cone]
        fused = fuse_vision_lidar(candidates, self.latest_clusters, max_bearing_diff_deg=8.0)

        reds = [b for b in fused if b.get('color') == 'red']
        greens = [b for b in fused if b.get('color') == 'green']
        if not reds or not greens:
            return

        nearest_red = min(reds, key=lambda b: b['range_m'])
        nearest_green = min(greens, key=lambda b: b['range_m'])

        mid_bearing_rel = (nearest_red['bearing_deg'] + nearest_green['bearing_deg']) / 2.0
        mid_dist = (nearest_red['range_m'] + nearest_green['range_m']) / 2.0

        if self.current_lat is not None and self.current_heading is not None:
            abs_bearing = (self.current_heading + mid_bearing_rel) % 360.0
            gate_lat, gate_lon = destination_point(
                self.current_lat, self.current_lon, abs_bearing, mid_dist)
            if self.gate_memory.is_already_passed(gate_lat, gate_lon):
                return
            self.pending_gate_pos = (gate_lat, gate_lon)

        if self.last_gate_seen_time is None:
            self.phase = 'TASK'
        self.last_gate_seen_time = self.get_clock().now()

        if self.tracking_min_dist is None or mid_dist < self.tracking_min_dist:
            self.tracking_min_dist = mid_dist
        elif mid_dist > self.tracking_min_dist + 1.0:
            self.gate_count += 1
            self.get_logger().info(f'게이트 {self.gate_count}번 통과 추정')
            if self.pending_gate_pos is not None:
                self.gate_memory.mark_passed(*self.pending_gate_pos)
            self.tracking_min_dist = None

        if self.current_heading is not None:
            h_msg = Float32()
            h_msg.data = (self.current_heading + mid_bearing_rel) % 360.0
            self.heading_pub.publish(h_msg)

        cmd = Twist()
        cmd.linear.x = 0.3
        cmd.angular.z = max(-1.0, min(1.0, 0.02 * mid_bearing_rel))
        self.cmd_pub.publish(cmd)

    def control_loop(self):
        if not self.active:
            return

        if self._pause_active:
            self._tick_pause()
            self._fix_this_tick = False
            return

        if self.exiting:
            self.run_exit()
            self._fix_this_tick = False
            return

        if self.last_gate_seen_time is None:
            self.run_moving()
            self._fix_this_tick = False
            return

        elapsed = (self.get_clock().now() - self.last_gate_seen_time).nanoseconds / 1e9
        if elapsed > self.NO_GATE_TIMEOUT and self.gate_count >= 1:
            self.get_logger().info(f'게이트 안 보임({elapsed:.1f}s) - 정지 후 종료지점으로 이동')
            self.exiting = True
            self.last_gate_seen_time = None
            self._leg_state = None
            self._start_pause(next_phase='EXIT_ENTER')  # 실제 phase 전환은 pause 종료 후

        self._fix_this_tick = False

    def run_moving(self):
        target = MISSION_TARGETS.get('m5s')
        if target is None:
            self.get_logger().warn('m5s 좌표 없음 (field_config.py 확인)', throttle_duration_sec=5.0)
            return
        if self._leg_state is None:
            self._leg_start(*target)
        result = self._leg_tick()
        if result == 'ARRIVED':
            self.get_logger().info('m5s 도착 - 게이트 탐색 대기 (TASK는 buoys_cb가 자동 시작)')
            # phase는 MOVING 유지 - buoys_cb가 게이트 잡으면 스스로 TASK로 전환함
            self._leg_state = 'DONE_WAIT_GATE'  # leg 재실행 안 되게 막기용

    def run_exit(self):
        target = MISSION_TARGETS.get('m5e')
        if target is None:
            self.get_logger().warn('m5e 좌표 없음 (field_config.py 확인)', throttle_duration_sec=5.0)
            return
        if self._leg_state is None or self._leg_state == 'DONE_WAIT_GATE':
            self._leg_start(*target)
        result = self._leg_tick()
        if result == 'ARRIVED':
            self.get_logger().info('mission_5 완료')
            self.cmd_pub.publish(Twist())
            done = String()
            done.data = self.MY_MISSION
            self.done_pub.publish(done)

    # ==== 좌표이동(leg) 로직 - mission0/1/3/4.py와 동일 ====

    def _leg_start(self, target_lat, target_lon):
        self._leg_target = (target_lat, target_lon)
        self._leg_state = 'REACQUIRE'
        self._leg_committed_linear = 0.0
        self._leg_committed_angular = 0.0
        self._leg_min_dist = None
        self._leg_cruise_start = None

    def _leg_tick(self):
        if self._leg_state in (None, 'DONE_WAIT_GATE'):
            return 'WAITING'

        if self._leg_state == 'REACQUIRE':
            self.cmd_pub.publish(Twist())
            if not self._fix_this_tick or self.current_lat is None:
                return 'WAITING'
            return self._leg_commit_and_cruise()

        if self._leg_state == 'ALIGN':
            if not self._fix_this_tick or self.current_lat is None:
                self.cmd_pub.publish(Twist())
                return 'WAITING'
            bearing = bearing_deg(self.current_lat, self.current_lon, *self._leg_target)
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
            dist = distance_m(self.current_lat, self.current_lon, *self._leg_target)
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
        dist = distance_m(self.current_lat, self.current_lon, *self._leg_target)
        if dist <= self.LEG_ARRIVAL_RADIUS_M:
            self._leg_state = None
            return 'ARRIVED'
        bearing = bearing_deg(self.current_lat, self.current_lon, *self._leg_target)
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
            self._pause_active = False
            self._leg_state = None
            # EXIT_ENTER는 실제 phase 이름이 아니라 그냥 pause 후 run_exit로
            # 자연스럽게 흘러가게 하는 표식 - exiting=True는 이미 설정돼있음.

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


def main(args=None):
    rclpy.init(args=args)
    node = Mission5()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
