import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MissionManager(Node):
    """
    상태 머신 - 미션 순서 관리자.
    하는 일:
      1. 지금 어떤 미션이 활성인지 계속 알림 (mission/active, 0.5초마다)
      2. 미션이 막 시작된 순간에 1회 알림 (mission/started)
      3. 미션 전환 시 전체 진행상황을 로그로 시각화 (완료/현재/대기 표시)
      4. STATUS_LOG_INTERVAL_SEC 마다 현재 미션 요약을 반복 출력

    실행 순서 = mission_0(장소이동) -> mission_1(장애물회피) -> mission_2(위치유지)
                -> mission_3(도킹) -> mission_4(탐색) -> mission_5(항로추종/게이트)
                -> finished
    """

    MISSIONS = [
        'mission_0',
        'mission_1',
        'mission_2',
        'mission_3',
        'mission_4',
        'mission_5',
        'finished',
    ]

    STATUS_LOG_INTERVAL_SEC = 5.0

    def __init__(self):
        super().__init__('mission_manager')

        self.current_index = 0

        self.active_pub = self.create_publisher(String, 'mission/active', 10)
        self.started_pub = self.create_publisher(String, 'mission/started', 10)
        self.create_subscription(String, 'mission/done', self.done_callback, 10)

        self.timer = self.create_timer(0.5, self.publish_active)
        self.status_timer = self.create_timer(self.STATUS_LOG_INTERVAL_SEC, self.log_status_summary)

        self.get_logger().info(f'미션 매니저 시작 - 현재: {self.current_mission()}')
        self.log_progress_board()
        self.publish_started()

    def current_mission(self):
        return self.MISSIONS[self.current_index]

    def publish_active(self):
        if self.current_mission() == 'finished':
            return
        msg = String()
        msg.data = self.current_mission()
        self.active_pub.publish(msg)

    def publish_started(self):
        if self.current_mission() == 'finished':
            return
        msg = String()
        msg.data = self.current_mission()
        self.started_pub.publish(msg)

    def done_callback(self, msg):
        if msg.data != self.current_mission():
            return
        if self.current_mission() == 'finished':
            return
        self.get_logger().info(f'{msg.data} 완료! 다음 미션으로 전환')
        self.current_index += 1
        self.get_logger().info(f'현재 미션: {self.current_mission()}')
        self.log_progress_board()
        self.publish_started()

    def log_progress_board(self):
        lines = []
        for i, name in enumerate(self.MISSIONS):
            if name == 'finished':
                mark = '🏁' if i == self.current_index else '  '
            elif i < self.current_index:
                mark = '✅'
            elif i == self.current_index:
                mark = '▶ '
            else:
                mark = '⬜'
            lines.append(f'{mark} {name}')
        board = '\n'.join(lines)
        self.get_logger().info(f'\n===== 미션 진행상황 =====\n{board}\n=========================')

    def log_status_summary(self):
        if self.current_mission() == 'finished':
            return
        self.get_logger().info(f'[진행중] {self.current_mission()} '
                                f'({self.current_index + 1}/{len(self.MISSIONS)})')


def main(args=None):
    rclpy.init(args=args)
    node = MissionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()        self.create_subscription(String, 'mission/done', self.done_callback, 10)

        self.timer = self.create_timer(0.5, self.publish_active)
        self.status_timer = self.create_timer(self.STATUS_LOG_INTERVAL_SEC, self.log_status_summary)

        self.get_logger().info(f'미션 매니저 시작 - 현재: {self.current_mission()}')
        self.log_progress_board()
        self.publish_started()

    def current_mission(self):
        return self.MISSIONS[self.current_index]

    def publish_active(self):
        if self.current_mission() == 'finished':
            return
        msg = String()
        msg.data = self.current_mission()
        self.active_pub.publish(msg)

    def publish_started(self):
        if self.current_mission() == 'finished':
            return
        msg = String()
        msg.data = self.current_mission()
        self.started_pub.publish(msg)

    def done_callback(self, msg):
        if msg.data != self.current_mission():
            return
        if self.current_mission() == 'finished':
            return
        self.get_logger().info(f'{msg.data} 완료! 다음 미션으로 전환')
        self.current_index += 1
        self.get_logger().info(f'현재 미션: {self.current_mission()}')
        self.log_progress_board()
        self.publish_started()

    def log_progress_board(self):
        lines = []
        for i, name in enumerate(self.MISSIONS):
            if name == 'finished':
                mark = '🏁' if i == self.current_index else '  '
            elif i < self.current_index:
                mark = '✅'
            elif i == self.current_index:
                mark = '▶ '
            else:
                mark = '⬜'
            lines.append(f'{mark} {name}')
        board = '\n'.join(lines)
        self.get_logger().info(f'\n===== 미션 진행상황 =====\n{board}\n=========================')

    def log_status_summary(self):
        if self.current_mission() == 'finished':
            return
        self.get_logger().info(f'[진행중] {self.current_mission()} '
                                f'({self.current_index + 1}/{len(self.MISSIONS)})')


def main(args=None):
    rclpy.init(args=args)
    node = MissionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
