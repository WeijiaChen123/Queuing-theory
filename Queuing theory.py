import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import seaborn as sns
from collections import deque

# 确保结果目录存在
os.makedirs('results', exist_ok=True)


class QueueSimulator:
    def __init__(self, num_servers, arrival_rate, service_rate, queue_capacity, simulation_time, model_type='SQMS'):
        """
        初始化队列仿真器

        参数:
        num_servers: 服务器数量
        arrival_rate: 顾客到达率
        service_rate: 每个服务器的服务率
        queue_capacity: 队列容量
        simulation_time: 仿真时间
        model_type: 模型类型 ('SQMS' 或 'MQMS')
        """
        self.num_servers = num_servers
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.queue_capacity = queue_capacity
        self.simulation_time = simulation_time
        self.model_type = model_type

        # 验证模型类型
        if model_type not in ['SQMS', 'MQMS']:
            raise ValueError("模型类型必须是 'SQMS' 或 'MQMS'")

    def simulate(self):
        """执行仿真"""
        if self.model_type == 'SQMS':
            return self._simulate_sqms()
        else:
            return self._simulate_mqms()

    def _simulate_sqms(self):
        """单队列多服务器仿真"""
        # 初始化变量
        current_time = 0
        next_arrival = np.random.exponential(1 / self.arrival_rate)

        # 初始化服务台
        servers = [{'busy': False, 'departure_time': float('inf')} for _ in range(self.num_servers)]

        # 初始化队列
        queue = deque()
        queue_length_history = []
        waiting_times = []

        # 统计指标
        total_customers = 0
        served_customers = 0
        rejected_customers = 0
        total_waiting_time = 0
        total_queue_length = 0
        last_event_time = 0

        # 事件循环
        with tqdm(total=self.simulation_time, desc=f'SQMS with {self.num_servers} servers') as pbar:
            while current_time < self.simulation_time:
                # 找出下一个事件时间
                next_departure = min(server['departure_time'] for server in servers)
                next_event = min(next_arrival, next_departure)

                # 更新队列长度统计
                time_elapsed = next_event - current_time
                total_queue_length += len(queue) * time_elapsed
                queue_length_history.append((current_time, len(queue)))
                current_time = next_event
                pbar.update(min(time_elapsed, self.simulation_time - pbar.n))

                # 处理到达事件
                if next_arrival <= next_departure:
                    total_customers += 1

                    # 检查是否有空闲服务台
                    idle_server = next((i for i, s in enumerate(servers) if not s['busy']), None)

                    if idle_server is not None:
                        # 直接服务，无需排队
                        service_time = np.random.exponential(1 / self.service_rate)
                        servers[idle_server]['busy'] = True
                        servers[idle_server]['departure_time'] = current_time + service_time
                        served_customers += 1
                    else:
                        # 检查队列是否有空间
                        if len(queue) < self.queue_capacity:
                            # 加入队列
                            queue.append(current_time)
                        else:
                            # 队列已满，顾客被拒绝
                            rejected_customers += 1

                    # 安排下一个到达
                    next_arrival = current_time + np.random.exponential(1 / self.arrival_rate)

                # 处理离开事件
                else:
                    # 找到完成服务的服务器
                    for i, server in enumerate(servers):
                        if server['departure_time'] == current_time:
                            # 如果队列中有等待的顾客
                            if queue:
                                # 从队列中取出下一个顾客
                                arrival_time = queue.popleft()
                                waiting_time = current_time - arrival_time
                                waiting_times.append(waiting_time)
                                total_waiting_time += waiting_time

                                # 开始服务新顾客
                                service_time = np.random.exponential(1 / self.service_rate)
                                server['departure_time'] = current_time + service_time
                                served_customers += 1
                            else:
                                # 没有顾客等待，服务器空闲
                                server['busy'] = False
                                server['departure_time'] = float('inf')
                            break

        # 计算性能指标
        avg_waiting_time = total_waiting_time / served_customers if served_customers > 0 else 0
        avg_queue_length = total_queue_length / current_time
        rejection_rate = rejected_customers / total_customers if total_customers > 0 else 0

        # 服务台利用率
        utilization = sum(s['busy'] for s in servers) / self.num_servers

        results = {
            'avg_waiting_time': avg_waiting_time,
            'avg_queue_length': avg_queue_length,
            'rejection_rate': rejection_rate,
            'utilization': utilization,
            'waiting_times': waiting_times,
            'queue_length_history': queue_length_history
        }

        return results

    def _simulate_mqms(self):
        """多队列多服务器仿真"""
        # 初始化变量
        current_time = 0
        next_arrival = np.random.exponential(1 / self.arrival_rate)

        # 初始化服务台和队列
        servers = [{'busy': False, 'departure_time': float('inf')} for _ in range(self.num_servers)]
        queues = [deque() for _ in range(self.num_servers)]
        queue_length_history = []
        waiting_times = []

        # 统计指标
        total_customers = 0
        served_customers = 0
        rejected_customers = 0
        total_waiting_time = 0
        total_queue_length = 0
        last_event_time = 0

        # 事件循环
        with tqdm(total=self.simulation_time, desc=f'MQMS with {self.num_servers} servers') as pbar:
            while current_time < self.simulation_time:
                # 找出下一个事件时间
                next_departure = min(server['departure_time'] for server in servers)
                next_event = min(next_arrival, next_departure)

                # 更新队列长度统计
                time_elapsed = next_event - current_time
                total_queue_length += sum(len(q) for q in queues) * time_elapsed
                queue_length_history.append((current_time, sum(len(q) for q in queues)))
                current_time = next_event
                pbar.update(min(time_elapsed, self.simulation_time - pbar.n))

                # 处理到达事件
                if next_arrival <= next_departure:
                    total_customers += 1

                    # 选择最短队列
                    queue_lengths = [len(q) for q in queues]
                    min_queue = min(range(self.num_servers), key=lambda i: queue_lengths[i])

                    # 检查队列是否有空间
                    if len(queues[min_queue]) < self.queue_capacity:
                        # 加入队列
                        queues[min_queue].append(current_time)

                        # 如果服务器空闲，立即服务
                        if not servers[min_queue]['busy']:
                            arrival_time = queues[min_queue].popleft()
                            service_time = np.random.exponential(1 / self.service_rate)
                            servers[min_queue]['busy'] = True
                            servers[min_queue]['departure_time'] = current_time + service_time
                            served_customers += 1
                    else:
                        # 队列已满，顾客被拒绝
                        rejected_customers += 1

                    # 安排下一个到达
                    next_arrival = current_time + np.random.exponential(1 / self.arrival_rate)

                # 处理离开事件
                else:
                    # 找到完成服务的服务器
                    for i, server in enumerate(servers):
                        if server['departure_time'] == current_time:
                            # 如果该服务器的队列中有等待的顾客
                            if queues[i]:
                                # 从队列中取出下一个顾客
                                arrival_time = queues[i].popleft()
                                waiting_time = current_time - arrival_time
                                waiting_times.append(waiting_time)
                                total_waiting_time += waiting_time

                                # 开始服务新顾客
                                service_time = np.random.exponential(1 / self.service_rate)
                                server['departure_time'] = current_time + service_time
                                served_customers += 1
                            else:
                                # 没有顾客等待，服务器空闲
                                server['busy'] = False
                                server['departure_time'] = float('inf')
                            break

        # 计算性能指标
        avg_waiting_time = total_waiting_time / served_customers if served_customers > 0 else 0
        avg_queue_length = total_queue_length / current_time
        rejection_rate = rejected_customers / total_customers if total_customers > 0 else 0

        # 服务台利用率
        utilization = sum(s['busy'] for s in servers) / self.num_servers

        results = {
            'avg_waiting_time': avg_waiting_time,
            'avg_queue_length': avg_queue_length,
            'rejection_rate': rejection_rate,
            'utilization': utilization,
            'waiting_times': waiting_times,
            'queue_length_history': queue_length_history
        }

        return results


def plot_results(results_sqms, results_mqms, num_servers, arrival_rate, service_rate):
    """绘制并保存结果图表"""
    # 创建对比图表
    metrics = ['avg_waiting_time', 'avg_queue_length', 'rejection_rate', 'utilization']
    metric_names = ['avg_waiting_time', 'avg_queue_length', 'rejection_rate', 'utilization']

    sqms_values = [results_sqms[metric] for metric in metrics]
    mqms_values = [results_mqms[metric] for metric in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(12, 8))
    plt.bar(x - width / 2, sqms_values, width, label='Single Queue Multi-Server')
    plt.bar(x + width / 2, mqms_values, width, label='Multi-Queue Multi-Server')

    plt.xlabel('Performance Metrics')
    plt.ylabel('Value')
    plt.title(f'System Performance Comparison ({num_servers}Service Desks)')
    plt.xticks(x, metric_names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加具体数值标签
    for i, v in enumerate(sqms_values):
        plt.text(i - width / 2, v + 0.01, f'{v:.3f}', ha='center')
    for i, v in enumerate(mqms_values):
        plt.text(i + width / 2, v + 0.01, f'{v:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig(f'results/comparison_{num_servers}_servers.png')
    plt.close()

    # 绘制等待时间分布
    plt.figure(figsize=(10, 6))
    sns.histplot(results_sqms['waiting_times'], bins=50, alpha=0.5, label='Single Queue Multi-Server')
    sns.histplot(results_mqms['waiting_times'], bins=50, alpha=0.5, label='Multi-Queue Multi-Server')
    plt.xlabel('Waiting Time')
    plt.ylabel('Frequency')
    plt.title(f'Waiting Time Distribution ({num_servers}Service Desks)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'results/waiting_time_distribution_{num_servers}_servers.png')
    plt.close()

    # 绘制队列长度变化
    times_sqms, lengths_sqms = zip(*results_sqms['queue_length_history'])
    times_mqms, lengths_mqms = zip(*results_mqms['queue_length_history'])

    plt.figure(figsize=(12, 6))
    plt.plot(times_sqms, lengths_sqms, label='Single Queue Multi-Server', alpha=0.7)
    plt.plot(times_mqms, lengths_mqms, label='Multi-Queue Multi-Server', alpha=0.7)
    plt.xlabel('Simulation Time')
    plt.ylabel('Queue Length')
    plt.title(f'Queue Length Changes Over Time ({num_servers}Service Desks)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'results/queue_length_history_{num_servers}_servers.png')
    plt.close()


def run_simulation(num_servers, arrival_rate, service_rate, queue_capacity, simulation_time):
    """运行仿真并生成结果"""
    # 创建仿真器
    sqms_simulator = QueueSimulator(
        num_servers=num_servers,
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        queue_capacity=queue_capacity,
        simulation_time=simulation_time,
        model_type='SQMS'
    )

    mqms_simulator = QueueSimulator(
        num_servers=num_servers,
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        queue_capacity=queue_capacity,
        simulation_time=simulation_time,
        model_type='MQMS'
    )

    # 运行仿真
    results_sqms = sqms_simulator.simulate()
    results_mqms = mqms_simulator.simulate()

    # 打印结果
    print("\n" + "=" * 50)
    print(f"仿真结果 ({num_servers}个服务台)")
    print("=" * 50)
    print(f"{'指标':<20} {'单队列多服务器':<15} {'多队列多服务器':<15}")
    print(f"{'平均等待时间':<20} {results_sqms['avg_waiting_time']:.4f}{'':<10} {results_mqms['avg_waiting_time']:.4f}")
    print(f"{'平均队列长度':<20} {results_sqms['avg_queue_length']:.4f}{'':<10} {results_mqms['avg_queue_length']:.4f}")
    print(f"{'拒绝率':<20} {results_sqms['rejection_rate']:.4f}{'':<10} {results_mqms['rejection_rate']:.4f}")
    print(f"{'服务台利用率':<20} {results_sqms['utilization']:.4f}{'':<10} {results_mqms['utilization']:.4f}")

    # 绘制结果
    plot_results(results_sqms, results_mqms, num_servers, arrival_rate, service_rate)

    return results_sqms, results_mqms


if __name__ == "__main__":
    # 参数设置
    simulation_time = 3000  # 仿真时间
    arrival_rate = 10  # 顾客到达率
    service_rate = 3  # 每个服务台的服务率
    queue_capacity = 20  # 队列容量

    # 对不同数量的服务台进行仿真
    for num_servers in range(2, 10):
        run_simulation(
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            queue_capacity=queue_capacity,
            simulation_time=simulation_time
        )