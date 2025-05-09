import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14  # 设置更大的基本字体大小

# 创建画布
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')  # 隐藏坐标轴

# 定义节点和箭头的样式
node_style = dict(boxstyle='round,pad=0.8', facecolor='#3498db', edgecolor='black', alpha=0.8)
arrow_style = dict(arrowstyle='->', color='black', linewidth=2, connectionstyle='arc3,rad=0.0')
process_style = dict(boxstyle='round,pad=0.8', facecolor='#2ecc71', edgecolor='black', alpha=0.8)
output_style = dict(boxstyle='round,pad=0.8', facecolor='#e74c3c', edgecolor='black', alpha=0.8)

# 定义流程图的每个节点
nodes = [
    # x, y, 宽, 高, 文本, 样式
    (5, 11, 6, 1, "开始", node_style),
    (5, 9, 8, 1, "1. 树莓派通过Websocket接收来自上位机的操作指令", process_style),
    (5, 7, 8, 1, "2. 分辨操作对象(小车/机械臂)和具体操作参数(方向/速度/角度)", process_style),
    (5, 5, 8, 1, "3. 计算PWM信号与占空比和方向控制数字信号", process_style),
    (5, 3, 8, 1, "4. 生成并输出对应PWM与数字电平控制信号", process_style),
    (5, 1, 6, 1, "硬件执行动作", output_style),
]

# 绘制节点
for x, y, w, h, text, style in nodes:
    rect = Rectangle((x - w/2, y - h/2), w, h, 
                    facecolor=style['facecolor'], 
                    edgecolor=style['edgecolor'],
                    alpha=style['alpha'],
                    zorder=1)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', color='white', 
            fontweight='bold', fontsize=16, zorder=2)

# 绘制连接箭头
for i in range(len(nodes) - 1):
    x1, y1 = nodes[i][0], nodes[i][1] - nodes[i][3]/2
    x2, y2 = nodes[i+1][0], nodes[i+1][1] + nodes[i+1][3]/2
    arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                           arrowstyle=arrow_style['arrowstyle'],
                           color=arrow_style['color'], 
                           linewidth=arrow_style['linewidth'],
                           connectionstyle=arrow_style['connectionstyle'],
                           zorder=1)
    ax.add_patch(arrow)

# 添加标题
ax.text(5, 12.3, '机器人控制系统工作流程', ha='center', fontsize=22, fontweight='bold')

# 添加更详细的说明
details = """
系统工作原理:
1. 树莓派作为控制核心，通过Websocket从上位机接收指令
2. 系统解析指令，确定操作对象和参数
3. 根据数学模型计算所需信号量
4. 输出信号到电机驱动和舵机模块
5. 实现机器人预期动作
"""

# 创建一个文本框来显示详细说明
plt.figtext(0.5, 0.02, details, ha='center', va='center', 
           bbox=dict(boxstyle='round', facecolor='#f1f1f1', alpha=0.9),
           fontsize=14)

plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.show()