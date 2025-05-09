import matplotlib.pyplot as plt
from matplotlib import rc

# Configure matplotlib to use LaTeX
rc('text', usetex=True)
rc('font', family='serif')
rc('text.latex', preamble=r'\usepackage{ctex} \usepackage{multirow} \usepackage{makecell}')

# The LaTeX table code (slightly modified for matplotlib)
latex_table = r"""
\begin{tabular}{|p{2cm}|p{5cm}|p{2.5cm}|p{2cm}|p{1.5cm}|p{1.5cm}|}
\hline
\textbf{命令类型} & \multicolumn{1}{c|}{\textbf{示例命令}} & \textbf{噪音水平} & \textbf{距语音传感器距离} & \textbf{WER(\%)} & \textbf{平均延时} \\
\hline
\multirow{7}{*}{短语命令} & 抓住物体， & \multirow{7}{*}{安静（30-40dB）} & \multirow{7}{*}{50cm} & \multirow{7}{*}{6.02} & \multirow{7}{*}{894ms} \\
 & 放下物体， &  &  &  &  \\
 & 抬起手臂， &  &  &  &  \\
 & 旋转90度， &  &  &  &  \\
 & 前进， &  &  &  &  \\
 & 后退， &  &  &  &  \\
 & 停止等 &  &  &  &  \\
\hline
\multirow{7}{*}{短语命令} & 抓住物体， & \multirow{7}{*}{中等（50-60dB）} & \multirow{7}{*}{50cm} & \multirow{7}{*}{8.73} & \multirow{7}{*}{903ms} \\
 & 放下物体， &  &  &  &  \\
 & 抬起手臂， &  &  &  &  \\
 & 旋转90度， &  &  &  &  \\
 & 前进， &  &  &  &  \\
 & 后退， &  &  &  &  \\
 & 停止等 &  &  &  &  \\
\hline
\multirow{3}{*}{串行命令} & 抓住物体，然后放到托盘上； & \multirow{3}{*}{安静（30-40dB）} & \multirow{3}{*}{50cm} & \multirow{3}{*}{5.43} & \multirow{3}{*}{843ms} \\
 & 抓取物体后，转向左边； &  &  &  &  \\
 & 先打开夹爪，再抓取物体等 &  &  &  &  \\
\hline
\multirow{3}{*}{串行命令} & 抓住物体，然后放到托盘上； & \multirow{3}{*}{中等（50-60dB）} & \multirow{3}{*}{50cm} & \multirow{3}{*}{7.89} & \multirow{3}{*}{923ms} \\
 & 抓取物体后，转向左边； &  &  &  &  \\
 & 先打开夹爪，再抓取物体等 &  &  &  &  \\
\hline
\end{tabular}
"""

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax.axis('off')

# Add the table
ax.text(0.5, 0.5, latex_table, ha='center', va='center', size=10)

# Add a title
plt.title('语音命令性能数据表', fontsize=16)

# Save and show
plt.tight_layout()
plt.savefig('latex_table_preview.png', dpi=200, bbox_inches='tight')
plt.show()

print("Preview image saved as 'latex_table_preview.png'")