# matlab-mcp

MATLAB 风格的数学计算 / 自动控制原理画图 MCP 服务器。专为学习《自动控制原理》《信号与系统》等课程时
"边做题边出图" 设计 —— 画完的图保存为 PNG 并返回绝对路径和关键数值结果，对话里直接配图讲解。

## 工具列表（9 个）

| 工具 | 用途 | 自动附带的结果 |
|---|---|---|
| `calc` | numpy 语法的数值计算器，变量跨调用保存（类似 MATLAB 工作区） | 计算结果 |
| `plot_function` | 任意函数曲线 y=f(x)，多条曲线 / 极坐标 / 对数轴 | 图片路径 |
| `plot_bode` | 伯德图（幅频+相频） | 幅值裕度 GM、相角裕度 PM、ω_pc、ω_gc |
| `plot_nyquist` | 奈奎斯特图，标注 (-1, j0) | 环绕圈数 W、P、Z = P−W，闭环稳定性结论 |
| `plot_root_locus` | 根轨迹 | 开环零极点、渐近线交点与夹角 |
| `plot_step` | 单位阶跃响应 | 超调量 σ%、峰值时间 tp、上升时间 tr、调节时间 ts、稳态值 |
| `plot_impulse` | 单位脉冲响应 | 闭环极点 |
| `plot_pole_zero` | 零极点分布图 | 每个极点的 ωn、ζ；稳定性判断 |
| `run_plot_code` | 逃生舱：直接跑 numpy/matplotlib 代码（3D、lsim 任意输入仿真、多子图等） | 所有 figure 的路径 |

## 传递函数写法

系数按 s 降幂排列，与 MATLAB `tf(num, den)` 一致：

```
G(s) = 10/(s² + 2s + 10)   →  num=[10],  den=[1, 2, 10]
G(s) = 1/(s(s+1)(s+2))     →  num=[1],   den=[1, 3, 2, 0]
G(s) = (s+2)/(s²+0.6s+4)   →  num=[1,2], den=[1, 0.6, 4]
```

## 使用示例

对话里直接说人话即可，例如：

- "画 G(s)=1/(s(s+1)(s+2)) 的伯德图，求稳定裕度"
- "画 K=10 时上面系统的奈氏图，判断闭环稳定性"
- "画 1/(s²+0.4s+1) 的单位阶跃响应，算超调量和调节时间"
- "画 K/((s+1)(s+2)) 的根轨迹"
- "画一下 sin(x)/x 的图像，x 从 -20 到 20"
- "算一下矩阵 [[0,1],[-2,-3]] 的特征值"（calc）

图片保存在本目录 `plots/` 下。

## 安装 / 依赖

```
pip install numpy scipy matplotlib "mcp<2"
```

已在 ZCode 用户配置 `~/.zcode/cli/config.json` → `mcp.servers.matlab` 注册：

```json
"matlab": {
  "type": "stdio",
  "command": "C:\\Users\\86187\\AppData\\Local\\Programs\\Python\\Python313\\python.exe",
  "args": ["E:\\opencode workplace\\matlab-mcp\\server.py"],
  "env": { "PYTHONIOENCODING": "utf-8" }
}
```

修改 server.py 后重启 ZCode（或重连 MCP）生效。不想用了：删掉配置里 `matlab` 这一项即可。

## 验证过的正确性（教科书结果）

- G=1/(s(s+1)(s+2))：GM=15.56dB(6 倍) @ ω_pc=1.414，PM=53.4° @ ω_gc=0.446 ✓
- 同系统 K=10 奈氏判据：W=−2，闭环不稳定 ✓
- 二阶 ζ=0.2：超调量 52.68%（理论 52.66%）✓
- 根轨迹渐近线：σ=−1.5±90°、σ=−1±60°/180° ✓

## 测试

```
python test_local.py   # 直接调内部函数, 全工具冒烟+数值核对
python test_mcp.py     # 按真实 MCP stdio 协议端到端调用
```
