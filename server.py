# -*- coding: utf-8 -*-
"""
matlab-mcp: MATLAB 风格的数学计算 / 自动控制原理画图 MCP 服务器 (stdio)

工具列表:
  calc            数值计算器(带工作区变量, numpy 语法): 矩阵、特征值、微分方程系数等
  plot_function   画任意函数曲线 y=f(x), 支持多条曲线 / 极坐标 / 对数轴
  plot_bode       伯德图(含幅值裕度 GM、相角裕度 PM 自动计算)
  plot_nyquist    奈奎斯特图(自动标出 -1 点, 统计环绕圈数并判断闭环稳定性)
  plot_root_locus 根轨迹(含渐近线)
  plot_step       单位阶跃响应(自动算超调量、峰值时间、调节时间)
  plot_impulse    单位脉冲响应
  plot_pole_zero  零极点分布图(含每个极点的 ζ、ωn)
  run_plot_code   逃生舱: 直接运行 numpy/matplotlib 代码画任意图

传递函数统一用系数列表表示(按 s 降幂), 与 MATLAB tf(num, den) 一致:
  G(s) = 10 / (s^2 + 2s + 10)  ->  num=[10], den=[1, 2, 10]
  G(s) = 1 / (s(s+1)(s+2))     ->  num=[1],  den=[1, 3, 2, 0]

所有图片保存到 本文件同目录/plots/ 下, 工具返回绝对路径与关键数值结果。
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import math
import re
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from mcp.server.fastmcp import FastMCP

BASE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

mcp = FastMCP("matlab")

_seq = [0]


def _save(fig: plt.Figure, tag: str) -> str:
    _seq[0] += 1
    name = f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}_{_seq[0]:03d}.png"
    path = os.path.join(PLOTS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 传递函数辅助

def _tf(num, den):
    num = [float(c) for c in num]
    den = [float(c) for c in den]
    if not num or not den:
        raise ValueError("num/den 不能为空")
    while len(num) > 1 and abs(num[0]) < 1e-15:
        num.pop(0)
    while len(den) > 1 and abs(den[0]) < 1e-15:
        den.pop(0)
    if abs(den[0]) < 1e-15:
        raise ValueError("den 全为零")
    if len(den) < len(num):
        raise ValueError(f"不允许分子阶次高于分母(非真有理): 分子 {len(num)-1} 阶 > 分母 {len(den)-1} 阶")
    return num, den


def _poly_str(c) -> str:
    c = [float(x) for x in c]
    while c and abs(c[0]) < 1e-15:
        c.pop(0)
    if not c:
        return "0"
    n = len(c) - 1
    parts = []
    for i, co in enumerate(c):
        pw = n - i
        if abs(co) < 1e-15:
            continue
        var = "s" if pw == 1 else (f"s^{pw}" if pw > 1 else "")
        if pw == 0:
            term = f"{co:g}"
        elif abs(co - 1) < 1e-15:
            term = var
        elif abs(co + 1) < 1e-15:
            term = "-" + var
        else:
            term = f"{co:g}·{var}"
        parts.append((co, term))
    out = ""
    for j, (co, term) in enumerate(parts):
        neg = co < 0
        t = term[1:] if term.startswith("-") else term
        if j == 0:
            out += ("-" if neg else "") + t
        else:
            out += (" - " if neg else " + ") + t
    return out


def _tf_str(num, den) -> str:
    ns, ds = _poly_str(num), _poly_str(den)
    if ds == "1":
        return f"G(s) = {ns}"
    return f"G(s) = ({ns}) / ({ds})"


def _auto_wrange(num, den):
    """根据零极点模值猜测合适的频率范围"""
    mags = []
    for c in (num, den):
        r = np.roots(np.asarray(c, float))
        r = np.abs(r[np.abs(r) > 1e-12])
        if r.size:
            mags.append(r)
    pts = np.concatenate(mags) if mags else np.array([1.0])
    lo = max(pts.min() / 80.0, 1e-4)
    hi = min(pts.max() * 80.0, 1e7)
    if lo >= hi:
        lo, hi = 1e-2, 1e2
    return float(lo), float(hi)


def _freq_resp(num, den, w):
    w = np.asarray(w, float)
    _, H = signal.freqs(num, den, worN=w)
    return H


def _margins(w, mag_db, ph_deg):
    """计算幅值裕度 GM / 相角裕度 PM 及对应频率(取最低频率处的穿越点)"""
    res = {}
    # 幅值穿越 (0 dB) -> 相角裕度
    s = np.sign(mag_db)
    for i in np.where(np.diff(s) != 0)[0]:
        t = -mag_db[i] / (mag_db[i + 1] - mag_db[i])
        wc = 10 ** (math.log10(w[i]) + t * (math.log10(w[i + 1]) - math.log10(w[i])))
        ph = ph_deg[i] + t * (ph_deg[i + 1] - ph_deg[i])
        pm = (180.0 + ph) % 360.0
        if pm > 180:
            pm -= 360
        res["w_gc"], res["PM_deg"], res["ph_at_gc"] = wc, pm, ph
        break
    # 相角穿越 (-180°+k·360°) -> 幅值裕度
    tgt = ph_deg + 180.0
    k = tgt / 360.0
    for i in range(len(w) - 1):
        if k[i] != k[i + 1] and math.floor(k[i]) != math.floor(k[i + 1]):
            lvl = 360.0 * (min(math.floor(k[i]), math.floor(k[i + 1])) + 1)
            t = (lvl - tgt[i]) / (tgt[i + 1] - tgt[i])
            wpc = 10 ** (math.log10(w[i]) + t * (math.log10(w[i + 1]) - math.log10(w[i])))
            gm = -(mag_db[i] + t * (mag_db[i + 1] - mag_db[i]))
            res["w_pc"], res["GM_dB"], res["gm_ratio"] = wpc, gm, 10 ** (gm / 20)
            break
    return res


def _num_integrator(den) -> int:
    """分母末尾 0 的个数 = 原点处极点重数(积分环节数)"""
    nu = 0
    for c in reversed(den):
        if abs(c) < 1e-15 and nu < len(den) - 1:
            nu += 1
        else:
            break
    return nu


# ---------------------------------------------------------------- 数值计算器

_WS = {}          # 持久工作区变量
_PRINT_BUF = []   # 捕获 print, 防止污染 stdio

_NUMPY_NAMES = [
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "hypot",
    "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",
    "exp", "log", "log2", "log10", "sqrt", "cbrt", "abs", "sign",
    "floor", "ceil", "round", "real", "imag", "conj", "angle",
    "pi", "e", "inf", "nan",
    "linspace", "logspace", "arange", "zeros", "ones", "eye", "diag",
    "array", "matrix", "dot", "cross", "vdot", "inner", "outer",
    "roots", "poly", "polyval", "polyadd", "polymul", "polyint", "polyder",
    "min", "max", "sum", "mean", "std", "var", "median", "norm", "trace",
    "argmin", "argmax", "argsort", "sort", "where", "clip", "cumsum",
    "fft", "ifft", "convolve", "correlate",
]
_SAFE_BUILTINS = {
    "range": range, "len": len, "abs": abs, "min": min, "max": max, "sum": sum,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "list": list, "tuple": tuple, "dict": dict, "set": set,
    "int": int, "float": float, "complex": complex, "str": str, "bool": bool,
    "round": round, "sorted": sorted, "reversed": reversed,
    "any": any, "all": all, "True": True, "False": False, "None": None,
    "isinstance": isinstance, "type": type, "Exception": Exception,
    "ValueError": ValueError, "TypeError": TypeError, "ZeroDivisionError": ZeroDivisionError,
}


def _safe_import(name, *args, **kwargs):
    allowed = ("numpy", "np", "matplotlib", "math", "scipy", "cmath", "itertools", "functools", "collections")
    root = name.split(".")[0]
    if root not in allowed:
        raise ImportError(f"只允许 import 这些库: {', '.join(allowed)}")
    return __import__(name, *args, **kwargs)


_SAFE_BUILTINS["__import__"] = _safe_import


def _calc_namespace():
    ns = {k: v for k, v in _WS.items()}
    ns["np"] = np
    ns["signal"] = signal
    ns["math"] = math
    ns["print"] = lambda *a, **k: _PRINT_BUF.append(" ".join(str(x) for x in a))
    for name in _NUMPY_NAMES:
        v = getattr(np, name, None)
        if v is not None:
            ns.setdefault(name, v)
    return ns


def _fmt_value(v) -> str:
    if isinstance(v, np.ndarray):
        return np.array2string(v, precision=6, suppress_small=True, max_line_width=110)
    if isinstance(v, np.bool_):
        return str(bool(v))
    if isinstance(v, (complex, np.complexfloating)):
        return f"{v.real:.8g}{v.imag:+.8g}j"
    if isinstance(v, (float, np.floating)):
        return f"{v:.10g}"
    if isinstance(v, np.integer):
        return str(int(v))
    return repr(v)


def _do_calc(expr: str) -> str:
    _PRINT_BUF.clear()
    expr = expr.strip()
    ns = _calc_namespace()
    g = {"__builtins__": _SAFE_BUILTINS}
    m = re.match(r"^\s*([A-Za-z_]\w*)\s*=(?!=)", expr)
    try:
        base = set(ns.keys())
        if m:
            exec(compile(expr, "<calc>", "exec"), g, ns)  # noqa: S102
            name = m.group(1)
            _WS.clear()
            _WS.update({k: v for k, v in ns.items() if k not in base})
            out = f"{name} = {_fmt_value(ns.get(name))}"
        else:
            val = eval(compile(expr, "<calc>", "eval"), g, ns)  # noqa: S102
            _WS["ans"] = val
            out = "ans = " + _fmt_value(val)
    except Exception as exc:  # noqa: BLE001
        return f"计算出错: {type(exc).__name__}: {exc}\n提示: 用 numpy 语法, 如 np.linalg.eig(A)、A@B、np.linalg.inv(A)"
    if len(out) > 1500:
        out = out[:1500] + "\n...(结果过长已截断)"
    if _PRINT_BUF:
        out += "\nprint: " + "\n".join(_PRINT_BUF[:20])
    return out


@mcp.tool()
def calc(expr: str) -> str:
    """MATLAB 风格数值计算器(numpy 语法), 变量跨调用保存在工作区。

    常用: 矩阵 A=[[1,2],[3,4]] 写作 A=np.array([[1,2],[3,4]]);
    特征值 np.linalg.eig(A); 逆 np.linalg.inv(A); 行列式 np.linalg.det(A);
    解方程 np.linalg.solve(A,b); 转置 A.T; 共轭转置 A.conj().T; 矩阵乘 A@B;
    卷积 np.convolve([1,2,3],[1,1]); 多项式求根 roots([1,3,2]);
    微积分数值: np.trapezoid(y,x) 积分, np.gradient(y,x) 导数;
    sin/cos/exp/sqrt/pi 等可直接用。例: calc("A = np.array([[0,1],[-2,-3]])"),
    然后 calc("np.linalg.eig(A)") 求特征值。"""
    return _do_calc(expr)


# ---------------------------------------------------------------- 函数曲线

def _do_plot_function(expr: str, x_min: float, x_max: float, n: int,
                      logx: bool, logy: bool, polar: bool,
                      title: str, xlabel: str, ylabel: str) -> str:
    if x_min >= x_max:
        return "出错: x_min 应小于 x_max"
    exprs = [e.strip() for e in expr.split(";") if e.strip()]
    if not exprs:
        return "出错: expr 为空"
    if polar:
        x = np.linspace(x_min, x_max, min(max(n, 50), 20000))
    elif logx:
        x = np.logspace(math.log10(max(x_min, 1e-12)), math.log10(max(x_max, 1e-12)), min(max(n, 50), 20000))
    else:
        x = np.linspace(x_min, x_max, min(max(n, 50), 20000))
    ns = _calc_namespace()
    ns["x"] = x
    g = {"__builtins__": _SAFE_BUILTINS}
    ys = []
    for e in exprs:
        try:
            y = eval(compile(e, "<plot>", "eval"), g, ns)  # noqa: S102
        except Exception as exc:  # noqa: BLE001
            return f"表达式 {e!r} 出错: {exc}"
        y = np.asarray(y, dtype=complex if np.iscomplexobj(y) else float)
        if y.shape != x.shape:
            try:
                y = np.broadcast_to(y, x.shape).copy()
            except Exception:
                return f"表达式 {e!r} 结果形状 {y.shape} 与 x 不匹配"
        ys.append((e, y))
    fig, ax = plt.subplots(figsize=(8.5, 5.8), subplot_kw={"projection": "polar"} if polar else {})
    for e, y in ys:
        if np.iscomplexobj(y):
            ax.plot(x, y.real, label=f"Re({e})")
            ax.plot(x, y.imag, "--", label=f"Im({e})")
        else:
            yy = np.where(np.isfinite(y) & (np.abs(y) < 1e15), y, np.nan) if not logy else y
            ax.plot(x, yy, label=e)
    if not polar:
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.35)
        ax.set_xlabel(xlabel or ("θ (rad)" if polar else "x"))
        if ylabel:
            ax.set_ylabel(ylabel)
    ax.set_title(title or "、".join(exprs), fontsize=11)
    if len(fig.axes[0].get_lines()) > 0 and len(exprs) >= 1:
        ax.legend(fontsize=9)
    path = _save(fig, "func")
    return f"已保存: {path}\n表达式: {'; '.join(exprs)}\nx 范围: [{x_min:g}, {x_max:g}], 采样 {len(x)} 点"


@mcp.tool()
def plot_function(expr: str, x_min: float = -10.0, x_max: float = 10.0, n: int = 2000,
                  logx: bool = False, logy: bool = False, polar: bool = False,
                  title: str = "", xlabel: str = "", ylabel: str = "") -> str:
    """画函数曲线 y=f(x)。expr 用 numpy 语法、自变量为 x, 多条曲线用分号分隔。

    例: expr="sin(x)/x" ; expr="exp(-0.2*x)*cos(x); exp(-0.2*x); -exp(-0.2*x)" (衰减振荡及包络);
    polar=True 时 expr 为 r=f(θ), x 即 θ, 默认范围建议 0~2π;
    logx/logy 控制对数坐标。支持 calc 工作区里的变量。"""
    return _do_plot_function(expr, x_min, x_max, n, logx, logy, polar, title, xlabel, ylabel)


# ---------------------------------------------------------------- 伯德图

def _do_bode(num, den, w_min, w_max, title) -> str:
    num, den = _tf(num, den)
    lo, hi = _auto_wrange(num, den)
    lo = w_min if w_min else lo
    hi = w_max if w_max else hi
    if lo >= hi:
        lo, hi = 1e-2, 1e2
    w = np.logspace(math.log10(lo), math.log10(hi), 4000)
    H = _freq_resp(num, den, w)
    mag_db = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
    ph = np.degrees(np.unwrap(np.angle(H)))
    mg = _margins(w, mag_db, ph)
    gst = _tf_str(num, den)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True)
    ax1.semilogx(w, mag_db, "b-", lw=1.4)
    ax1.grid(True, which="both", alpha=0.35)
    ax1.set_ylabel("幅值 (dB)")
    ax2.semilogx(w, ph, "b-", lw=1.4)
    ax2.grid(True, which="both", alpha=0.35)
    ax2.set_ylabel("相位 (°)")
    ax2.set_xlabel("ω (rad/s)")
    fig.suptitle(title or f"Bode 图  {gst}", fontsize=11)

    lines = [f"已保存: {{PATH}}", f"系统: {gst}", "稳定裕度:"]
    if "GM_dB" in mg:
        ax1.plot(mg["w_pc"], -mg["GM_dB"], "ro", ms=5)
        ax1.axvline(mg["w_pc"], color="r", ls="--", lw=0.8, alpha=0.6)
        ax2.axvline(mg["w_pc"], color="r", ls="--", lw=0.8, alpha=0.6)
        ax2.axhline(-180, color="gray", ls=":", lw=0.8)
        lines.append(f"  幅值裕度 GM = {mg['GM_dB']:.2f} dB (即 {mg['gm_ratio']:.3g} 倍), "
                     f"相角穿越频率 ω_pc = {mg['w_pc']:.4g} rad/s")
    else:
        lines.append("  在给定频率范围内未找到相角穿越点(相角未穿过 -180°)")
    if "PM_deg" in mg:
        ax1.axhline(0, color="gray", ls=":", lw=0.8)
        ax1.axvline(mg["w_gc"], color="g", ls="--", lw=0.8, alpha=0.6)
        ax2.plot(mg["w_gc"], mg["ph_at_gc"], "go", ms=5)
        ax2.axvline(mg["w_gc"], color="g", ls="--", lw=0.8, alpha=0.6)
        lines.append(f"  相角裕度 PM = {mg['PM_deg']:.2f}°, "
                     f"幅值穿越频率 ω_gc = {mg['w_gc']:.4g} rad/s")
    else:
        lines.append("  在给定频率范围内未找到幅值穿越点(0 dB)")
    lines.append("判断: GM>0dB 且 PM>0° (最小相位系统) 通常意味着闭环稳定。")
    path = _save(fig, "bode")
    return "\n".join(lines).replace("{PATH}", path)


@mcp.tool()
def plot_bode(num: list[float], den: list[float], w_min: float | None = None,
              w_max: float | None = None, title: str = "") -> str:
    """画传递函数的伯德图(Bode 图, 幅频+相频), 并自动计算幅值裕度 GM 与相角裕度 PM。

    num/den 为按 s 降幂排列的系数列表(同 MATLAB tf):
      G=10/(s²+2s+10) -> num=[10], den=[1,2,10]
      G=1/(s(s+1)(s+2)) -> num=[1], den=[1,3,2,0]
    w_min/w_max 可指定角频率范围(rad/s), 省略则自动。返回图片路径和裕度数值。"""
    return _do_bode(num, den, w_min, w_max, title)


# ---------------------------------------------------------------- 奈奎斯特图

def _do_nyquist(num, den, w_min, w_max, xlim, ylim) -> str:
    num, den = _tf(num, den)
    lo, hi = _auto_wrange(num, den)
    lo = w_min if w_min else lo / 50
    hi = w_max if w_max else hi * 20
    if lo >= hi:
        lo, hi = 1e-3, 1e3
    wp = np.logspace(math.log10(lo), math.log10(hi), 4000)
    Gp = _freq_resp(num, den, wp)
    vis = np.concatenate([np.conj(Gp[::-1]), Gp])   # w: -hi→-lo, lo→hi

    # 统计绕 -1 点圈数(逆时针为正); 含积分环节时补上半径无穷大的映射圆弧
    nu = _num_integrator(den)
    pts = vis
    if nu > 0:
        R = 10 * max(np.abs(vis).max(), 10.0)
        arc = R * np.exp(1j * np.linspace(0.5 * nu * math.pi, -0.5 * nu * math.pi, 200))
        pts = np.concatenate([np.conj(Gp[::-1]), arc, Gp])
    P = int(np.sum(np.roots(np.asarray(den, float)).real > 0))
    vec = 1 + pts
    ang = np.unwrap(np.angle(vec))
    W = int(round((ang[-1] - ang[0]) / (2 * math.pi)))
    Z = P - W
    gst = _tf_str(num, den)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.plot(vis.real, vis.imag, "b-", lw=1.3)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.plot(-1, 0, "rx", ms=11, mew=2)
    ax.annotate("(-1, j0)", xy=(-1, 0), xytext=(-1 + 0.06, 0.1), color="r", fontsize=10)
    # 方向箭头(正频率支: 低频→高频)
    i = len(vis) // 2 + len(vis) // 8
    ax.annotate("", xy=(vis[i + 1].real, vis[i + 1].imag),
                xytext=(vis[i].real, vis[i].imag),
                arrowprops=dict(arrowstyle="->", color="b", lw=1.5))
    ax.set_xlabel("Re(G)")
    ax.set_ylabel("Im(G)")
    ax.set_aspect("equal", adjustable="datalim")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.35)
    ax.set_title(f"Nyquist 图  {gst}", fontsize=10)
    path = _save(fig, "nyquist")

    lines = [f"已保存: {path}", f"系统: {gst}",
             f"开环右半平面极点数 P = {P}",
             f"对 (-1, j0) 的环绕圈数 W = {W} (逆时针为正, 负值=顺时针)",
             f"闭环右半平面极点数 Z = P - W = {Z} → 闭环{'稳定' if Z == 0 else '不稳定'}"]
    if nu:
        lines.append(f"注: 开环含 {nu} 个积分环节(原点极点), 补了大圆弧后计圈。")
    lines.append("如需放大查看 -1 点附近, 可传 xlim/ylim, 如 xlim=[-3,1], ylim=[-2,2]。")
    return "\n".join(lines)


@mcp.tool()
def plot_nyquist(num: list[float], den: list[float], w_min: float | None = None,
                 w_max: float | None = None, xlim: list[float] | None = None,
                 ylim: list[float] | None = None) -> str:
    """画传递函数的奈奎斯特图(Nyquist 图), 自动标注 (-1, j0) 点、统计环绕圈数,
    并用奈奎斯特判据给出闭环稳定性结论 (Z = P - W, Z=0 即闭环稳定)。

    num/den 为按 s 降幂排列的系数列表(同 MATLAB tf), 例:
      G=1/(s(s+1)(s+2)) -> num=[1], den=[1,3,2,0]
    xlim/ylim 可放大 -1 附近, 如 xlim=[-3,1], ylim=[-2,2]。"""
    return _do_nyquist(num, den, w_min, w_max, xlim, ylim)


# ---------------------------------------------------------------- 根轨迹

def _do_root_locus(num, den, k_max, n_points, title) -> str:
    num, den = _tf(num, den)
    poles = np.roots(np.asarray(den, float))
    zeros = np.roots(np.asarray(num, float))
    n_p, n_z = len(poles), len(zeros)
    if k_max is None:
        scale = max(np.abs(den).max() / max(np.abs(num).max(), 1e-12), 1e-9)
        k_max = float(1e4 * scale)
    ks = np.concatenate([[0.0], np.logspace(math.log10(max(k_max * 1e-6, 1e-12)), math.log10(k_max), n_points)])
    allroots = np.array([np.roots(np.polyadd(np.asarray(den, float), k * np.asarray(num, float))) for k in ks])

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(allroots.real, allroots.imag, ".", color="steelblue", ms=1.6)
    ax.plot(poles.real, poles.imag, "rx", ms=10, mew=2, label="开环极点")
    if n_z:
        ax.plot(zeros.real, zeros.imag, "go", ms=8, mfc="none", mew=1.8, label="开环零点")
    lines = [f"已保存: {{PATH}}", f"系统: {_tf_str(num, den)}",
             f"开环极点: {_fmt_poles(poles)}"]
    if n_z:
        lines.append(f"开环零点: {_fmt_poles(zeros)}")
    d = n_p - n_z
    if d >= 1:
        sigma = (poles.sum() - zeros.sum()) / d
        thetas = [(2 * l + 1) * math.pi / d for l in range(d)]
        R = 1.3 * max(1.0, np.abs(allroots).max(), abs(sigma))
        for th in thetas:
            ax.plot([sigma, sigma + R * math.cos(th)], [0, R * math.sin(th)],
                    "--", color="gray", lw=0.9, alpha=0.8)
        ax.plot(sigma, 0, "s", color="gray", ms=5)
        theta_str = "°, ".join(f"{math.degrees(t):.1f}" for t in thetas)
        lines.append(f"渐近线: 交点 σ = {sigma:.4g}, 与实轴夹角 {theta_str}°")
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel("Re(s)")
    ax.set_ylabel("Im(s)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(title or f"根轨迹  {_tf_str(num, den)}  (k: 0 → {k_max:.3g})", fontsize=10)
    path = _save(fig, "rlocus")
    return "\n".join(lines).replace("{PATH}", path)


def _fmt_pole(p, paired: bool = False) -> str:
    if abs(p.imag) < 1e-9:
        return f"{p.real:.4g}"
    if paired:
        return f"{p.real:.4g}±{abs(p.imag):.4g}j"
    return f"{p.real:.4g}{p.imag:+.4g}j"


def _fmt_poles(roots) -> str:
    """把共轭对合并成 a±bj 的形式展示"""
    rs = sorted(roots, key=lambda z: (round(z.real, 9), round(z.imag, 9)))
    out, i = [], 0
    while i < len(rs):
        if (i + 1 < len(rs) and rs[i].imag > 1e-9
                and abs(rs[i] - rs[i + 1].conj()) < 1e-6):
            out.append(_fmt_pole(rs[i], paired=True))
            i += 2
        else:
            out.append(_fmt_pole(rs[i]))
            i += 1
    return ", ".join(out)


@mcp.tool()
def plot_root_locus(num: list[float], den: list[float], k_max: float | None = None,
                    n_points: int = 3000, title: str = "") -> str:
    """画传递函数的根轨迹(Evan's root locus), 含开环零极点、渐近线及其交点。

    num/den 为按 s 降幂排列的系数列表(同 MATLAB tf), 例:
      G=K/((s+1)(s+2)) -> num=[1], den=[1,3,2]
    k_max 可指定增益上限(省略自动)。"""
    return _do_root_locus(num, den, k_max, n_points, title)


# ---------------------------------------------------------------- 时域响应

def _auto_tend(num, den) -> float:
    poles = np.roots(np.asarray(den, float))
    if poles.size == 0:
        return 10.0
    re = poles.real
    im = np.abs(poles.imag)
    if np.any(re > 1e-9):                      # 不稳定
        r = np.abs(poles[re > 1e-9])
        return float(np.clip(5.0 / r.min(), 1.0, 500))
    neg = re[re < -1e-9]
    tau = -neg.max() if neg.size else 1.0      # 最慢的稳定极点
    T = 8.0 / tau
    if np.any(im > 1e-9):
        T = max(T, 4 * 2 * math.pi / im.max())
    if np.any(np.abs(re) <= 1e-9):             # 含积分/临界极点
        T = max(T, 12.0)
    return float(np.clip(T, 0.5, 5000))


def _time_response(num, den, kind: str, t_end: float | None):
    num, den = _tf(num, den)
    sys = signal.lti(num, den)
    T = t_end if t_end else _auto_tend(num, den)
    t = np.linspace(0, T, 5000)
    if kind == "step":
        _, y = signal.step(sys, T=t)
    else:
        _, y = signal.impulse(sys, T=t)
    return num, den, t, y


def _step_metrics(t, y):
    yf = y[-1]
    lines = []
    if abs(yf) < 1e-9 or not np.isfinite(yf) or y[-1] != y[-1]:
        return lines, yf
    yp = y.max()
    overshoot = (yp - yf) / abs(yf) * 100
    if overshoot > 0.05:
        tp = t[int(np.argmax(y))]
        lines.append(f"超调量 σ% = {overshoot:.2f}%  (峰值 {yp:.4g}, 峰值时间 tp = {tp:.4g}s)")
    # 上升时间 10%→90%
    try:
        i10 = np.where(y >= 0.1 * yf)[0][0]
        i90 = np.where(y >= 0.9 * yf)[0][0]
        if i90 > i10:
            lines.append(f"上升时间 tr(10%~90%) = {t[i90] - t[i10]:.4g}s")
    except IndexError:
        pass
    # 调节时间(±2%)
    outside = np.where(np.abs(y - yf) > 0.02 * abs(yf))[0]
    if outside.size:
        ts = t[outside[-1]]
        lines.append(f"调节时间 ts(±2%) ≈ {ts:.4g}s")
    lines.append(f"稳态值 y(∞) = {yf:.6g}")
    return lines, yf


def _do_time(num, den, kind, t_end, title) -> str:
    num, den, t, y = _time_response(num, den, kind, t_end)
    gst = _tf_str(num, den)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(t, y, "b-", lw=1.4)
    ax.grid(True, alpha=0.35)
    ax.set_xlabel("t (s)")
    ax.set_ylabel(kind == "step" and "y(t)" or "y(t)")
    kind_cn = "单位阶跃响应" if kind == "step" else "单位脉冲响应"
    ax.set_title(title or f"{kind_cn}  {gst}", fontsize=10)
    lines = [f"已保存: {{PATH}}", f"系统: {gst}", f"{kind_cn}, 时长 {t[-1]:.4g}s:"]
    if kind == "step":
        m, yf = _step_metrics(t, y)
        lines += ["  " + x for x in m] or ["  (响应未进入稳态或稳态值为 0, 无法计算典型指标)"]
        if abs(yf) > 1e-9:
            ax.axhline(yf, color="gray", ls=":", lw=0.9)
    poles = np.roots(np.asarray(den, float))
    lines.append("闭环极点(此处即所给系统极点): " + _fmt_poles(poles))
    path = _save(fig, kind)
    return "\n".join(lines).replace("{PATH}", path)


@mcp.tool()
def plot_step(num: list[float], den: list[float], t_end: float | None = None,
              title: str = "") -> str:
    """画传递函数的单位阶跃响应, 自动计算超调量 σ%、峰值时间 tp、上升时间 tr、
    调节时间 ts(±2%) 和稳态值。num/den 为按 s 降幂排列的系数(同 MATLAB tf),
    例: G=1/(s²+2s+1) -> num=[1], den=[1,2,1]。t_end 可指定仿真时长(秒), 省略自动。"""
    return _do_time(num, den, "step", t_end, title)


@mcp.tool()
def plot_impulse(num: list[float], den: list[float], t_end: float | None = None,
                 title: str = "") -> str:
    """画传递函数的单位脉冲响应。num/den 为按 s 降幂排列的系数(同 MATLAB tf)。
    t_end 可指定仿真时长(秒), 省略自动。"""
    return _do_time(num, den, "impulse", t_end, title)


# ---------------------------------------------------------------- 零极点

def _do_pole_zero(num, den, title) -> str:
    num, den = _tf(num, den)
    poles = np.roots(np.asarray(den, float))
    zeros = np.roots(np.asarray(num, float))
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    if len(zeros):
        ax.plot(zeros.real, zeros.imag, "go", ms=9, mfc="none", mew=1.8, label="零点")
    ax.plot(poles.real, poles.imag, "rx", ms=10, mew=2, label="极点")
    span = max(1.0, np.abs(np.concatenate([poles, zeros if len(zeros) else np.array([1.0])])).max() * 1.3)
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlabel("Re(s)")
    ax.set_ylabel("Im(s)")
    ax.set_title(title or f"零极点分布  {_tf_str(num, den)}", fontsize=10)
    path = _save(fig, "pzmap")
    lines = [f"已保存: {path}", f"系统: {_tf_str(num, den)}"]
    for p in poles:
        if abs(p.imag) > 1e-9:
            wn = abs(p)
            zeta = -p.real / wn if wn > 0 else 0.0
            lines.append(f"  极点 {p.real:.4g}{p.imag:+.4g}j : ωn={wn:.4g} rad/s, ζ={zeta:.4g}")
        else:
            lines.append(f"  极点 {p.real:.4g}")
    for z in zeros:
        lines.append(f"  零点 {z.real:.4g}{z.imag:+.4g}j" if abs(z.imag) > 1e-9 else f"  零点 {z.real:.4g}")
    stable = np.all(poles.real < 1e-9)
    lines.append(f"极点全部位于左半平面: {'是 → 系统稳定' if stable else '否 → 系统不稳定(或临界)'}")
    return "\n".join(lines)


@mcp.tool()
def plot_pole_zero(num: list[float], den: list[float], title: str = "") -> str:
    """画传递函数的零极点分布图(s 平面), 并列出每个极点的固有频率 ωn 和阻尼比 ζ。
    num/den 为按 s 降幂排列的系数(同 MATLAB tf)。"""
    return _do_pole_zero(num, den, title)


# ---------------------------------------------------------------- 自定义代码

@mcp.tool()
def run_plot_code(code: str) -> str:
    """逃生舱: 直接运行 matplotlib/numpy 画图代码, 覆盖上面工具画不了的图
    (3D 曲面/向量场、lsim 任意输入仿真、多子图对比、信号处理等)。

    命名空间已提供: np(numpy), plt(matplotlib.pyplot), math, signal(scipy.signal)。
    不要调用 plt.show(); 画完的每个 figure 会自动保存为 PNG 并返回路径。
    例(任意输入响应):
      t=np.linspace(0,10,2000); u=np.sin(2*t)\n
      t2,y,x=signal.lsim(signal.lti([1],[1,2,1]), U=u, T=t)\n
      plt.plot(t2,u,'--',label='u(t)'); plt.plot(t2,y,label='y(t)'); plt.legend(); plt.xlabel('t/s')
    """
    _PRINT_BUF.clear()
    ns = {"np": np, "plt": plt, "math": math, "signal": signal,
          "print": lambda *a, **k: _PRINT_BUF.append(" ".join(str(x) for x in a))}
    g = {"__builtins__": _SAFE_BUILTINS}
    try:
        exec(compile(code, "<plot_code>", "exec"), g, ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        plt.close("all")
        return f"代码出错: {type(exc).__name__}: {exc}"
    nums = plt.get_fignums()
    if not nums:
        return "代码运行成功但没有创建任何图形(plt.figure/plt.plot), 无图可保存。"
    paths = []
    for num in nums:
        fig = plt.figure(num)
        paths.append(_save(fig, "code"))
    plt.close("all")
    out = "已保存: " + "\n已保存: ".join(paths)
    if _PRINT_BUF:
        out += "\nprint: " + "\n".join(_PRINT_BUF[:30])
    return out


if __name__ == "__main__":
    mcp.run()
