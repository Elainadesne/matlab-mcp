# -*- coding: utf-8 -*-
"""本地直接调用 server 内部函数, 验证各工具能跑通、数值与教科书结果一致。"""
import io
import os
import sys
import warnings
from contextlib import redirect_stderr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter("always")

buf = io.StringIO()
with redirect_stderr(buf):  # matplotlib 字体等告警进 buf, 不影响 stdio
    import server as S

    print("== calc ==")
    print(S._do_calc("A = np.array([[0,1],[-2,-3]])"))
    print(S._do_calc("np.linalg.eig(A)"))
    print(S._do_calc("ans"))

    print("\n== plot_function ==")
    print(S._do_plot_function("sin(x)/x", -20, 20, 2000, False, False, False, "sinc", "x", "y"))

    print("\n== bode G=1/(s(s+1)(s+2))  教科书: GM=6(15.6dB)@w_pc=1.414, PM≈53°@w_gc≈0.45 ==")
    print(S._do_bode([1], [1, 3, 2, 0], None, None, ""))

    print("\n== bode G=10/(s^2+2s+10)  教科书: PM≈-? 检查穿越 ==")
    print(S._do_bode([10], [1, 2, 10], None, None, ""))

    print("\n== nyquist G=1/(s(s+1)(s+2))  预期: P=0, W=0, 闭环稳定 ==")
    print(S._do_nyquist([1], [1, 3, 2, 0], None, None, None, None))

    print("\n== nyquist G=10/(s(s+1)(s+2))  预期: P=0, W=-2, 闭环不稳定 ==")
    print(S._do_nyquist([10], [1, 3, 2, 0], None, None, [-3, 1], [-2, 2]))

    print("\n== nyquist G=100/((s+1)(s+2)(s+3))  预期: W=-2, 不稳定 ==")
    print(S._do_nyquist([100], [1, 6, 11, 6], None, None, None, None))

    print("\n== root_locus G=K/((s+1)(s+2))  渐近线 σ=-1.5, ±90° ==")
    print(S._do_root_locus([1], [1, 3, 2], None, 3000, ""))

    print("\n== root_locus G=K/(s(s+1)(s+2))  渐近线 σ=-1, ±60/180° ==")
    print(S._do_root_locus([1], [1, 3, 2, 0], None, 3000, ""))

    print("\n== step G=1/(s^2+0.4s+1)  欠阻尼二阶: ζ=0.2, σ%≈52.7% ==")
    print(S._do_time([1], [1, 0.4, 1], "step", None, ""))

    print("\n== step G=1/s (积分环节, 斜坡) ==")
    print(S._do_time([1], [1, 0], "step", 8.0, ""))

    print("\n== impulse ==")
    print(S._do_time([1], [1, 2, 1], "impulse", None, ""))

    print("\n== pole_zero ==")
    print(S._do_pole_zero([1, 2], [1, 0.6, 4], ""))

    print("\n== run_plot_code (lsim 任意输入) ==")
    print(S.run_plot_code(
        "import numpy as np\n"
        "t = np.linspace(0, 10, 2000)\n"
        "u = np.sin(2*t)\n"
        "t2, y, _ = signal.lsim(signal.lti([1],[1,2,1]), U=u, T=t)\n"
        "plt.plot(t2, u, '--', label='u(t)=sin2t')\n"
        "plt.plot(t2, y, label='y(t)')\n"
        "plt.legend(); plt.grid(alpha=0.4); plt.xlabel('t/s'); plt.title('中文标题测试: 正弦输入响应')\n"
    ))

print("\n== 生成的图片 ==")
for f in sorted(os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"))):
    print(" ", f)

err = buf.getvalue()
glyph_issues = [ln for ln in err.splitlines() if "Glyph" in ln or "missing from" in ln]
print("\n字体告警:", glyph_issues[:5] if glyph_issues else "无")
print("其他 stderr(前10行):")
for ln in [l for l in err.splitlines() if l not in glyph_issues][:10]:
    print("  ", ln)
