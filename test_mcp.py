# -*- coding: utf-8 -*-
"""MCP stdio 端到端测试: 像真实客户端一样起 server 进程、列工具、调工具。"""
import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


async def main():
    params = StdioServerParameters(
        command=PY,
        args=[os.path.join(HERE, "server.py")],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print("工具列表:", names)
            assert {"calc", "plot_bode", "plot_nyquist", "plot_step"} <= set(names), "核心工具缺失"

            r = await session.call_tool("calc", {"expr": "A = np.array([[1., 2.], [3., 4.]])"})
            print("\ncalc:", r.content[0].text)
            r = await session.call_tool("calc", {"expr": "np.linalg.eig(A)[0]"})
            print("calc 特征值:", r.content[0].text)

            r = await session.call_tool("plot_bode", {"num": [1], "den": [1, 3, 2, 0]})
            print("\nplot_bode:\n", r.content[0].text)

            r = await session.call_tool("plot_function",
                                        {"expr": "exp(-0.3*x)*cos(2*x); exp(-0.3*x); -exp(-0.3*x)",
                                         "x_min": 0, "x_max": 12, "title": "衰减振荡"})
            print("\nplot_function:\n", r.content[0].text)
            print("\nMCP 端到端 OK")


asyncio.run(main())
