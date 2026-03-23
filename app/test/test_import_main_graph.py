import json
from pathlib import Path

from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state
import sys
from app.core.logger import logger

logger.info("===== 开始测试 =====")

path = Path(__file__).resolve().parents[2] / "doc/万用表RS-12的使用.pdf"

initial_state = create_default_state(local_file_path=str(path))
# initial_state = create_default_state(pdf_path="万用表RS-12的使用.pdf")
final_state = None

# 在序列化之前，将 Path 对象转换为字符串
def convert_path_to_str(state):
    """递归地将 state 中的 Path 对象转换为字符串"""
    if isinstance(state, dict):
        return {k: convert_path_to_str(v) for k, v in state.items()}
    elif isinstance(state, list):
        return [convert_path_to_str(item) for item in state]
    elif hasattr(state, '__fspath__'):  # Path 对象
        return str(state)
    else:
        return state


# 只输出更最终的状态值（字典形式），不包含节点名称、执行日志、元数据等额外信息
for event in kb_import_app.stream(initial_state):
    for key, value in event.items():
        logger.info(f"节点: {key}")
        final_state = value

# 格式化输出最终状态
logger.info(f"最终状态: {json.dumps(convert_path_to_str(final_state), indent=4, ensure_ascii=False)}")

logger.info("图结构:")
# uv add grandalf
kb_import_app.get_graph().print_ascii()

logger.info("===== 测试结束 =====")
