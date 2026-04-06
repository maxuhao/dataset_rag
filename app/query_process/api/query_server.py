import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks,Request
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from app.clients.mongo_history_utils import get_recent_messages, clear_history
from app.core.logger import logger
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state
from app.utils.path_util import PROJECT_ROOT
from fastapi.responses import FileResponse

from app.utils.sse_utils import create_sse_queue, sse_generator, push_to_session, SSEEvent
from app.utils.task_utils import get_task_result, update_task_status

#定义fastapi对象
app = FastAPI(title="query service", description="掌柜智库查询服务!")

# 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 仅测试用
    allow_methods=["*"], # 允许所有方法
    allow_headers=["*"], # 允许所有头
    allow_credentials=True, # 允许携带cookie
)

#查询健康状态
@app.get("/health")
async def health():
    logger.info("触发后台检测接口，数据正常！")
    return {"status": "ok"}

# 返回chat.html页面
@app.get("/chat")
async def chat():
    path = PROJECT_ROOT / "app" / "query_process" / "page" / "chat.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path)

class QueryRequest(BaseModel):
    query: str = Field(...,title = "查询内容，必须传递")
    session_id : str = Field(None,title="会话ID，可以不传递，后台会生成uuid")
    is_stream: bool = Field(False,title="是否流式返回结果")

def run_query_graph(query: str, session_id: str, is_stream: bool):
    # 更新任务状态
    update_task_status(session_id, "processing", is_stream)

    state = create_query_default_state(session_id=session_id, original_query=query, is_stream=is_stream)
    try:
        # 执行图
        query_app.invoke(state)
        # 更新任务状态
        update_task_status(session_id, "completed", is_stream)
    except Exception as e:
        logger.error(f"query:{query}:查询异常！{e}")
        # 修改 event = progress
        update_task_status(session_id, "failed", is_stream)
        # 推送指定类型的事件
        push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})




# 查询接口 开始执行graph 并查询 rag 并返回结果
@app.post("/query")
async def query(request: QueryRequest,background_tasks: BackgroundTasks):
    """
    :param request:  请求参数
    :param background_tasks: 异步执行函数 is_stream = True
    :return:
    """
    query_str = request.query
    session_id = request.session_id or str(uuid.uuid4())
    is_stream = request.is_stream

    if is_stream: # 流式返回结果
        # 只要开启流式处理，我们业务中就是将数据，插入到队列中
        # 创建当前session_id 对应的队列
        create_sse_queue(session_id)
        # 创建一个异步任务 立即返回结果给前端 | 中间过程使用 sse 一点一点 推送 给前端
        background_tasks.add_task(run_query_graph, query_str, session_id, is_stream)
        logger.info(f"query:{query_str}:开启了异步查询！！")
        return {
            "session_id": session_id,
            "message": "本次查询处理中..."
        }
    else: # 非流式返回结果
        run_query_graph(query_str, session_id, is_stream)

        # 获取最后一个节点插入的结果！ node_answer_output(answer)
        answer = get_task_result(session_id, "answer") # task_utils 封装的一个存储会话结果的函数
        logger.info(f"query:{query_str}:开启了同步查询！！")
        return {
            "session_id": session_id,
            "answer": answer, # 返回结果
            "message": "查询成功！",
            "done_list":[], # 待完成列表
        }

@app.get("/stream/{session_id}")
async def stream(session_id: str,request: Request):
    """
    :param session_id:
    :param request:
    :return:
    """
    logger.info(f"session_id:{session_id}:开始返回结果...")
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream"
    )

@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 10):
    """
    :param limit:
    :param session_id:
    :return:
    """
    # 查询指定会话的最近N条对话记录
    chats =  get_recent_messages(session_id, limit)
    logger.info(f"session_id:{session_id}:开始返回结果...")
    return {
        "session_id": session_id,
        "chats": chats,
    }

@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    """
    :param session_id:
    :return:
    """
    # 删除指定会话的聊天记录
    delete_result =clear_history(session_id)
    logger.info(f"session_id:{session_id}:删除结果：{delete_result}")
    return {
        "delete_result": delete_result,
        "message": f"{session_id}删除成功！",
    }





if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
















