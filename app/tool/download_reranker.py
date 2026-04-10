#modelscope：字节跳动旗下的魔搭 AI 社区（国内主流的开源模型仓库，类似 Hugging Face，适配国内网络环境）
from modelscope.hub.snapshot_download import snapshot_download

local_dir = r"D:\ai_models\modelscope_cache\models\rerank"

snapshot_download(
  #北京人工智能研究院（BAAI） 发布的大尺寸 BGE 重排序模型，专为文本相关性排序设计，精度较高
  model_id="BAAI/bge-reranker-large",
  cache_dir=local_dir,
)

print("下载完成，模型目录：", local_dir)