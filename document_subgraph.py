from typing import TypedDict, Optional, Any, Dict, List
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("langchain 未安装，将使用简单文本分割器")

try:
    from config import document_config
except ImportError:
    logger.warning("配置文件未找到，使用默认参数")
    class DocumentConfig:
        chunk_size = 500
        chunk_overlap = 50
    document_config = DocumentConfig()

class DocumentState(TypedDict):
    file_content: bytes
    file_name: str
    text_chunks: List[str]
    embeddings: Optional[np.ndarray]
    index_path: str
    success: bool
    error: Optional[str]
    chunks_count: int

class DocumentIngestionSubGraph:
    def __init__(self, embedding_model=None, index_dir: str = "./vector_db"):
        self.embedding_model = embedding_model
        self.index_dir = index_dir
        self.chunk_size = document_config.chunk_size
        self.chunk_overlap = document_config.chunk_overlap
        os.makedirs(index_dir, exist_ok=True)
    
    async def _extract_text(self, file_content: bytes, file_name: str) -> str:
        if file_name.lower().endswith(".pdf"):
            try:
                from pdfminer.high_level import extract_text
                from io import BytesIO
                return extract_text(BytesIO(file_content))
            except ImportError:
                logger.warning("pdfminer not installed, using raw content")
                return file_content.decode('utf-8', errors='ignore')
        elif file_name.lower().endswith(".txt"):
            return file_content.decode('utf-8', errors='ignore')
        else:
            raise ValueError(f"不支持的文件格式: {file_name}")
    
    async def _text_splitter(self, text: str) -> List[str]:
        if LANGCHAIN_AVAILABLE:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
            )
            chunks = text_splitter.split_text(text)
        else:
            chunk_size = self.chunk_size
            chunk_overlap = self.chunk_overlap

            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                if end < len(text):
                    for sep in ['。', '！', '？', '\n', '，', ' ']:
                        last_sep = text.rfind(sep, start, end)
                        if last_sep != -1 and last_sep > start + chunk_overlap:
                            end = last_sep + 1
                            break
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start = end - chunk_overlap

        logger.info(f"文本切割完成: {len(chunks)} 个片段")
        return chunks
    
    async def _embedding_generator(self, chunks: List[str]) -> Optional[np.ndarray]:
        if self.embedding_model is None:
            logger.warning("嵌入模型未配置，跳过嵌入生成")
            return None
        
        embeddings = self.embedding_model.embed_documents(chunks)
        embeddings = np.array(embeddings).astype('float32')
        
        logger.info(f"嵌入生成完成: {embeddings.shape[0]} 个向量")
        return embeddings
    
    async def _vector_storage(self, embeddings: np.ndarray, chunks: List[str], file_name: str) -> str:
        if embeddings is None:
            logger.warning("无嵌入向量，跳过向量入库")
            return ""
        
        try:
            import faiss
            index_path = os.path.join(self.index_dir, f"{os.path.splitext(file_name)[0]}.index")
            
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            
            faiss.write_index(index, index_path)
            
            chunks_path = os.path.join(self.index_dir, f"{os.path.splitext(file_name)[0]}.chunks")
            with open(chunks_path, 'w', encoding='utf-8') as f:
                for i, chunk in enumerate(chunks):
                    f.write(f"=== CHUNK {i} ===\n{chunk}\n\n")
            
            logger.info(f"向量入库完成: {index_path}")
            return index_path
        except ImportError:
            logger.warning("faiss not installed, skipping vector storage")
            return ""
    
    async def run(self, state: DocumentState) -> DocumentState:
        try:
            full_text = await self._extract_text(state["file_content"], state["file_name"])
            chunks = await self._text_splitter(full_text)
            embeddings = await self._embedding_generator(chunks)
            index_path = await self._vector_storage(embeddings, chunks, state["file_name"])
            
            return {
                **state,
                "text_chunks": chunks,
                "embeddings": embeddings,
                "index_path": index_path,
                "chunks_count": len(chunks),
                "success": True,
                "error": None
            }
        
        except Exception as e:
            logger.error(f"文档处理失败: {str(e)}")
            return {
                **state,
                "success": False,
                "error": str(e)
            }
    
    def search(self, query: str, top_k: int = 3, kb_ids: List[str] = None) -> List[str]:
        if self.embedding_model is None:
            logger.warning("嵌入模型未配置，无法进行检索")
            return []
        
        try:
            import faiss
            query_embedding = self.embedding_model.embed_query(query)
            query_embedding = np.array([query_embedding]).astype('float32')
            
            results = []
            for idx_file in os.listdir(self.index_dir):
                if idx_file.endswith('.index'):
                    # 检查是否需要按知识库过滤
                    kb_id = idx_file.replace('.index', '')
                    if kb_ids and kb_id not in kb_ids:
                        continue
                    
                    index_path = os.path.join(self.index_dir, idx_file)
                    chunks_path = os.path.join(self.index_dir, idx_file.replace('.index', '.chunks'))
                    
                    index = faiss.read_index(index_path)
                    distances, indices = index.search(query_embedding, top_k)
                    
                    with open(chunks_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for i in indices[0]:
                        if i != -1:
                            chunks = content.split("=== CHUNK ")[1:]
                            if i < len(chunks):
                                chunk_content = chunks[i].split("\n===\n")[1].strip()
                                results.append((distances[0][list(indices[0]).index(i)], chunk_content))
            
            results.sort(key=lambda x: x[0])
            return [r[1] for r in results[:top_k]]
        
        except Exception as e:
            logger.error(f"向量检索失败: {str(e)}")
            return []
