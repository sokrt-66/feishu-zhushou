"""
向量存储与混合检索模块
使用 FAISS 实现高性能向量检索，支持混合检索（向量+关键词）
"""
import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """基于 FAISS 的向量存储"""
    
    def __init__(self, dimension: int = 768, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type
        
        # FAISS 索引
        self.index = None
        self.documents = []
        self.metadata = []
        
        # 初始化索引
        self._init_index()
    
    def _init_index(self):
        """初始化 FAISS 索引"""
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS 未安装，使用模拟向量存储")
            self.index = None
            return
        
        if self.index_type == "flat":
            # 精确搜索
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ivf":
            # 倒排索引（适合大规模数据）
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "hnsw":
            # HNSW 图索引（高性能）
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            logger.warning(f"未知索引类型: {self.index_type}，使用默认 flat 索引")
            self.index = faiss.IndexFlatL2(self.dimension)
        
        logger.info(f"FAISS 索引初始化完成: type={self.index_type}, dimension={self.dimension}")
    
    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """添加文档到向量库"""
        if self.index is None:
            logger.warning("FAISS 索引未初始化，使用模拟存储")
            self._add_mock_documents(embeddings, documents, metadata)
            return
        
        try:
            # 确保 embeddings 是 numpy 数组
            embeddings = np.array(embeddings, dtype=np.float32)
            
            # 添加到索引
            self.index.add(embeddings)
            
            # 存储文档和元数据
            self.documents.extend(documents)
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{} for _ in documents])
            
            logger.info(f"添加 {len(documents)} 个文档到向量库")
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise
    
    def _add_mock_documents(self, embeddings: np.ndarray, documents: List[str], metadata: List[Dict[str, Any]]):
        """模拟添加文档"""
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in documents])
        logger.info(f"模拟添加 {len(documents)} 个文档到向量库")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """向量搜索"""
        if self.index is None:
            return self._search_mock(query_embedding, k)
        
        try:
            query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            
            # 搜索
            distances, indices = self.index.search(query_embedding, k)
            
            # 构建结果
            results = []
            for i in range(k):
                idx = indices[0][i]
                if idx < len(self.documents):
                    results.append({
                        "document": self.documents[idx],
                        "score": float(1 / (1 + distances[0][i])),  # 转换为相似度分数
                        "distance": float(distances[0][i]),
                        "metadata": self.metadata[idx] if idx < len(self.metadata) else {}
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def _search_mock(self, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """模拟搜索"""
        results = []
        for i, doc in enumerate(self.documents[:k]):
            results.append({
                "document": doc,
                "score": 0.8 - i * 0.1,
                "distance": 0.2 + i * 0.1,
                "metadata": self.metadata[i] if i < len(self.metadata) else {}
            })
        return results
    
    def save(self, path: str):
        """保存向量库"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            data = {
                "documents": self.documents,
                "metadata": self.metadata,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "saved_at": datetime.now().isoformat()
            }
            
            # 保存数据
            with open(f"{path}.data", "wb") as f:
                pickle.dump(data, f)
            
            # 保存 FAISS 索引
            if self.index is not None:
                import faiss
                faiss.write_index(self.index, f"{path}.index")
            
            logger.info(f"向量库保存到: {path}")
            
        except Exception as e:
            logger.error(f"保存向量库失败: {e}")
            raise
    
    def load(self, path: str):
        """加载向量库"""
        try:
            # 加载数据
            with open(f"{path}.data", "rb") as f:
                data = pickle.load(f)
            
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            self.dimension = data["dimension"]
            self.index_type = data["index_type"]
            
            # 加载 FAISS 索引
            if self.index is not None:
                import faiss
                self.index = faiss.read_index(f"{path}.index")
            
            logger.info(f"向量库加载完成: {path}, 文档数: {len(self.documents)}")
            
        except Exception as e:
            logger.error(f"加载向量库失败: {e}")
            raise
    
    def get_document_count(self) -> int:
        """获取文档数量"""
        return len(self.documents)
    
    def clear(self):
        """清空向量库"""
        self.documents = []
        self.metadata = []
        if self.index is not None:
            self._init_index()
        logger.info("向量库已清空")

class HybridRetriever:
    """混合检索器（向量+关键词）"""
    
    def __init__(self, vector_store: FAISSVectorStore, keyword_weight: float = 0.3):
        self.vector_store = vector_store
        self.keyword_weight = keyword_weight
        self.vector_weight = 1.0 - keyword_weight
    
    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int = 5,
        enable_keyword: bool = True
    ) -> List[Dict[str, Any]]:
        """混合检索"""
        # 向量检索
        vector_results = self.vector_store.search(query_embedding, k * 2)
        
        if not enable_keyword:
            # 只使用向量检索
            return vector_results[:k]
        
        # 关键词检索
        keyword_results = self._keyword_search(query, k * 2)
        
        # 融合结果
        fused_results = self._fuse_results(vector_results, keyword_results, k)
        
        return fused_results
    
    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """关键词搜索"""
        query_terms = query.lower().split()
        results = []
        
        for i, doc in enumerate(self.vector_store.documents):
            doc_lower = doc.lower()
            score = 0.0
            
            # 计算词频
            for term in query_terms:
                if term in doc_lower:
                    score += doc_lower.count(term) / len(doc_lower)
            
            if score > 0:
                results.append({
                    "document": doc,
                    "score": min(score * 10, 1.0),  # 归一化
                    "metadata": self.vector_store.metadata[i] if i < len(self.vector_store.metadata) else {}
                })
        
        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:k]
    
    def _fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        k: int
    ) -> List[Dict[str, Any]]:
        """融合检索结果"""
        # 创建文档映射
        doc_scores = {}
        
        # 添加向量检索分数
        for result in vector_results:
            doc = result["document"]
            doc_scores[doc] = {
                "vector_score": result["score"],
                "keyword_score": 0.0,
                "metadata": result["metadata"]
            }
        
        # 添加关键词检索分数
        for result in keyword_results:
            doc = result["document"]
            if doc in doc_scores:
                doc_scores[doc]["keyword_score"] = result["score"]
            else:
                doc_scores[doc] = {
                    "vector_score": 0.0,
                    "keyword_score": result["score"],
                    "metadata": result["metadata"]
                }
        
        # 计算融合分数
        fused_results = []
        for doc, scores in doc_scores.items():
            fused_score = (
                self.vector_weight * scores["vector_score"] +
                self.keyword_weight * scores["keyword_score"]
            )
            fused_results.append({
                "document": doc,
                "score": fused_score,
                "vector_score": scores["vector_score"],
                "keyword_score": scores["keyword_score"],
                "metadata": scores["metadata"]
            })
        
        # 按融合分数排序
        fused_results.sort(key=lambda x: x["score"], reverse=True)
        
        return fused_results[:k]

class ReRanker:
    """重排序器"""
    
    def __init__(self):
        pass
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """重排序结果"""
        if not results:
            return results
        
        # 计算额外特征
        for result in results:
            doc = result["document"]
            
            # 特征1: 查询词在文档中的位置（越靠前越好）
            query_terms = query.lower().split()
            doc_lower = doc.lower()
            first_match_pos = len(doc)
            for term in query_terms:
                pos = doc_lower.find(term)
                if pos != -1 and pos < first_match_pos:
                    first_match_pos = pos
            
            position_score = 1.0 / (1 + first_match_pos / 100)
            
            # 特征2: 文档长度（适中的长度更好）
            doc_length = len(doc)
            length_score = 1.0 / (1 + abs(doc_length - 500) / 500)
            
            # 特征3: 查询词覆盖率
            matched_terms = sum(1 for term in query_terms if term in doc_lower)
            coverage_score = matched_terms / len(query_terms) if query_terms else 0
            
            # 综合分数
            result["rerank_score"] = (
                0.6 * result["score"] +
                0.2 * position_score +
                0.1 * length_score +
                0.1 * coverage_score
            )
        
        # 按重排序分数排序
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return results[:top_k]

# 全局实例
def create_vector_store(dimension: int = 768, index_type: str = "flat") -> FAISSVectorStore:
    """创建向量存储"""
    return FAISSVectorStore(dimension, index_type)

def create_hybrid_retriever(
    vector_store: FAISSVectorStore,
    keyword_weight: float = 0.3
) -> HybridRetriever:
    """创建混合检索器"""
    return HybridRetriever(vector_store, keyword_weight)

def create_reranker() -> ReRanker:
    """创建重排序器"""
    return ReRanker()