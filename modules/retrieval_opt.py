"""
Module 05 & 07: 检索前处理(PreRetrieval) & 检索后处理(PostRetrieval)
- 检索优化: Query Expansion (查询扩展)
- 检索结果优化: 重排序, 过滤
技术栈: NLP工具, ML模型
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class QueryExpander:
    """
    Module 05: 检索前处理 - 查询扩展
    功能: 扩展用户查询以提高检索召回率
    """
    
    def __init__(self, expansion_method: str = "synonym"):
        """
        初始化查询扩展器
        
        Args:
            expansion_method: 扩展方法 ("synonym", "llm", "embedding")
        """
        self.expansion_method = expansion_method
        self.synonym_dict = self._load_synonyms()
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """加载同义词词典"""
        # SQL相关的同义词映射
        return {
            "查询": ["搜索", "获取", "检索", "查找", "SELECT"],
            "用户": ["客户", "会员", "账户", "user"],
            "订单": ["交易", "购买记录", "order"],
            "统计": ["计算", "汇总", "聚合", "COUNT", "SUM"],
            "平均": ["均值", "AVG"],
            "最大": ["最高", "MAX"],
            "最小": ["最低", "MIN"],
            "删除": ["移除", "清除", "DELETE"],
            "更新": ["修改", "UPDATE"],
            "插入": ["添加", "新增", "INSERT"],
        }
    
    def expand_query(self, query: str) -> List[str]:
        """
        扩展查询
        
        Args:
            query: 原始查询
            
        Returns:
            扩展后的查询列表
        """
        expanded_queries = [query]
        
        if self.expansion_method == "synonym":
            expanded_queries.extend(self._synonym_expansion(query))
        elif self.expansion_method == "decomposition":
            expanded_queries.extend(self._query_decomposition(query))
        
        return list(set(expanded_queries))
    
    def _synonym_expansion(self, query: str) -> List[str]:
        """基于同义词的查询扩展"""
        expanded = []
        for word, synonyms in self.synonym_dict.items():
            if word in query:
                for syn in synonyms[:2]:  # 限制扩展数量
                    expanded.append(query.replace(word, syn))
        return expanded
    
    def _query_decomposition(self, query: str) -> List[str]:
        """查询分解 - 将复杂查询分解为简单子查询"""
        # 简单的分解逻辑，实际可以使用LLM
        sub_queries = []
        if "和" in query or "并且" in query:
            parts = query.replace("并且", "和").split("和")
            sub_queries.extend([p.strip() for p in parts if p.strip()])
        return sub_queries


class ResultReranker:
    """
    Module 07: 检索后处理 - 结果重排序
    功能: 对检索结果进行重排序和过滤
    """
    
    def __init__(self, rerank_method: str = "cross_encoder"):
        """
        初始化重排序器
        
        Args:
            rerank_method: 重排序方法 ("cross_encoder", "similarity", "hybrid")
        """
        self.rerank_method = rerank_method
        self.cross_encoder = None
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        重排序检索结果
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
            top_k: 返回前k个结果
            
        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []
        
        if self.rerank_method == "similarity":
            return self._similarity_rerank(query, documents, top_k)
        elif self.rerank_method == "cross_encoder":
            return self._cross_encoder_rerank(query, documents, top_k)
        else:
            return self._hybrid_rerank(query, documents, top_k)
    
    def _similarity_rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """基于相似度的重排序"""
        # 使用现有的相似度分数进行排序
        sorted_docs = sorted(
            documents, 
            key=lambda x: x.get('score', 0), 
            reverse=True
        )
        return sorted_docs[:top_k]
    
    def _cross_encoder_rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        基于Cross-Encoder的重排序
        Cross-Encoder可以更准确地计算query-document相关性
        """
        try:
            from sentence_transformers import CrossEncoder
            
            if self.cross_encoder is None:
                # 使用多语言cross-encoder模型
                self.cross_encoder = CrossEncoder(
                    'cross-encoder/ms-marco-MiniLM-L-6-v2',
                    max_length=512
                )
            
            # 准备输入对
            pairs = [(query, doc.get('content', '')) for doc in documents]
            
            # 计算分数
            scores = self.cross_encoder.predict(pairs)
            
            # 添加重排序分数并排序
            for doc, score in zip(documents, scores):
                doc['rerank_score'] = float(score)
            
            sorted_docs = sorted(
                documents, 
                key=lambda x: x.get('rerank_score', 0), 
                reverse=True
            )
            return sorted_docs[:top_k]
            
        except ImportError:
            # 如果没有安装，回退到相似度重排序
            return self._similarity_rerank(query, documents, top_k)
    
    def _hybrid_rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """混合重排序 - 结合多种信号"""
        for doc in documents:
            original_score = doc.get('score', 0)
            # 添加长度惩罚（太短或太长的答案可能质量不高）
            content = doc.get('content', '')
            length_score = min(len(content) / 500, 1.0)  # 归一化
            
            # 关键词匹配加分
            keyword_score = self._keyword_match_score(query, content)
            
            # 综合分数
            doc['hybrid_score'] = (
                0.5 * original_score + 
                0.2 * length_score + 
                0.3 * keyword_score
            )
        
        sorted_docs = sorted(
            documents, 
            key=lambda x: x.get('hybrid_score', 0), 
            reverse=True
        )
        return sorted_docs[:top_k]
    
    def _keyword_match_score(self, query: str, content: str) -> float:
        """计算关键词匹配分数"""
        query_words = set(query.lower().split())
        content_lower = content.lower()
        matches = sum(1 for word in query_words if word in content_lower)
        return matches / len(query_words) if query_words else 0


class ResultFilter:
    """
    Module 07: 检索后处理 - 结果过滤
    功能: 根据各种条件过滤检索结果
    """
    
    def __init__(self):
        self.filters = []
    
    def add_filter(self, filter_func):
        """添加过滤函数"""
        self.filters.append(filter_func)
    
    def filter_results(
        self, 
        documents: List[Dict[str, Any]], 
        min_score: float = 0.0,
        max_length: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        过滤检索结果
        
        Args:
            documents: 文档列表
            min_score: 最低分数阈值
            max_length: 最大内容长度
            
        Returns:
            过滤后的文档列表
        """
        filtered = []
        
        for doc in documents:
            # 分数过滤
            if doc.get('score', 0) < min_score:
                continue
            
            # 长度过滤
            content = doc.get('content', '')
            if max_length and len(content) > max_length:
                continue
            
            # 应用自定义过滤器
            passed = True
            for filter_func in self.filters:
                if not filter_func(doc):
                    passed = False
                    break
            
            if passed:
                filtered.append(doc)
        
        return filtered
    
    def deduplicate(
        self, 
        documents: List[Dict[str, Any]], 
        similarity_threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """
        去除重复或高度相似的文档
        
        Args:
            documents: 文档列表
            similarity_threshold: 相似度阈值
            
        Returns:
            去重后的文档列表
        """
        if len(documents) <= 1:
            return documents
        
        unique_docs = [documents[0]]
        
        for doc in documents[1:]:
            is_duplicate = False
            for unique_doc in unique_docs:
                # 简单的文本相似度检查
                similarity = self._text_similarity(
                    doc.get('content', ''), 
                    unique_doc.get('content', '')
                )
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_docs.append(doc)
        
        return unique_docs
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度（基于Jaccard）"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
