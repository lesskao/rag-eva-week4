"""
Module 06: 索引优化(Indexing)
- 索引构建和优化
- 层次索引, 关键词索引
技术栈: 搜索引擎 (Whoosh)
"""

from typing import List, Dict, Any, Optional
import os
import json
from pathlib import Path


class IndexBuilder:
    """
    Module 06: 索引构建器
    功能: 构建和管理多种类型的索引
    """
    
    def __init__(self, index_dir: str = "./vector_db/indexes"):
        """
        初始化索引构建器
        
        Args:
            index_dir: 索引存储目录
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
    
    def build_hierarchical_index(
        self, 
        documents: List[Dict[str, Any]],
        hierarchy_field: str = "category"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        构建层次索引
        
        Args:
            documents: 文档列表
            hierarchy_field: 用于分层的字段名
            
        Returns:
            层次化的文档索引
        """
        hierarchical_index = {}
        
        for doc in documents:
            category = doc.get(hierarchy_field, "default")
            if category not in hierarchical_index:
                hierarchical_index[category] = []
            hierarchical_index[category].append(doc)
        
        # 保存索引
        index_path = self.index_dir / "hierarchical_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(hierarchical_index, f, ensure_ascii=False, indent=2)
        
        return hierarchical_index
    
    def build_inverted_index(
        self, 
        documents: List[Dict[str, Any]],
        content_field: str = "content"
    ) -> Dict[str, List[int]]:
        """
        构建倒排索引
        
        Args:
            documents: 文档列表
            content_field: 内容字段名
            
        Returns:
            倒排索引 (词 -> 文档ID列表)
        """
        inverted_index = {}
        
        for doc_id, doc in enumerate(documents):
            content = doc.get(content_field, "")
            # 简单分词
            words = self._tokenize(content)
            
            for word in words:
                if word not in inverted_index:
                    inverted_index[word] = []
                if doc_id not in inverted_index[word]:
                    inverted_index[word].append(doc_id)
        
        # 保存索引
        index_path = self.index_dir / "inverted_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(inverted_index, f, ensure_ascii=False, indent=2)
        
        return inverted_index
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        # 移除标点，转小写，分词
        import re
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [w for w in text.split() if len(w) > 1]
    
    def load_index(self, index_type: str) -> Optional[Dict]:
        """
        加载已保存的索引
        
        Args:
            index_type: 索引类型 ("hierarchical", "inverted")
            
        Returns:
            索引字典，如果不存在返回None
        """
        index_path = self.index_dir / f"{index_type}_index.json"
        if index_path.exists():
            with open(index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None


class KeywordIndex:
    """
    Module 06: 关键词索引
    功能: 基于Whoosh的全文搜索索引
    """
    
    def __init__(self, index_dir: str = "./vector_db/whoosh_index"):
        """
        初始化关键词索引
        
        Args:
            index_dir: Whoosh索引存储目录
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.ix = None
        self._init_index()
    
    def _init_index(self):
        """初始化Whoosh索引"""
        try:
            from whoosh.index import create_in, open_dir, exists_in
            from whoosh.fields import Schema, TEXT, ID, STORED
            
            self.schema = Schema(
                id=ID(stored=True),
                content=TEXT(stored=True),
                question=TEXT(stored=True),
                sql=STORED,
                metadata=STORED
            )
            
            if exists_in(str(self.index_dir)):
                self.ix = open_dir(str(self.index_dir))
            else:
                self.ix = create_in(str(self.index_dir), self.schema)
                
        except ImportError:
            print("Warning: Whoosh not installed. Keyword search will be limited.")
            self.ix = None
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        添加文档到索引
        
        Args:
            documents: 文档列表
        """
        if self.ix is None:
            return
        
        from whoosh.index import open_dir
        
        writer = self.ix.writer()
        
        for doc in documents:
            writer.add_document(
                id=str(doc.get('id', '')),
                content=doc.get('context', ''),
                question=doc.get('question', ''),
                sql=doc.get('sql', ''),
                metadata=json.dumps(doc)
            )
        
        writer.commit()
    
    def search(
        self, 
        query: str, 
        field: str = "content",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        搜索文档
        
        Args:
            query: 搜索查询
            field: 搜索字段
            limit: 返回结果数量限制
            
        Returns:
            匹配的文档列表
        """
        if self.ix is None:
            return []
        
        from whoosh.qparser import QueryParser
        
        results = []
        
        with self.ix.searcher() as searcher:
            parser = QueryParser(field, self.ix.schema)
            q = parser.parse(query)
            hits = searcher.search(q, limit=limit)
            
            for hit in hits:
                result = {
                    'id': hit['id'],
                    'content': hit.get('content', ''),
                    'question': hit.get('question', ''),
                    'sql': hit.get('sql', ''),
                    'score': hit.score
                }
                if hit.get('metadata'):
                    try:
                        result['metadata'] = json.loads(hit['metadata'])
                    except:
                        pass
                results.append(result)
        
        return results
    
    def hybrid_search(
        self, 
        query: str, 
        vector_results: List[Dict[str, Any]],
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        混合搜索 - 结合向量检索和关键词检索
        
        Args:
            query: 搜索查询
            vector_results: 向量检索结果
            keyword_weight: 关键词搜索权重
            
        Returns:
            融合后的结果列表
        """
        # 获取关键词搜索结果
        keyword_results = self.search(query, limit=len(vector_results) * 2)
        
        # 创建ID到结果的映射
        result_map = {}
        
        # 处理向量结果
        for rank, doc in enumerate(vector_results):
            doc_id = str(doc.get('id', rank))
            vector_score = doc.get('score', 1.0 / (rank + 1))
            result_map[doc_id] = {
                **doc,
                'vector_score': vector_score,
                'keyword_score': 0,
                'final_score': vector_score * (1 - keyword_weight)
            }
        
        # 处理关键词结果
        for rank, doc in enumerate(keyword_results):
            doc_id = str(doc.get('id', ''))
            keyword_score = doc.get('score', 1.0 / (rank + 1))
            
            if doc_id in result_map:
                result_map[doc_id]['keyword_score'] = keyword_score
                result_map[doc_id]['final_score'] += keyword_score * keyword_weight
            else:
                result_map[doc_id] = {
                    **doc,
                    'vector_score': 0,
                    'keyword_score': keyword_score,
                    'final_score': keyword_score * keyword_weight
                }
        
        # 按最终分数排序
        sorted_results = sorted(
            result_map.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )
        
        return sorted_results


class IndexOptimizer:
    """
    Module 06: 索引优化器
    功能: 优化索引性能和质量
    """
    
    def __init__(self):
        pass
    
    def optimize_chunk_overlap(
        self, 
        documents: List[str], 
        chunk_size: int = 500,
        overlap_ratio: float = 0.2
    ) -> List[str]:
        """
        优化文档分块的重叠策略
        
        Args:
            documents: 原始文档列表
            chunk_size: 分块大小
            overlap_ratio: 重叠比例
            
        Returns:
            优化后的分块列表
        """
        chunks = []
        overlap = int(chunk_size * overlap_ratio)
        
        for doc in documents:
            start = 0
            while start < len(doc):
                end = min(start + chunk_size, len(doc))
                chunk = doc[start:end]
                
                # 尝试在句子边界切分
                if end < len(doc):
                    # 寻找最近的句子结束符
                    for sep in ['。', '！', '？', '.', '!', '?', '\n']:
                        last_sep = chunk.rfind(sep)
                        if last_sep > chunk_size * 0.5:  # 至少保留一半
                            chunk = chunk[:last_sep + 1]
                            end = start + last_sep + 1
                            break
                
                chunks.append(chunk)
                start = end - overlap if end < len(doc) else end
        
        return chunks
    
    def build_summary_index(
        self, 
        documents: List[Dict[str, Any]],
        summarizer=None
    ) -> Dict[str, str]:
        """
        构建摘要索引 - 为每个文档生成摘要用于快速检索
        
        Args:
            documents: 文档列表
            summarizer: 摘要生成器（可选，可以是LLM）
            
        Returns:
            文档ID到摘要的映射
        """
        summary_index = {}
        
        for doc in documents:
            doc_id = str(doc.get('id', ''))
            content = doc.get('content', doc.get('context', ''))
            
            if summarizer:
                # 使用提供的摘要生成器
                summary = summarizer(content)
            else:
                # 简单提取前N个字符作为摘要
                summary = content[:200] + "..." if len(content) > 200 else content
            
            summary_index[doc_id] = summary
        
        return summary_index
