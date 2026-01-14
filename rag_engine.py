"""
RAG Engine - 核心 RAG 流水线实现
封装 Module 01-08 的完整 RAG 系统
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# 加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


@dataclass
class RAGConfig:
    data_path: str = "./data/q2sql_pairs.json"
    vector_db_path: str = "./vector_db"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    top_k: int = 5
    enable_query_expansion: bool = True
    enable_rerank: bool = True
    # DeepSeek LLM 配置
    llm_provider: str = "deepseek"  # deepseek 或 openai
    llm_model: str = "deepseek-chat"
    llm_api_key: str = None  # 优先使用环境变量 DEEPSEEK_API_KEY
    llm_base_url: str = "https://api.deepseek.com"
    temperature: float = 0.7
    use_llm: bool = True  # 是否启用LLM生成


class DataLoader:
    """Module 01: 数据加载和预处理"""
    def __init__(self, config: RAGConfig):
        self.config = config
    
    def load_json(self, file_path: str = None) -> List[Dict]:
        path = file_path or self.config.data_path
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def preprocess(self, data: List[Dict]) -> List[Dict]:
        processed = []
        for item in data:
            doc_content = f"问题: {item.get('question', '')}\n"
            doc_content += f"上下文: {item.get('context', '')}\n"
            doc_content += f"SQL: {item.get('sql', '')}"
            processed.append({
                'id': item.get('id'),
                'content': doc_content,
                'question': item.get('question', ''),
                'sql': item.get('sql', ''),
                'context': item.get('context', ''),
                'difficulty': item.get('difficulty', 'medium'),
                'metadata': item
            })
        return processed


class DocumentChunker:
    """Module 02: 文档分块策略"""
    def __init__(self, config: RAGConfig):
        self.config = config
        self.splitter = None
        if LANGCHAIN_AVAILABLE:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        chunked = []
        for doc in documents:
            content = doc.get('content', '')
            if self.splitter and len(content) > self.config.chunk_size:
                chunks = self.splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    chunked.append({**doc, 'content': chunk, 'chunk_id': i})
            else:
                chunked.append(doc)
        return chunked


class EmbeddingEngine:
    """Module 03: 文本向量化"""
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        if ST_AVAILABLE:
            try:
                self.model = SentenceTransformer(config.embedding_model)
            except:
                pass
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.model:
            return self.model.encode(texts, convert_to_numpy=True).tolist()
        return [[0.0] * 384 for _ in texts]
    
    def embed_query(self, query: str) -> List[float]:
        if self.model:
            return self.model.encode([query], convert_to_numpy=True)[0].tolist()
        return [0.0] * 384


class VectorStore:
    """Module 04: 向量数据库操作"""
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = None
        self.collection = None
        if CHROMA_AVAILABLE:
            db_path = Path(config.vector_db_path) / "chroma"
            db_path.mkdir(parents=True, exist_ok=True)
            try:
                self.client = chromadb.PersistentClient(path=str(db_path))
                self.collection = self.client.get_or_create_collection("rag_docs")
            except:
                pass
    
    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        if not self.collection:
            return
        ids = [str(doc.get('id', i)) for i, doc in enumerate(documents)]
        contents = [doc.get('content', '') for doc in documents]
        metadatas = [{'question': str(doc.get('question', '')), 'sql': str(doc.get('sql', ''))} for doc in documents]
        try:
            self.collection.delete(ids=ids)
        except:
            pass
        self.collection.add(ids=ids, embeddings=embeddings, documents=contents, metadatas=metadatas)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        if not self.collection:
            return []
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k, include=['documents', 'metadatas', 'distances'])
        docs = []
        for i in range(len(results['ids'][0])):
            docs.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'score': 1 - results['distances'][0][i],
                **results['metadatas'][0][i]
            })
        return docs
    
    def get_stats(self) -> Dict:
        if not self.collection:
            return {"count": 0}
        return {"count": self.collection.count()}


class AnswerGenerator:
    """Module 08: 答案生成 - 支持 DeepSeek LLM"""
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = None
        self._init_llm_client()
    
    def _init_llm_client(self):
        """初始化LLM客户端"""
        try:
            from openai import OpenAI
            api_key = self.config.llm_api_key or os.environ.get('DEEPSEEK_API_KEY')
            if api_key:
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=self.config.llm_base_url
                )
        except ImportError:
            pass
    
    def generate(self, query: str, context_docs: List[Dict]) -> str:
        if not context_docs:
            return "抱歉，没有找到相关信息。"
        
        # 如果启用LLM且客户端可用，使用DeepSeek生成
        if self.config.use_llm and self.client:
            return self._llm_generate(query, context_docs)
        
        # 否则使用模板生成
        return self._template_generate(query, context_docs)
    
    def _template_generate(self, query: str, context_docs: List[Dict]) -> str:
        """模板生成（不使用LLM）"""
        best = context_docs[0]
        return f"""根据您的问题："{query}"

找到最匹配的SQL查询：

**原始问题**: {best.get('question', 'N/A')}

**SQL语句**:
```sql
{best.get('sql', 'N/A')}
```

**相似度得分**: {best.get('score', 0):.2%}
"""
    
    def _llm_generate(self, query: str, context_docs: List[Dict]) -> str:
        """使用DeepSeek LLM生成答案"""
        # 构建上下文
        context = "\n\n".join([
            f"参考示例{i+1}:\n问题: {doc.get('question', '')}\nSQL: {doc.get('sql', '')}\n上下文: {doc.get('context', doc.get('metadata', {}).get('context', ''))}"
            for i, doc in enumerate(context_docs[:3])
        ])
        
        prompt = f"""你是一个SQL专家。根据以下参考示例，为用户的问题生成合适的SQL查询。

参考示例:
{context}

用户问题: {query}

请生成SQL查询并解释你的思路。如果参考示例中有非常相似的查询，可以直接参考或修改使用。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的SQL专家，擅长将自然语言问题转换为SQL查询。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            # LLM调用失败时回退到模板生成
            return f"⚠️ LLM调用失败: {str(e)}\n\n" + self._template_generate(query, context_docs)


class RAGEngine:
    """完整的RAG引擎 - 封装所有模块"""
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.data_loader = DataLoader(self.config)
        self.chunker = DocumentChunker(self.config)
        self.embedder = EmbeddingEngine(self.config)
        self.vector_store = VectorStore(self.config)
        self.generator = AnswerGenerator(self.config)
        try:
            from modules.retrieval_opt import QueryExpander, ResultReranker
            self.query_expander = QueryExpander()
            self.reranker = ResultReranker()
        except:
            self.query_expander = None
            self.reranker = None
        self.is_initialized = False
    
    def initialize(self, force_reload: bool = False):
        stats = self.vector_store.get_stats()
        if stats.get('count', 0) > 0 and not force_reload:
            self.is_initialized = True
            return
        raw_data = self.data_loader.load_json()
        processed = self.data_loader.preprocess(raw_data)
        chunked = self.chunker.chunk_documents(processed)
        contents = [doc.get('content', '') for doc in chunked]
        embeddings = self.embedder.embed_texts(contents)
        self.vector_store.add_documents(chunked, embeddings)
        self.is_initialized = True
    
    def query(self, question: str, top_k: int = None) -> Dict:
        if not self.is_initialized:
            self.initialize()
        top_k = top_k or self.config.top_k
        queries = [question]
        if self.query_expander:
            queries = self.query_expander.expand_query(question)
        all_results = []
        for q in queries:
            emb = self.embedder.embed_query(q)
            results = self.vector_store.search(emb, top_k=top_k * 2)
            all_results.extend(results)
        seen = set()
        unique = [r for r in all_results if not (r['id'] in seen or seen.add(r['id']))]
        if self.reranker:
            final = self.reranker.rerank(question, unique, top_k=top_k)
        else:
            final = sorted(unique, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
        answer = self.generator.generate(question, final)
        return {'question': question, 'answer': answer, 'retrieved_documents': final, 'expanded_queries': queries}
    
    def get_status(self) -> Dict:
        return {'initialized': self.is_initialized, 'vector_store': self.vector_store.get_stats()}
