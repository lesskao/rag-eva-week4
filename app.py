"""
Streamlit 主程序 - 展示层
负责成果展示与交互
"""
import streamlit as st
import json
from pathlib import Path

# 页面配置
st.set_page_config(
    page_title="RAG 系统评估平台",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .result-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sql-code {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Consolas', monospace;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_engine():
    """加载并初始化RAG引擎"""
    from rag_engine import RAGEngine, RAGConfig
    config = RAGConfig()
    engine = RAGEngine(config)
    engine.initialize()
    return engine


@st.cache_data
def load_test_data():
    """加载测试数据"""
    data_path = Path("./data/q2sql_pairs.json")
    if data_path.exists():
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def render_sidebar():
    """渲染侧边栏"""
    st.sidebar.markdown("## 🎛️ 系统配置")
    
    st.sidebar.markdown("### 检索设置")
    top_k = st.sidebar.slider("返回结果数 (Top-K)", 1, 10, 5)
    enable_expansion = st.sidebar.checkbox("启用查询扩展", value=True)
    enable_rerank = st.sidebar.checkbox("启用重排序", value=True)
    
    st.sidebar.markdown("### 模型设置")
    embedding_model = st.sidebar.selectbox(
        "Embedding模型",
        ["BAAI/bge-small-zh-v1.5", "BAAI/bge-base-zh-v1.5", "sentence-transformers/all-MiniLM-L6-v2"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 系统状态")
    
    return {
        'top_k': top_k,
        'enable_expansion': enable_expansion,
        'enable_rerank': enable_rerank,
        'embedding_model': embedding_model
    }


def render_query_tab(engine, config):
    """渲染查询标签页"""
    st.markdown("### 🔍 智能SQL查询")
    
    query = st.text_input(
        "输入您的问题",
        placeholder="例如：查询所有用户的信息",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_btn = st.button("🚀 搜索", type="primary")
    
    if search_btn and query:
        with st.spinner("正在检索..."):
            result = engine.query(query, top_k=config['top_k'])
        
        st.markdown("---")
        st.markdown("### 📋 查询结果")
        
        # 显示答案
        st.markdown(result.get('answer', ''))
        
        # 显示检索详情
        with st.expander("🔎 检索详情", expanded=True):
            docs = result.get('retrieved_documents', [])
            for i, doc in enumerate(docs):
                st.markdown(f"""
                <div class="result-box">
                    <strong>结果 {i+1}</strong> | 相似度: {doc.get('score', 0):.2%}<br>
                    <strong>问题:</strong> {doc.get('question', 'N/A')}<br>
                    <strong>SQL:</strong> <code>{doc.get('sql', 'N/A')}</code>
                </div>
                """, unsafe_allow_html=True)
        
        # 显示扩展查询
        if config['enable_expansion']:
            queries = result.get('expanded_queries', [])
            if len(queries) > 1:
                with st.expander("🔄 扩展查询"):
                    for q in queries:
                        st.write(f"- {q}")


def render_evaluation_tab(engine):
    """渲染评估标签页"""
    st.markdown("### 📊 系统评估")
    
    test_data = load_test_data()
    
    if not test_data:
        st.warning("未找到测试数据，请确保 data/q2sql_pairs.json 存在")
        return
    
    st.info(f"📁 已加载 {len(test_data)} 条测试数据")
    
    if st.button("🧪 运行评估", type="primary"):
        from evaluator import RAGEvaluator
        
        evaluator = RAGEvaluator(engine)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        for i, item in enumerate(test_data):
            status_text.text(f"正在评估 {i+1}/{len(test_data)}...")
            progress_bar.progress((i + 1) / len(test_data))
            
            rag_result = engine.query(item['question'])
            
            # 从检索结果中获取Top-1的SQL，而不是从答案文本中提取
            retrieved_docs = rag_result.get('retrieved_documents', [])
            if retrieved_docs:
                # 使用检索到的最相似文档的SQL
                predicted_sql = retrieved_docs[0].get('sql', '')
            else:
                predicted_sql = ''
            
            sql_result = evaluator.sql_evaluator.evaluate_single(
                question=item['question'],
                predicted=predicted_sql,
                ground_truth=item['sql'],
                context=item.get('context', '')
            )
            results.append(sql_result)
        
        progress_bar.empty()
        status_text.empty()
        
        # 显示评估结果
        st.markdown("---")
        st.markdown("### 📈 评估结果")
        
        col1, col2, col3, col4 = st.columns(4)
        
        exact_match_rate = sum(1 for r in results if r.exact_match) / len(results)
        avg_similarity = sum(r.similarity_score for r in results) / len(results)
        avg_overall = sum(r.overall_score for r in results) / len(results)
        
        with col1:
            st.metric("样本数", len(results))
        with col2:
            st.metric("精确匹配率", f"{exact_match_rate:.1%}")
        with col3:
            st.metric("平均相似度", f"{avg_similarity:.1%}")
        with col4:
            st.metric("整体得分", f"{avg_overall:.1%}")
        
        # 详细结果表格
        st.markdown("### 📋 详细结果")
        import pandas as pd
        df = pd.DataFrame([{
            '问题': r.question[:50] + '...' if len(r.question) > 50 else r.question,
            '精确匹配': '✅' if r.exact_match else '❌',
            '相似度': f"{r.similarity_score:.2%}",
            '整体得分': f"{r.overall_score:.2%}"
        } for r in results])
        st.dataframe(df, use_container_width=True)


def render_data_tab():
    """渲染数据管理标签页"""
    st.markdown("### 📂 数据管理")
    
    test_data = load_test_data()
    
    if test_data:
        st.success(f"✅ 已加载 {len(test_data)} 条数据")
        
        import pandas as pd
        df = pd.DataFrame(test_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("未找到数据文件")


def main():
    """主函数"""
    st.markdown('<h1 class="main-header">🔍 RAG 系统评估平台</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">基于 LangChain + ChromaDB 的智能检索增强生成系统</p>', unsafe_allow_html=True)
    
    # 侧边栏配置
    config = render_sidebar()
    
    # 初始化RAG引擎
    try:
        engine = load_rag_engine()
        status = engine.get_status()
        st.sidebar.success(f"✅ 向量库: {status['vector_store'].get('count', 0)} 条")
    except Exception as e:
        st.sidebar.error(f"❌ 初始化失败: {str(e)}")
        engine = None
    
    # 主要标签页
    tab1, tab2, tab3 = st.tabs(["🔍 查询", "📊 评估", "📂 数据"])
    
    with tab1:
        if engine:
            render_query_tab(engine, config)
        else:
            st.error("RAG引擎未初始化")
    
    with tab2:
        if engine:
            render_evaluation_tab(engine)
        else:
            st.error("RAG引擎未初始化")
    
    with tab3:
        render_data_tab()


if __name__ == "__main__":
    main()
