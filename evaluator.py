"""
Evaluator - 系统性能评估模块
封装 Module 09: 基于 JSON 的自动评分逻辑
技术栈: RAGAS, TruLens
"""
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import difflib


@dataclass
class EvaluationResult:
    """评估结果数据类"""
    question: str
    predicted_sql: str
    ground_truth_sql: str
    exact_match: bool
    similarity_score: float
    context_relevance: float
    answer_faithfulness: float
    overall_score: float


class SQLEvaluator:
    """SQL查询评估器"""
    
    def __init__(self):
        self.results = []
    
    def normalize_sql(self, sql: str) -> str:
        """标准化SQL以便比较"""
        sql = sql.upper().strip()
        sql = ' '.join(sql.split())
        for kw in ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'JOIN', 'ON', 'AND', 'OR']:
            sql = sql.replace(f' {kw} ', f'\n{kw} ')
        return sql
    
    def exact_match(self, pred: str, truth: str) -> bool:
        """精确匹配检查"""
        return self.normalize_sql(pred) == self.normalize_sql(truth)
    
    def similarity_score(self, pred: str, truth: str) -> float:
        """计算SQL相似度分数"""
        pred_norm = self.normalize_sql(pred)
        truth_norm = self.normalize_sql(truth)
        return difflib.SequenceMatcher(None, pred_norm, truth_norm).ratio()
    
    def evaluate_single(self, question: str, predicted: str, ground_truth: str, context: str = "") -> EvaluationResult:
        """评估单个查询"""
        exact = self.exact_match(predicted, ground_truth)
        sim = self.similarity_score(predicted, ground_truth)
        context_rel = self._context_relevance(question, context)
        faithfulness = self._answer_faithfulness(predicted, ground_truth, context)
        overall = 0.3 * (1.0 if exact else 0.0) + 0.3 * sim + 0.2 * context_rel + 0.2 * faithfulness
        result = EvaluationResult(
            question=question, predicted_sql=predicted, ground_truth_sql=ground_truth,
            exact_match=exact, similarity_score=sim, context_relevance=context_rel,
            answer_faithfulness=faithfulness, overall_score=overall
        )
        self.results.append(result)
        return result
    
    def _context_relevance(self, question: str, context: str) -> float:
        if not context:
            return 0.5
        q_words = set(question.lower().split())
        c_words = set(context.lower().split())
        overlap = len(q_words & c_words)
        return min(overlap / max(len(q_words), 1), 1.0)
    
    def _answer_faithfulness(self, pred: str, truth: str, context: str) -> float:
        pred_words = set(pred.upper().split())
        truth_words = set(truth.upper().split())
        if not truth_words:
            return 0.0
        return len(pred_words & truth_words) / len(truth_words)
    
    def evaluate_batch(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
        """批量评估"""
        results = []
        truth_map = {str(gt.get('id')): gt for gt in ground_truths}
        for pred in predictions:
            pred_id = str(pred.get('id', ''))
            if pred_id in truth_map:
                gt = truth_map[pred_id]
                result = self.evaluate_single(
                    question=pred.get('question', gt.get('question', '')),
                    predicted=pred.get('predicted_sql', pred.get('sql', '')),
                    ground_truth=gt.get('sql', ''),
                    context=gt.get('context', '')
                )
                results.append(result)
        return self._aggregate_results(results)
    
    def _aggregate_results(self, results: List[EvaluationResult]) -> Dict:
        if not results:
            return {'count': 0, 'exact_match_rate': 0, 'avg_similarity': 0, 'avg_overall': 0}
        n = len(results)
        return {
            'count': n,
            'exact_match_rate': sum(1 for r in results if r.exact_match) / n,
            'avg_similarity': sum(r.similarity_score for r in results) / n,
            'avg_context_relevance': sum(r.context_relevance for r in results) / n,
            'avg_faithfulness': sum(r.answer_faithfulness for r in results) / n,
            'avg_overall': sum(r.overall_score for r in results) / n,
            'detailed_results': [{'question': r.question, 'exact_match': r.exact_match, 'similarity': r.similarity_score, 'overall': r.overall_score} for r in results]
        }


class RAGEvaluator:
    """RAG系统综合评估器"""
    
    def __init__(self, rag_engine=None):
        self.rag_engine = rag_engine
        self.sql_evaluator = SQLEvaluator()
    
    def evaluate_retrieval(self, query: str, expected_ids: List[str], retrieved_docs: List[Dict]) -> Dict:
        """评估检索质量"""
        retrieved_ids = set(str(doc.get('id', '')) for doc in retrieved_docs)
        expected_set = set(expected_ids)
        hits = len(retrieved_ids & expected_set)
        precision = hits / len(retrieved_ids) if retrieved_ids else 0
        recall = hits / len(expected_set) if expected_set else 0
        mrr = 0.0
        for i, doc in enumerate(retrieved_docs):
            if str(doc.get('id', '')) in expected_set:
                mrr = 1.0 / (i + 1)
                break
        return {'precision': precision, 'recall': recall, 'mrr': mrr, 'hits': hits, 'total_expected': len(expected_set), 'total_retrieved': len(retrieved_ids)}
    
    def evaluate_end_to_end(self, test_data: List[Dict]) -> Dict:
        """端到端评估"""
        if not self.rag_engine:
            return {'error': 'RAG engine not initialized'}
        results = []
        for item in test_data:
            question = item.get('question', '')
            expected_sql = item.get('sql', '')
            rag_result = self.rag_engine.query(question)
            predicted_sql = self._extract_sql_from_answer(rag_result.get('answer', ''))
            eval_result = self.sql_evaluator.evaluate_single(question, predicted_sql, expected_sql, item.get('context', ''))
            retrieval_result = self.evaluate_retrieval(question, [str(item.get('id', ''))], rag_result.get('retrieved_documents', []))
            results.append({'sql_evaluation': eval_result, 'retrieval_evaluation': retrieval_result, 'question': question})
        return self._aggregate_e2e_results(results)
    
    def _extract_sql_from_answer(self, answer: str) -> str:
        if '```sql' in answer:
            parts = answer.split('```sql')
            if len(parts) > 1:
                return parts[1].split('```')[0].strip()
        if 'SELECT' in answer.upper():
            lines = answer.split('\n')
            for line in lines:
                if 'SELECT' in line.upper():
                    return line.strip()
        return answer.strip()
    
    def _aggregate_e2e_results(self, results: List[Dict]) -> Dict:
        if not results:
            return {'count': 0}
        sql_scores = [r['sql_evaluation'].overall_score for r in results]
        retrieval_scores = [r['retrieval_evaluation']['mrr'] for r in results]
        return {
            'count': len(results),
            'avg_sql_score': sum(sql_scores) / len(sql_scores),
            'avg_retrieval_mrr': sum(retrieval_scores) / len(retrieval_scores),
            'exact_match_rate': sum(1 for r in results if r['sql_evaluation'].exact_match) / len(results)
        }
    
    def generate_report(self, evaluation_results: Dict) -> str:
        """生成评估报告"""
        report = "# RAG系统评估报告\n\n"
        report += f"## 概览\n- 评估样本数: {evaluation_results.get('count', 0)}\n"
        report += f"- 精确匹配率: {evaluation_results.get('exact_match_rate', 0):.2%}\n"
        report += f"- 平均相似度: {evaluation_results.get('avg_similarity', 0):.2%}\n"
        report += f"- 平均整体得分: {evaluation_results.get('avg_overall', 0):.2%}\n"
        return report


def load_ground_truth(file_path: str) -> List[Dict]:
    """加载Ground Truth数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_evaluation(rag_engine, test_data_path: str = "./data/q2sql_pairs.json") -> Dict:
    """运行完整评估"""
    test_data = load_ground_truth(test_data_path)
    evaluator = RAGEvaluator(rag_engine)
    return evaluator.evaluate_end_to_end(test_data)
