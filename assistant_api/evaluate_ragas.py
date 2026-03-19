"""
Оценка качества RAG системы через RAGAS для assistant_api.
Использует OpenAI API для RAG и для метрик RAGAS.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from datasets import Dataset
from ragas import evaluate

# Правильный импорт для RAGAS 0.4.x - используем классы метрик
try:
    # Новый способ импорта (RAGAS 0.4+)
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._context_precision import ContextPrecision
    faithfulness = Faithfulness
    context_precision = ContextPrecision
except ImportError:
    try:
        # Альтернативный импорт из collections
        from ragas.metrics.collections import faithfulness, context_precision
    except ImportError:
        # Fallback на старый импорт
        from ragas.metrics import faithfulness, context_precision

try:
    from .rag_pipeline import RAGPipeline
    from .paths import DEFAULT_DATA_SOURCES, EVALUATION_CACHE_DB_PATH
except ImportError:
    from rag_pipeline import RAGPipeline
    from paths import DEFAULT_DATA_SOURCES, EVALUATION_CACHE_DB_PATH


# Тестовые вопросы для оценки RAG системы
EVALUATION_QUESTIONS = [
    "Что такое машинное обучение?",
    "Какие основные типы машинного обучения существуют?",
    "Что такое нейронная сеть?",
    "Как работают трансформеры в NLP?",
    "Что такое RAG и как он работает?"
]

# Дополнительные вопросы для тестирования Python источника
PYTHON_EVALUATION_QUESTIONS = [
    "Что такое переменная в Python?",
    "Какие типы данных существуют в Python?",
    "Чем отличаются изменяемые и неизменяемые типы данных?",
    "Что такое функция в Python?",
    "Как работает область видимости в Python?"
]


def prepare_dataset(pipeline: RAGPipeline, questions: list, source_filter: str = None) -> Dataset:
    """
    Подготовка датасета для RAGAS из вопросов.
    
    Args:
        pipeline: RAG pipeline для получения ответов
        questions: список вопросов для оценки
        source_filter: фильтр по источнику (например, "python")
    
    Returns:
        Dataset для RAGAS с полями: question, answer, contexts, ground_truth
    """
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []
    
    filter_info = f" (источник: {source_filter})" if source_filter else ""
    print(f"[*] Получение ответов от RAG системы{filter_info}...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"  {i}/{len(questions)}: {question}")
        
        # Получаем ответ от RAG системы (без использования кеша)
        result = pipeline.query(question, use_cache=False, source_filter=source_filter)
        
        # Формируем данные для RAGAS
        questions_list.append(question)
        answers_list.append(result["answer"])
        
        # Контекст - список текстов из найденных документов
        context_texts = [doc["text"] for doc in result["context_docs"]]
        contexts_list.append(context_texts)
        
        # Ground truth - эталонный ответ (для демонстрации используем часть ответа)
        # В реальном проекте здесь должны быть вручную подготовленные эталонные ответы
        ground_truths_list.append(result["answer"][:100])
        
        # Показываем источники в контексте
        if result.get('context_docs'):
            sources_used = set(doc.get('source', 'unknown') for doc in result['context_docs'])
            print(f"     [+] Ответ получен от OpenAI API (источники: {', '.join(sources_used)})")
        else:
            print(f"     [+] Ответ получен от OpenAI API")
    
    print()
    
    # Создаём датасет для RAGAS
    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def evaluate_rag_system():
    """
    Основная функция оценки RAG-системы через RAGAS.
    
    Процесс:
    1. Инициализация RAG pipeline
    2. Генерация ответов на тестовые вопросы
    3. Подготовка датасета для RAGAS
    4. Запуск оценки метрик
    5. Вывод результатов
    """
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ (API MODE) ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()
    
    # Проверка наличия API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("[ОШИБКА] OPENAI_API_KEY не установлен")
        print("\nУстановите переменную окружения:")
        print("  Windows (PowerShell): $env:OPENAI_API_KEY='your-key'")
        print("  Windows (CMD): set OPENAI_API_KEY=your-key")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key'")
        print("\nИли создайте файл .env в корне проекта с содержимым:")
        print("  OPENAI_API_KEY=your-key-here")
        sys.exit(1)
    
    # Инициализация RAG pipeline
    try:
        print("[*] Инициализация RAG системы (API mode)...\n")
        
        # Определяем источники данных (как в обновлённой версии)
        pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path=EVALUATION_CACHE_DB_PATH,
            data_sources=DEFAULT_DATA_SOURCES,
            model="gpt-4o-mini"
        )
        print("\n[OK] RAG система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        sys.exit(1)
    
    # Показываем статистику системы
    stats = pipeline.get_stats()
    print("📊 Статистика системы:")
    print(f"   Всего документов: {stats['vector_store']['count']}")
    if 'sources' in stats['vector_store']:
        for source, count in stats['vector_store']['sources'].items():
            print(f"   {source}: {count} документов")
    print()
    
    # Комплексная оценка с разными источниками
    dataset_general = prepare_dataset(pipeline, EVALUATION_QUESTIONS)
    dataset_python = prepare_dataset(pipeline, PYTHON_EVALUATION_QUESTIONS, source_filter="python")
    
    print("=" * 70)
    print("ЗАПУСК ОЦЕНКИ RAGAS")
    print("=" * 70)
    
    # Оценка общих вопросов
    print("\n[*] Оценка общих вопросов (все источники)...")
    print("   Метрики: Faithfulness, Context Precision")
    print("   (это займёт 1-2 минуты, так как RAGAS использует OpenAI для оценки)\n")
    
    metrics_to_use = [faithfulness(), context_precision()]
    
    try:
        result_general = evaluate(
            dataset=dataset_general,
            metrics=metrics_to_use
        )
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке общих вопросов: {e}")
        result_general = None
    
    # Оценка Python вопросов с фильтром
    print("\n[*] Оценка Python вопросов (фильтр по источнику 'python')...")
    
    try:
        result_python = evaluate(
            dataset=dataset_python,
            metrics=metrics_to_use
        )
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке Python вопросов: {e}")
        result_python = None
    
    # Вывод результатов
    print_results_comparison(result_general, result_python, EVALUATION_QUESTIONS, PYTHON_EVALUATION_QUESTIONS)


def print_results_comparison(result_general, result_python, questions_general, questions_python):
    """Сравнительный вывод результатов оценки."""
    import math
    
    print("\n" + "=" * 70)
    print("СРАВНИТЕЛЬНЫЕ РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)
    
    # Функция для вычисления средних значений
    def calculate_averages(result):
        if result is None:
            return 0, 0, 0
        
        faithfulness_values = [
            v for v in result['faithfulness'] 
            if not (isinstance(v, float) and math.isnan(v))
        ]
        context_precision_values = [
            v for v in result['context_precision'] 
            if not (isinstance(v, float) and math.isnan(v))
        ]
        
        avg_faithfulness = (
            sum(faithfulness_values) / len(faithfulness_values) 
            if faithfulness_values else 0
        )
        avg_context_precision = (
            sum(context_precision_values) / len(context_precision_values) 
            if context_precision_values else 0
        )
        avg_score = (avg_faithfulness + avg_context_precision) / 2
        
        return avg_faithfulness, avg_context_precision, avg_score
    
    # Вычисляем результаты
    faith_gen, cp_gen, score_gen = calculate_averages(result_general)
    faith_py, cp_py, score_py = calculate_averages(result_python)
    
    print("\n📊 ОБЩИЕ ВОПРОСЫ (все источники):")
    print(f"   Faithfulness:       {faith_gen:.4f}")
    print(f"   Context Precision:  {cp_gen:.4f}")
    print(f"   Средний балл:       {score_gen:.4f}")
    
    print("\n🐍 PYTHON ВОПРОСЫ (фильтр 'python'):")
    print(f"   Faithfulness:       {faith_py:.4f}")
    print(f"   Context Precision:  {cp_py:.4f}")
    print(f"   Средний балл:       {score_py:.4f}")
    
    # Сравнение эффективности фильтрации
    print(f"\n{'─'*70}")
    print("📈 АНАЛИЗ ЭФФЕКТИВНОСТИ ФИЛЬТРАЦИИ:")
    
    if score_py > score_gen:
        improvement = ((score_py - score_gen) / score_gen) * 100 if score_gen > 0 else 0
        print(f"   ✅ Фильтрация по источнику 'python' улучшила качество на {improvement:.1f}%")
        print("   Рекомендация: использовать фильтры для специализированных вопросов")
    elif score_py < score_gen:
        decline = ((score_gen - score_py) / score_gen) * 100 if score_gen > 0 else 0
        print(f"   ⚠️  Фильтрация снизила качество на {decline:.1f}%")
        print("   Возможные причины: недостаточно данных в источнике 'python'")
    else:
        print("   ➡️  Фильтрация не повлияла на качество")
    
    # Общая оценка системы
    overall_score = (score_gen + score_py) / 2
    print(f"\n{'='*70}")
    print(f"🎯 ОБЩАЯ ОЦЕНКА СИСТЕМЫ: {overall_score:.4f}")
    
    if overall_score >= 0.7:
        print("   Статус: Отличное качество! Система готова к продакшену ✅")
    elif overall_score >= 0.5:
        print("   Статус: Удовлетворительное качество. Рекомендуются улучшения ⚠️")
    else:
        print("   Статус: Требует значительного улучшения ❌")
    
    print("\n💡 РЕКОМЕНДАЦИИ:")
    print("   • Используйте фильтры источников для специализированных вопросов")
    print("   • Расширьте базу знаний Python для лучшего покрытия тем")
    print("   • Рассмотрите улучшение стратегии chunking для лучшего контекста")
    
    print("=" * 70)
    print("[OK] Комплексная оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()
