"""
Основной RAG pipeline для API режима.
Управляет потоком: запрос -> кеш -> vector search -> LLM -> ответ -> кеш.
"""

from typing import Dict, Any, List
import os
from openai import OpenAI

from vector_store import VectorStore
from cache import RAGCache


class RAGPipeline:
    """Основной pipeline для RAG системы в API режиме."""
    
    def __init__(self, 
                 collection_name: str = "rag_collection",
                 cache_db_path: str = "rag_cache.db",
                 data_sources: Dict[str, str] = None,
                 model: str = "gpt-4o-mini"):
        """
        Инициализация RAG pipeline.
        
        Args:
            collection_name: имя коллекции в ChromaDB
            cache_db_path: путь к базе данных кеша
            data_sources: словарь источников {название: путь_к_файлу}
            model: модель OpenAI для генерации ответов
        """
        # Проверка API ключа
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY не установлен")
        
        self.model = model
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Инициализация компонентов
        print("Инициализация векторного хранилища...")
        self.vector_store = VectorStore(collection_name=collection_name)
        
        # Загрузка документов из источников, если коллекция пустая
        if self.vector_store.collection.count() == 0 and data_sources:
            print("Загрузка документов из источников...")
            self.vector_store.load_multiple_sources(data_sources)
        elif data_sources:
            print("Документы уже загружены в коллекцию")
        
        print("Инициализация кеша...")
        self.cache = RAGCache(db_path=cache_db_path)
        
        print("RAG Pipeline инициализирован (API mode)")
    
    def _create_prompt(self, query: str, context_docs: List[Dict[str, Any]], source_filter: str = None) -> str:
        """
        Создание промпта для LLM с контекстом.
        
        Args:
            query: вопрос пользователя
            context_docs: релевантные документы из векторного хранилища
            source_filter: фильтр по источнику
            
        Returns:
            сформированный промпт
        """
        # Формирование контекста из документов
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source_info = f" (Источник: {doc.get('source', 'unknown')})" if doc.get('source') else ""
            context_parts.append(f"Документ {i}{source_info}:\n{doc['text']}\n")
        
        context = "\n".join(context_parts)
        
        # Дополнительная инструкция для фильтрации по источнику
        source_instruction = ""
        if source_filter:
            source_instruction = f"\n- Отвечай преимущественно на основе информации из источника '{source_filter}'"
        
        # Создание промпта
        prompt = f"""Ты - полезный AI ассистент. Ответь на вопрос пользователя на основе предоставленного контекста.

Контекст:
{context}

Вопрос: {query}

Инструкции:
- Отвечай только на основе предоставленного контекста
- Если в контексте нет информации для ответа, скажи об этом
- Будь точным и кратким
- Отвечай на русском языке{source_instruction}

Ответ:"""
        
        return prompt
    
    def _generate_answer(self, prompt: str) -> str:
        """
        Генерация ответа через OpenAI API.
        
        Args:
            prompt: промпт для модели
            
        Returns:
            сгенерированный ответ
        """
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Ты - полезный AI ассистент, который отвечает на вопросы на основе предоставленного контекста."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Низкая температура для более точных ответов
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    def query(self, user_query: str, use_cache: bool = True, source_filter: str = None) -> Dict[str, Any]:
        """
        Основной метод для обработки запроса пользователя через API.
        
        Поток:
        1. Проверка кеша
        2. Если в кеше нет - поиск в векторном хранилище
        3. Формирование промпта с контекстом
        4. Генерация ответа через LLM API
        5. Сохранение в кеш
        
        Args:
            user_query: запрос пользователя
            use_cache: использовать ли кеш
            source_filter: фильтр по источнику (например, "python")
            
        Returns:
            словарь с ответом и метаданными
        """
        print(f"\n{'='*60}")
        print(f"Запрос: {user_query}")
        if source_filter:
            print(f"Фильтр по источнику: {source_filter}")
        print(f"{'='*60}")
        
        # Создаём ключ кеша с учётом фильтра источника
        cache_key = f"{user_query}|source:{source_filter}" if source_filter else user_query
        
        # Шаг 1: Проверка кеша
        if use_cache:
            print("[*] Проверка кеша...")
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                print("[+] Ответ найден в кеше")
                return {
                    "query": user_query,
                    "answer": cached_result["answer"],
                    "from_cache": True,
                    "context_docs": cached_result.get("context"),
                    "cached_at": cached_result.get("created_at"),
                    "source_filter": source_filter
                }
            else:
                print("[-] Ответ не найден в кеше")
        
        # Шаг 2: Поиск релевантных документов
        print("[*] Поиск релевантных документов через API...")
        context_docs = self.vector_store.search(user_query, top_k=3, source_filter=source_filter)
        print(f"[+] Найдено {len(context_docs)} релевантных документов")
        
        if context_docs and source_filter:
            sources_found = set(doc.get('source', 'unknown') for doc in context_docs)
            print(f"[+] Источники в результатах: {', '.join(sources_found)}")
        
        # Шаг 3: Формирование промпта
        print("[*] Формирование промпта...")
        prompt = self._create_prompt(user_query, context_docs, source_filter)
        
        # Шаг 4: Генерация ответа через API
        print(f"[*] Генерация ответа через OpenAI API ({self.model})...")
        answer = self._generate_answer(prompt)
        print("[+] Ответ получен от API")
        
        # Шаг 5: Сохранение в кеш
        if use_cache:
            print("[*] Сохранение в кеш...")
            context_for_cache = [doc['text'] for doc in context_docs]
            self.cache.set(cache_key, answer, context_for_cache)
            print("[+] Сохранено в кеш")
        
        return {
            "query": user_query,
            "answer": answer,
            "from_cache": False,
            "context_docs": context_docs,
            "model": self.model,
            "mode": "API",
            "source_filter": source_filter
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики системы.
        
        Returns:
            словарь со статистикой
        """
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "model": self.model,
            "mode": "API"
        }


if __name__ == "__main__":
    # Тестирование RAG pipeline в API режиме
    import sys
    
    try:
        pipeline = RAGPipeline()
        
        # Тестовые запросы
        test_queries = [
            "Что такое машинное обучение?",
            "Что такое RAG?",
            "Как работают трансформеры?"
        ]
        
        for query in test_queries:
            result = pipeline.query(query)
            print(f"\n{'='*60}")
            print(f"Вопрос: {result['query']}")
            print(f"Из кеша: {result['from_cache']}")
            print(f"Ответ: {result['answer']}")
            print(f"{'='*60}\n")
        
        # Повторный запрос (должен быть из кеша)
        print("\n--- Повторный запрос ---")
        result = pipeline.query(test_queries[0])
        print(f"Из кеша: {result['from_cache']}")
        
        # Статистика
        stats = pipeline.get_stats()
        print(f"\nСтатистика системы:")
        print(f"Векторное хранилище: {stats['vector_store']}")
        print(f"Кеш: {stats['cache']}")
        print(f"Режим: {stats['mode']}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

