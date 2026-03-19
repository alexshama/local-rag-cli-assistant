from typing import Dict, Any, List
import os
import logging
from pathlib import Path
from openai import OpenAI

try:
    from .vector_store import VectorStore
    from .cache import RAGCache
    from .paths import CHROMA_DB_DIR, DEFAULT_CACHE_DB_PATH, DEFAULT_DATA_SOURCES
except ImportError:
    from vector_store import VectorStore
    from cache import RAGCache
    from paths import CHROMA_DB_DIR, DEFAULT_CACHE_DB_PATH, DEFAULT_DATA_SOURCES


logger = logging.getLogger(__name__)


class RAGPipeline:
    """Основной pipeline для RAG системы в API режиме."""
    
    def __init__(self, 
                 collection_name: str = "rag_collection",
                 cache_db_path: str | Path = DEFAULT_CACHE_DB_PATH,
                 data_sources: Dict[str, str | Path] | None = None,
                 model: str = "gpt-4o-mini",
                 force_reindex: bool = False):
        """
        Инициализация RAG pipeline.
        
        Args:
            collection_name: имя коллекции в ChromaDB
            cache_db_path: путь к базе данных кеша
            data_sources: словарь источников {название: путь_к_файлу}
            model: модель OpenAI для генерации ответов
            force_reindex: принудительно пересоздать индекс перед загрузкой данных
        """
        # Проверка API ключа
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY не установлен")
        
        self.model = model
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.data_sources = {
            name: Path(path).resolve() for name, path in (data_sources or DEFAULT_DATA_SOURCES).items()
        }
        
        # Инициализация компонентов
        logger.info("Инициализация RAG pipeline: model=%s, collection=%s", self.model, collection_name)
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=CHROMA_DB_DIR,
        )
        
        # Загрузка документов из источников, если коллекция пустая
        if force_reindex:
            logger.warning("Включена принудительная переиндексация")
            self.vector_store.reset_collection()
            self.vector_store.load_multiple_sources(self.data_sources)
        elif self.vector_store.collection.count() == 0 and self.data_sources:
            logger.info("Коллекция пуста, запускается первичная индексация")
            self.vector_store.load_multiple_sources(self.data_sources)
        elif self.data_sources:
            logger.info("Индексация пропущена: коллекция уже содержит документы")
        
        self.cache = RAGCache(db_path=cache_db_path)
        logger.info("RAG pipeline инициализирован")

        if force_reindex:
            logger.warning("Очистка cache после принудительной переиндексации")
            self.cache.clear()

    def reindex(self):
        """Явная принудительная переиндексация всех источников."""
        logger.warning("Запущена явная переиндексация всех источников")
        self.vector_store.reset_collection()
        self.vector_store.load_multiple_sources(self.data_sources)
        logger.warning("Очистка cache после явной переиндексации")
        self.cache.clear()

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
        logger.info(
            "Начало обработки запроса: query_length=%s, source_filter=%s",
            len(user_query),
            source_filter or "all",
        )
        
        # Создаём ключ кеша с учётом фильтра источника
        cache_key = f"{user_query}|source:{source_filter}" if source_filter else user_query
        
        try:
            # Шаг 1: Проверка кеша
            if use_cache:
                logger.info("Проверка cache")
                cached_result = self.cache.get(cache_key)
                
                if cached_result:
                    return {
                        "query": user_query,
                        "answer": cached_result["answer"],
                        "from_cache": True,
                        "context_docs": cached_result.get("context_docs", []),
                        "cached_at": cached_result.get("created_at"),
                        "source_filter": source_filter
                    }
            
            # Шаг 2: Поиск релевантных документов
            logger.info("Запуск поиска в vector store")
            context_docs = self.vector_store.search(user_query, top_k=3, source_filter=source_filter)
            logger.info("Найдено релевантных чанков: %s", len(context_docs))
            
            if context_docs and source_filter:
                sources_found = sorted({doc.get('source', 'unknown') for doc in context_docs})
                logger.debug("Источники в результатах: %s", ", ".join(sources_found))
            
            # Шаг 3: Формирование промпта
            logger.info("Формирование контекста и промпта")
            prompt = self._create_prompt(user_query, context_docs, source_filter)
            
            # Шаг 4: Генерация ответа через API
            logger.info("Отправка запроса в OpenAI: model=%s", self.model)
            answer = self._generate_answer(prompt)
            logger.info("Ответ от OpenAI успешно получен")
            
            # Шаг 5: Сохранение в кеш
            if use_cache:
                logger.info("Сохранение результата в cache")
                self.cache.set(cache_key, answer, context_docs)
            
            logger.info("Обработка запроса завершена")
            return {
                "query": user_query,
                "answer": answer,
                "from_cache": False,
                "context_docs": context_docs,
                "model": self.model,
                "mode": "API",
                "source_filter": source_filter
            }
        except Exception:
            logger.exception("Ошибка обработки запроса в RAG pipeline")
            raise
    
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
