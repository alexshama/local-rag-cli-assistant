import sqlite3
import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, Any
import logging
from pathlib import Path

from paths import DEFAULT_CACHE_DB_PATH


logger = logging.getLogger(__name__)


class RAGCache:
    """Кеш для хранения результатов RAG запросов."""
    
    def __init__(self, db_path: str | Path = DEFAULT_CACHE_DB_PATH):
        """
        Инициализация кеша.
        
        Args:
            db_path: путь к файлу базы данных SQLite
        """
        self.db_path = Path(db_path).resolve()
        logger.info("Инициализация SQLite cache: %s", self.db_path)
        self._init_db()
    
    def _init_db(self):
        """Создание таблицы кеша, если она не существует."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    context TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
        except sqlite3.Error:
            logger.exception("Ошибка инициализации SQLite cache: %s", self.db_path)
            raise
    
    def _get_query_hash(self, query: str) -> str:
        """
        Вычисление хеша запроса для использования как ключ кеша.
        
        Args:
            query: текст запроса
            
        Returns:
            SHA-256 хеш запроса
        """
        # Нормализация запроса: lowercase и удаление лишних пробелов
        normalized_query = " ".join(query.lower().strip().split())
        return hashlib.sha256(normalized_query.encode()).hexdigest()

    def _normalize_context_docs(self, raw_context: Any) -> list[dict[str, Any]]:
        """
        Приведение контекста к единому формату.

        Поддерживает как новый формат (список словарей), так и legacy-формат
        (список строк), чтобы старый cache не ломал CLI.
        """
        if not raw_context:
            return []

        normalized_docs = []
        for index, item in enumerate(raw_context, 1):
            if isinstance(item, dict):
                normalized_docs.append({
                    "id": item.get("id", f"cached_{index}"),
                    "text": item.get("text", ""),
                    "distance": item.get("distance"),
                    "source": item.get("source", "cache"),
                    "file_path": item.get("file_path", "cache"),
                })
            elif isinstance(item, str):
                normalized_docs.append({
                    "id": f"cached_{index}",
                    "text": item,
                    "distance": None,
                    "source": "cache",
                    "file_path": "cache",
                })
            else:
                logger.warning("Неизвестный формат элемента context_docs в cache: %s", type(item).__name__)

        return normalized_docs
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Получение ответа из кеша.
        
        Args:
            query: текст запроса
            
        Returns:
            Словарь с ответом и метаданными, или None если не найдено
        """
        query_hash = self._get_query_hash(query)
        logger.info("Попытка чтения из cache")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT query, answer, context, created_at
                FROM cache
                WHERE query_hash = ?
            """, (query_hash,))
            
            result = cursor.fetchone()
            conn.close()
        except sqlite3.Error:
            logger.exception("Ошибка чтения из SQLite cache")
            raise
        
        if result:
            logger.info("Cache hit")
            raw_context = json.loads(result[2]) if result[2] else []
            return {
                "query": result[0],
                "answer": result[1],
                "context_docs": self._normalize_context_docs(raw_context),
                "created_at": result[3],
                "from_cache": True
            }
        
        logger.info("Cache miss")
        return None

    def set(self, query: str, answer: str, context_docs: list[dict[str, Any]] | None = None):
        """
        Сохранение ответа в кеш.
        
        Args:
            query: текст запроса
            answer: текст ответа
            context_docs: список документов, использованных как контекст
        """
        query_hash = self._get_query_hash(query)
        context_json = json.dumps(context_docs, ensure_ascii=False) if context_docs else None
        logger.info("Запись результата в cache")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Используем INSERT OR REPLACE для обновления существующих записей
            cursor.execute("""
                INSERT OR REPLACE INTO cache (query_hash, query, answer, context, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (query_hash, query, answer, context_json, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
        except sqlite3.Error:
            logger.exception("Ошибка записи в SQLite cache")
            raise
    
    def clear(self):
        """Очистка всего кеша."""
        logger.warning("Очистка всего cache")
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM cache")
            
            conn.commit()
            conn.close()
        except sqlite3.Error:
            logger.exception("Ошибка очистки SQLite cache")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики кеша.
        
        Returns:
            Словарь со статистикой
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM cache")
            count = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM cache")
            dates = cursor.fetchone()
            
            conn.close()
        except sqlite3.Error:
            logger.exception("Ошибка получения статистики SQLite cache")
            raise
        
        return {
            "total_entries": count,
            "oldest_entry": dates[0] if dates[0] else None,
            "newest_entry": dates[1] if dates[1] else None,
            "db_size_mb": self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0,
            "db_path": str(self.db_path)
        }


if __name__ == "__main__":
    # Тестирование кеша
    cache = RAGCache("test_cache.db")
    
    # Сохранение
    cache.set(
        query="Что такое машинное обучение?",
        answer="Машинное обучение - это раздел искусственного интеллекта...",
        context_docs=[
            {"id": "doc1", "text": "doc1", "distance": None, "source": "test", "file_path": "test.txt"},
            {"id": "doc2", "text": "doc2", "distance": None, "source": "test", "file_path": "test.txt"}
        ]
    )
    
    # Получение
    result = cache.get("Что такое машинное обучение?")
    print("Результат из кеша:", result)
    
    # Статистика
    stats = cache.get_stats()
    print("Статистика кеша:", stats)
    
    # Очистка тестовой БД
    import os
    if os.path.exists("test_cache.db"):
        os.remove("test_cache.db")

