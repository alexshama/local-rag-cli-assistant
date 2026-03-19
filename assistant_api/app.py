"""
Консольное приложение для взаимодействия с RAG ассистентом (API mode).
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from logging_config import setup_logging
from paths import DEFAULT_CACHE_DB_PATH, DEFAULT_DATA_SOURCES, ENV_FILE_PATH
from rag_pipeline import RAGPipeline


logger = logging.getLogger(__name__)


if ENV_FILE_PATH.exists():
    load_dotenv(ENV_FILE_PATH)
else:
    load_dotenv()


def print_banner():
    """Вывод приветственного баннера."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║         RAG Ассистент (API Mode)                        ║
║  Retrieval-Augmented Generation через OpenAI API        ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("Введите 'exit' или 'quit' для выхода")
    print("Введите 'stats' для просмотра статистики")
    print("Введите 'clear' для очистки кеша")
    print("Введите 'reindex' для принудительной переиндексации базы знаний")
    print("Добавьте '@python' к вопросу для поиска только в базе знаний Python")
    print("Пример: '@python Что такое переменная?'\n")


def print_response(result: dict):
    """
    Форматированный вывод ответа.

    Args:
        result: словарь с результатом запроса
    """
    print(f"\n{'─' * 60}")
    print(f"📝 Вопрос: {result['query']}")
    if result.get("source_filter"):
        print(f"🔌 Фильтр: {result['source_filter']}")
    print(f"{'─' * 60}")

    if result["from_cache"]:
        print("💾 Источник: КЕШ")
        if "cached_at" in result:
            print(f"   Сохранено: {result['cached_at']}")
    else:
        print(f"🌐 Источник: OpenAI API ({result.get('model', 'LLM')})")
        print(f"   Использовано документов: {len(result.get('context_docs', []))}")

    print(f"\n💬 Ответ:\n{result['answer']}")

    if result.get("context_docs"):
        print("\n📚 Использованный контекст:")
        for i, doc in enumerate(result["context_docs"][:2], 1):
            source_info = f" [{doc.get('source', 'unknown')}]" if doc.get("source") else ""
            preview = doc["text"][:150] + "..." if len(doc["text"]) > 150 else doc["text"]
            print(f"   {i}.{source_info} {preview}")

    print(f"{'─' * 60}\n")


def print_stats(pipeline: RAGPipeline):
    """
    Вывод статистики системы.

    Args:
        pipeline: экземпляр RAG pipeline
    """
    stats = pipeline.get_stats()

    print(f"\n{'═' * 60}")
    print("📊 СТАТИСТИКА СИСТЕМЫ")
    print(f"{'═' * 60}")

    print("\n🗄️  Векторное хранилище:")
    print(f"   Коллекция: {stats['vector_store']['name']}")
    print(f"   Документов: {stats['vector_store']['count']}")
    print(f"   Директория: {stats['vector_store']['persist_directory']}")

    if stats["vector_store"].get("sources"):
        print("\n📂 Источники данных:")
        for source, count in stats["vector_store"]["sources"].items():
            print(f"   {source}: {count} документов")

    print("\n💾 Кеш:")
    print(f"   Записей: {stats['cache']['total_entries']}")
    print(f"   Размер БД: {stats['cache']['db_size_mb']:.2f} MB")
    print(f"   Путь к БД: {stats['cache']['db_path']}")
    if stats["cache"]["oldest_entry"]:
        print(f"   Первая запись: {stats['cache']['oldest_entry']}")
    if stats["cache"]["newest_entry"]:
        print(f"   Последняя запись: {stats['cache']['newest_entry']}")

    print(f"\n🤖 Модель: {stats['model']}")
    print(f"🌐 Режим: {stats['mode']}")
    print(f"{'═' * 60}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI для RAG ассистента")
    parser.add_argument("--debug", action="store_true", help="включить DEBUG-логирование")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="принудительно переиндексировать документы при запуске",
    )
    return parser.parse_args()


def main():
    """Главная функция приложения."""
    args = parse_args()
    setup_logging(debug=args.debug)
    logger.info("Запуск CLI приложения")
    print_banner()

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY не установлен")
        print("❌ Ошибка: переменная окружения OPENAI_API_KEY не установлена")
        print("\nУстановите её следующим образом:")
        print("  Windows (PowerShell): $env:OPENAI_API_KEY='your-key'")
        print("  Windows (CMD): set OPENAI_API_KEY=your-key")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key'")
        sys.exit(1)

    try:
        print("🚀 Инициализация системы...\n")
        pipeline = RAGPipeline(
            collection_name="semantic_rag_collection_v2",
            cache_db_path=DEFAULT_CACHE_DB_PATH,
            data_sources=DEFAULT_DATA_SOURCES,
            model="gpt-4o-mini",
            force_reindex=args.reindex,
        )
        print("\n✅ Система готова к работе!\n")
    except Exception as e:
        logger.exception("Ошибка инициализации приложения")
        print(f"❌ Ошибка инициализации: {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("💭 Ваш вопрос: ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                logger.info("Завершение работы CLI пользователем")
                print("\n👋 До свидания!")
                break

            if user_input.lower() == "stats":
                print_stats(pipeline)
                continue

            if user_input.lower() == "clear":
                confirm = input("⚠️  Вы уверены, что хотите очистить кеш? (yes/no): ")
                if confirm.lower() in ["yes", "y", "да"]:
                    pipeline.cache.clear()
                    print("✅ Кеш очищен")
                continue

            if user_input.lower() == "reindex":
                logger.warning("Пользователь запустил принудительную переиндексацию")
                print("♻️ Переиндексация базы знаний...\n")
                pipeline.reindex()
                print("✅ Переиндексация завершена\n")
                continue

            if not user_input:
                print("⚠️  Пожалуйста, введите вопрос\n")
                continue

            source_filter = None
            if user_input.startswith("@python "):
                source_filter = "python"
                user_input = user_input[8:]
                print(f"🔌 Применён фильтр по источнику: {source_filter}")
            elif user_input.startswith("@docs "):
                source_filter = "docs"
                user_input = user_input[6:]
                print(f"🔌 Применён фильтр по источнику: {source_filter}")

            logger.info(
                "Получен запрос от пользователя: source_filter=%s, length=%s",
                source_filter or "all",
                len(user_input),
            )
            result = pipeline.query(user_input, source_filter=source_filter)
            logger.info("Обработка пользовательского запроса завершена")
            print_response(result)

        except KeyboardInterrupt:
            logger.info("CLI прерван пользователем через KeyboardInterrupt")
            print("\n\n👋 Прервано пользователем. До свидания!")
            break
        except Exception as e:
            logger.exception("Ошибка верхнего уровня в CLI")
            print(f"\n❌ Ошибка: {e}\n")


if __name__ == "__main__":
    main()
