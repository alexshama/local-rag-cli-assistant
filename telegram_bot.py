"""
Telegram bot for the local RAG assistant.
"""

import argparse
import asyncio
import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from assistant_api.interaction_logger import InteractionLogger
from assistant_api.logging_config import setup_logging
from assistant_api.paths import DEFAULT_CACHE_DB_PATH, DEFAULT_DATA_SOURCES, ENV_FILE_PATH
from assistant_api.rag_pipeline import RAGPipeline


logger = logging.getLogger(__name__)


if ENV_FILE_PATH.exists():
    load_dotenv(ENV_FILE_PATH)
else:
    load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telegram bot for the local RAG assistant")
    parser.add_argument("--debug", action="store_true", help="enable DEBUG logging")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="force reindex on startup before bot starts polling",
    )
    return parser.parse_args()


def parse_source_filter(message_text: str) -> tuple[str, Optional[str]]:
    source_filter = None
    cleaned_text = message_text.strip()

    if cleaned_text.startswith("@python "):
        source_filter = "python"
        cleaned_text = cleaned_text[8:].strip()
    elif cleaned_text.startswith("@docs "):
        source_filter = "docs"
        cleaned_text = cleaned_text[6:].strip()

    return cleaned_text, source_filter


class TelegramRAGBot:
    """Telegram bot wrapper around the current RAG pipeline."""

    def __init__(self, token: str, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.interaction_logger = InteractionLogger()
        self.application = Application.builder().token(token).build()

        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("logs", self.logs_command))
        self.application.add_handler(CommandHandler("reindex", self.reindex_command))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            (
                "Привет! Я Telegram-версия локального RAG-ассистента.\n\n"
                "Я отвечаю на вопросы на основе локальной базы знаний и OpenAI.\n"
                "Можно писать обычные вопросы или использовать фильтры:\n"
                "@python Что такое переменная?\n"
                "@docs Что такое RAG?\n\n"
                "Команды:\n"
                "/help - справка\n"
                "/stats - статистика индекса и кэша\n"
                "/logs - отправить логи в CSV\n"
                "/reindex - принудительная переиндексация"
            )
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            (
                "Как пользоваться ботом:\n\n"
                "- просто отправьте вопрос;\n"
                "- для поиска только по Python используйте префикс @python;\n"
                "- для поиска только по общим документам используйте @docs.\n\n"
                "Примеры:\n"
                "@python Что такое tuple?\n"
                "@docs Что такое embeddings?\n"
                "Что такое машинное обучение?\n\n"
                "Команды:\n"
                "/stats - статистика системы\n"
                "/logs - получить логи в CSV\n"
                "/reindex - переиндексация базы знаний"
            )
        )

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            stats = await asyncio.to_thread(self.pipeline.get_stats)
            vector_stats = stats["vector_store"]
            cache_stats = stats["cache"]

            sources_text = "\n".join(
                f"- {source}: {count}" for source, count in vector_stats.get("sources", {}).items()
            )
            if not sources_text:
                sources_text = "- источники пока не проиндексированы"

            message = (
                "Статистика системы:\n\n"
                f"Коллекция: {vector_stats['name']}\n"
                f"Документов: {vector_stats['count']}\n"
                f"Кэш-записей: {cache_stats['total_entries']}\n"
                f"Модель: {stats['model']}\n\n"
                "Источники:\n"
                f"{sources_text}"
            )
            await update.message.reply_text(message)
        except Exception:
            logger.exception("Ошибка при выполнении команды /stats")
            await update.message.reply_text("Не удалось получить статистику системы.")

    async def logs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        csv_path = None
        try:
            user_id = str(update.effective_user.id) if update.effective_user else None
            csv_path = await asyncio.to_thread(self.interaction_logger.export_csv, user_id)
            if csv_path is None:
                await update.message.reply_text("Для вас пока нет записей в журнале взаимодействий.")
                return

            with csv_path.open("rb") as log_file:
                await update.message.reply_document(
                    document=log_file,
                    filename="interaction_logs.csv",
                    caption="Журнал ваших взаимодействий с ботом в CSV-формате",
                )
        except Exception:
            logger.exception("Ошибка при выполнении команды /logs")
            await update.message.reply_text("Не удалось подготовить или отправить CSV с журналом.")
        finally:
            if csv_path and csv_path.exists():
                csv_path.unlink(missing_ok=True)

    async def reindex_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Запускаю переиндексацию. Это может занять некоторое время.")
        try:
            await asyncio.to_thread(self.pipeline.reindex)
            await update.message.reply_text("Переиндексация завершена.")
        except Exception:
            logger.exception("Ошибка при выполнении команды /reindex")
            await update.message.reply_text("Не удалось выполнить переиндексацию.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            return

        original_message = update.message.text
        query, source_filter = parse_source_filter(original_message)

        if not query:
            await update.message.reply_text("Пустой запрос. Напишите вопрос текстом.")
            return

        logger.info(
            "Telegram request received: user_id=%s, source_filter=%s, length=%s",
            update.effective_user.id if update.effective_user else "unknown",
            source_filter or "all",
            len(query),
        )

        await update.message.chat.send_action(action=ChatAction.TYPING)
        start_time = time.perf_counter()
        user_id = str(update.effective_user.id) if update.effective_user else "unknown"
        username = (
            update.effective_user.username
            or update.effective_user.first_name
            or "unknown"
        ) if update.effective_user else "unknown"

        try:
            result = await asyncio.to_thread(
                self.pipeline.query,
                query,
                True,
                source_filter,
            )

            answer = result["answer"]
            suffix = "\n\n💾 Ответ из кэша." if result["from_cache"] else ""
            full_text = answer + suffix
            await self.reply_in_chunks(update, full_text)

            response_time_ms = int((time.perf_counter() - start_time) * 1000)
            await asyncio.to_thread(
                self.interaction_logger.log_interaction,
                user_id,
                username,
                "telegram",
                query,
                answer,
                result["from_cache"],
                response_time_ms,
            )
        except Exception:
            logger.exception("Ошибка обработки Telegram-сообщения")
            await update.message.reply_text("Произошла ошибка при обработке запроса.")

    async def reply_in_chunks(self, update: Update, text: str, chunk_size: int = 4000):
        for start in range(0, len(text), chunk_size):
            await update.message.reply_text(text[start:start + chunk_size])

    def run(self):
        logger.info("Запуск Telegram bot polling")
        self.application.run_polling()


def main():
    args = parse_args()
    setup_logging(debug=args.debug)

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_token:
        raise ValueError("TELEGRAM_BOT_TOKEN не установлен в .env")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY не установлен в .env")

    logger.info("Инициализация Telegram bot")
    pipeline = RAGPipeline(
        collection_name="semantic_rag_collection_v2",
        cache_db_path=DEFAULT_CACHE_DB_PATH,
        data_sources=DEFAULT_DATA_SOURCES,
        model="gpt-4o-mini",
        force_reindex=args.reindex,
    )

    bot = TelegramRAGBot(token=telegram_token, pipeline=pipeline)
    bot.run()


if __name__ == "__main__":
    main()
