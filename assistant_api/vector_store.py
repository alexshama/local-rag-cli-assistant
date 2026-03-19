import chromadb
from typing import List, Dict, Any
import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

try:
    from .paths import CHROMA_DB_DIR, DATA_DIR, ENV_FILE_PATH
except ImportError:
    from paths import CHROMA_DB_DIR, DATA_DIR, ENV_FILE_PATH


logger = logging.getLogger(__name__)

if ENV_FILE_PATH.exists():
    load_dotenv(ENV_FILE_PATH)
else:
    # Пытаемся загрузить из текущей директории
    load_dotenv()


class VectorStore:
    """Векторное хранилище на основе ChromaDB."""
    
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str | Path = CHROMA_DB_DIR):
        """
        Инициализация векторного хранилища.
        
        Args:
            collection_name: имя коллекции в ChromaDB
            persist_directory: директория для хранения данных
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Инициализация vector store: collection=%s, path=%s",
            self.collection_name,
            self.persist_directory,
        )
        
        # Инициализация ChromaDB клиента
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        
        # Получение или создание коллекции
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(
                "Коллекция загружена: name=%s, documents=%s",
                collection_name,
                self.collection.count(),
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Создана новая коллекция: %s", collection_name)
        
        # OpenAI клиент для создания embeddings
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _chunk_text_semantic(self, text: str, source: str = "docs") -> List[str]:
        """
        Семантическое разбиение текста на чанки по понятиям.
        Для Python файла - одно понятие = один чанк.
        Для обычных файлов - используется стандартное разбиение.
        
        Args:
            text: исходный текст
            source: источник данных для определения стратегии разбиения
            
        Returns:
            список чанков
        """
        if source == "python":
            return self._chunk_python_concepts(text)
        else:
            return self._chunk_text(text)
    
    def _chunk_python_concepts(self, text: str) -> List[str]:
        """
        Специальное разбиение Python файла на концептуальные чанки.
        Каждый чанк содержит одно понятие с его определением и примерами.
        
        Args:
            text: текст Python базы знаний
            
        Returns:
            список концептуальных чанков
        """
        chunks = []
        
        # Разделяем по секциям
        sections = text.split('====================')
        
        for section in sections:
            section = section.strip()
            if not section or section in ['PYTHON KNOWLEDGE BASE', 'FORMAT: QA + DEFINITIONS']:
                continue
            
            # Извлекаем название секции
            lines = section.split('\n')
            section_name = ""
            content_lines = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    section_name = line[1:-1]  # Убираем скобки
                elif line:
                    content_lines.append(line)
            
            # Обрабатываем содержимое секции
            i = 0
            while i < len(content_lines):
                line = content_lines[i]
                
                # Начало вопроса-ответа
                if line.startswith('Вопрос:'):
                    chunk_content = [line]
                    i += 1
                    
                    # Ищем ответ
                    while i < len(content_lines) and not content_lines[i].startswith('Ответ:'):
                        i += 1
                    
                    if i < len(content_lines):
                        chunk_content.append(content_lines[i])  # Добавляем ответ
                        i += 1
                        
                        # Добавляем связанные определения (следующие строки до пустой строки или нового вопроса)
                        while i < len(content_lines):
                            next_line = content_lines[i]
                            if (next_line.startswith('Вопрос:') or 
                                next_line == '' or 
                                (i + 1 < len(content_lines) and content_lines[i + 1].startswith('Вопрос:'))):
                                break
                            if " — " in next_line:  # Это определение
                                chunk_content.append("")  # Пустая строка для разделения
                                chunk_content.append(next_line)
                            i += 1
                        
                        # Создаем чанк для вопроса-ответа
                        chunk_text = f"[{section_name}]\n\n" + "\n".join(chunk_content)
                        chunks.append(chunk_text)
                
                # Отдельное определение (не связанное с вопросом)
                elif " — " in line and not any(chunk for chunk in chunks[-3:] if line.split(" — ")[0].strip() in chunk):
                    concept_name = line.split(" — ")[0].strip()
                    chunk_text = f"[{section_name}] Определение: {concept_name}\n\n{line}"
                    chunks.append(chunk_text)
                    i += 1
                else:
                    i += 1
        
        # Фильтруем пустые и слишком короткие чанки
        filtered_chunks = []
        for chunk in chunks:
            if len(chunk.strip()) >= 30:  # Минимальная длина чанка
                filtered_chunks.append(chunk.strip())
        
        return filtered_chunks
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Умное разбиение текста на чанки с учётом семантики.
        
        Стратегия:
        1. Приоритет абзацам (разделение по \n\n)
        2. Разбиение длинных абзацев по предложениям
        3. Сохранение контекста через overlap
        4. Учёт минимального и максимального размера чанка
        
        Args:
            text: исходный текст
            chunk_size: целевой размер чанка в символах
            overlap: размер перекрытия между чанками
            
        Returns:
            список чанков
        """
        # Разделяем текст на абзацы
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Если абзац помещается в текущий чанк
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # Если текущий чанк не пустой и добавление абзаца превысит размер
            elif current_chunk:
                chunks.append(current_chunk)
                # Добавляем overlap из конца предыдущего чанка
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            
            # Если абзац слишком большой, разбиваем его на предложения
            else:
                if len(paragraph) > chunk_size:
                    # Разбиваем длинный абзац на предложения
                    sentence_chunks = self._split_long_paragraph(paragraph, chunk_size, overlap)
                    
                    # Добавляем все чанки кроме последнего
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk)
        
        # Пост-обработка: фильтруем слишком короткие чанки
        chunks = [chunk for chunk in chunks if len(chunk) >= 50]
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Получение текста для overlap из конца предыдущего чанка.
        Пытается взять целые предложения.
        
        Args:
            text: текст для извлечения overlap
            overlap_size: желаемый размер overlap
            
        Returns:
            текст overlap
        """
        if len(text) <= overlap_size:
            return text
        
        # Берём последние overlap_size символов
        overlap_candidate = text[-overlap_size:]
        
        # Ищем начало предложения в overlap
        sentence_starts = ['. ', '! ', '? ', '\n']
        best_start = 0
        
        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)
        
        if best_start > 0:
            return overlap_candidate[best_start:].strip()
        
        return overlap_candidate.strip()
    
    def _split_long_paragraph(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Разбиение длинного абзаца на чанки по предложениям.
        
        Args:
            paragraph: абзац для разбиения
            chunk_size: целевой размер чанка
            overlap: размер перекрытия
            
        Returns:
            список чанков
        """
        # Разделяем на предложения
        import re
        sentences = re.split(r'([.!?]+\s+)', paragraph)
        
        # Собираем предложения обратно с их разделителями
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])
        
        # Если осталось что-то в конце без разделителя
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])
        
        chunks = []
        current_chunk = ""
        
        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Если предложение помещается в текущий чанк
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunks.append(current_chunk)
                    # Добавляем overlap
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    # Если одно предложение больше chunk_size, всё равно добавляем его
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def load_documents(self, file_path: str, source: str = "docs"):
        """
        Загрузка документов из файла в векторное хранилище с метаданными источника.
        
        Args:
            file_path: путь к файлу с документами
            source: источник документов (например, "docs", "python")
        """
        file_path = Path(file_path).resolve()
        logger.info("Начало индексации источника: source=%s, file=%s", source, file_path)

        # Проверка существования файла
        if not file_path.exists():
            raise FileNotFoundError(f"Файл {file_path} не найден")

        try:
            # Чтение файла
            with file_path.open('r', encoding='utf-8') as f:
                text = f.read()
            
            # Разбиение на чанки с учётом источника
            chunks = self._chunk_text_semantic(text, source)
            logger.info("Подготовлены чанки для индексации: source=%s, chunks=%s", source, len(chunks))
            
            # Создание embeddings и добавление в ChromaDB
            documents = []
            ids = []
            embeddings = []
            metadatas = []
            
            # Получаем текущее количество документов для уникальных ID
            current_count = self.collection.count()
            
            for i, chunk in enumerate(chunks):
                # Создание embedding через OpenAI
                embedding = self._create_embedding(chunk)
                
                documents.append(chunk)
                ids.append(f"{source}_{current_count + i}")
                embeddings.append(embedding)
                metadatas.append({"source": source, "file_path": str(file_path)})
                
                if (i + 1) % 10 == 0:
                    logger.debug("Проиндексировано чанков: source=%s, progress=%s/%s", source, i + 1, len(chunks))
            
            if documents:
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    ids=ids,
                    metadatas=metadatas
                )

            logger.info(
                "Индексация завершена: collection=%s, source=%s, documents=%s",
                self.collection_name,
                source,
                len(chunks),
            )
        except Exception:
            logger.exception("Ошибка индексации источника: source=%s", source)
            raise
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        Создание векторного представления текста через OpenAI.
        
        Args:
            text: текст для векторизации
            
        Returns:
            вектор embeddings
        """
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def load_multiple_sources(self, sources: Dict[str, str]):
        """
        Загрузка документов из нескольких источников.
        
        Args:
            sources: словарь {источник: путь_к_файлу}
        """
        for source, file_path in sources.items():
            self.load_documents(file_path, source)

    def reset_collection(self):
        """Полное удаление и пересоздание коллекции для принудительной переиндексации."""
        logger.warning("Принудительный сброс коллекции перед переиндексацией: %s", self.collection_name)
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Коллекция пересоздана: %s", self.collection_name)
    
    def search(self, query: str, top_k: int = 3, source_filter: str = None) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу с возможностью фильтрации по источнику.
        
        Args:
            query: текст запроса
            top_k: количество документов для возврата
            source_filter: фильтр по источнику (например, "python")
            
        Returns:
            список документов с метаданными
        """
        logger.info("Поиск в vector store: top_k=%s, source_filter=%s", top_k, source_filter or "all")
        try:
            # Создание embedding для запроса
            query_embedding = self._create_embedding(query)
            
            # Подготовка фильтра метаданных
            where_filter = None
            if source_filter:
                where_filter = {"source": source_filter}
            
            # Поиск в ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter
            )
            
            # Форматирование результатов
            documents = []
            if results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    doc_metadata = results['metadatas'][0][i] if 'metadatas' in results and results['metadatas'] else {}
                    documents.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None,
                        'source': doc_metadata.get('source', 'unknown'),
                        'file_path': doc_metadata.get('file_path', 'unknown')
                    })

            logger.info("Поиск завершён: results=%s", len(documents))
            logger.debug("Источники найденных документов: %s", [doc["source"] for doc in documents])
            return documents
        except Exception:
            logger.exception("Ошибка поиска в vector store")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики коллекции с разбивкой по источникам.
        
        Returns:
            словарь со статистикой
        """
        total_count = self.collection.count()
        
        # Получаем статистику по источникам
        sources_stats = {}
        if total_count > 0:
            # Получаем все документы для подсчёта по источникам
            all_docs = self.collection.get(include=['metadatas'])
            if all_docs['metadatas']:
                for metadata in all_docs['metadatas']:
                    source = metadata.get('source', 'unknown')
                    sources_stats[source] = sources_stats.get(source, 0) + 1
        
        return {
            'name': self.collection_name,
            'count': total_count,
            'persist_directory': str(self.persist_directory),
            'sources': sources_stats
        }


if __name__ == "__main__":
    # Тестирование векторного хранилища
    import sys
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Ошибка: установите переменную окружения OPENAI_API_KEY")
        sys.exit(1)
    
    vector_store = VectorStore(collection_name="test_collection")
    
    # Загрузка документов
    docs_path = DATA_DIR / "docs.txt"
    if docs_path.exists():
        vector_store.load_documents(docs_path)
    
    # Поиск
    results = vector_store.search("Что такое машинное обучение?", top_k=3)
    print("\nРезультаты поиска:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['text'][:200]}...")
        print(f"   Distance: {doc['distance']}")
    
    # Статистика
    stats = vector_store.get_collection_stats()
    print(f"\nСтатистика: {stats}")

