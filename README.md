# Local RAG CLI Assistant

Небольшой учебный Python-проект с Retrieval-Augmented Generation (RAG).
Приложение загружает локальные текстовые источники в ChromaDB, ищет по ним релевантные чанки, отправляет найденный контекст в OpenAI и кэширует ответы в SQLite.

Проект ориентирован на простую и понятную структуру без лишней архитектурной сложности. Его удобно использовать как базу для изучения RAG-пайплайна, локального векторного поиска и CLI-интерфейса.

## What This Project Does

- индексирует локальные текстовые файлы в ChromaDB;
- поддерживает несколько источников знаний;
- позволяет фильтровать поиск по источнику через CLI;
- генерирует ответ через OpenAI на основе найденного контекста;
- кэширует ответы в SQLite;
- поддерживает принудительную переиндексацию;
- пишет диагностические логи в консоль и в `app.log`.

## How It Works

Пайплайн выглядит так:

1. пользователь вводит вопрос в CLI;
2. приложение проверяет SQLite cache;
3. если ответа нет в кэше, создаётся embedding для запроса;
4. ChromaDB ищет ближайшие чанки по локальной базе знаний;
5. найденный контекст передаётся в OpenAI;
6. модель возвращает финальный ответ;
7. результат сохраняется в кэш.

Поддерживаются два источника знаний по умолчанию:

- `docs` — общие материалы по AI / ML / RAG;
- `python` — отдельная база знаний по Python.

## Tech Stack

- Python 3.11+
- OpenAI API
- ChromaDB
- SQLite
- python-dotenv
- datasets / ragas для отдельного evaluation-сценария

## Project Structure

```text
.
├── assistant_api/
│   ├── app.py               # CLI entrypoint
│   ├── rag_pipeline.py      # основной RAG-пайплайн
│   ├── vector_store.py      # загрузка, chunking и поиск по ChromaDB
│   ├── cache.py             # SQLite cache
│   ├── logging_config.py    # централизованная настройка logging
│   ├── paths.py             # стабильные пути внутри проекта
│   ├── evaluate_ragas.py    # сценарий оценки качества через RAGAS
│   └── data/
│       ├── docs.txt         # общая база знаний
│       └── python.txt       # база знаний по Python
├── .env.example
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Клонируйте репозиторий:

```bash
git clone <YOUR_GIT_REMOTE_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

2. Создайте виртуальное окружение:

```bash
python -m venv .venv
```

3. Активируйте окружение:

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows CMD:

```cmd
.venv\Scripts\activate.bat
```

Linux / macOS:

```bash
source .venv/bin/activate
```

4. Установите зависимости:

```bash
pip install -r requirements.txt
```

## Environment Variables

Создайте `.env` на основе шаблона:

```bash
cp .env.example .env
```

Для Windows можно просто создать файл вручную.

Минимальное содержимое:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

Важно:

- не публикуйте `.env` в GitHub;
- не коммитьте реальные API-ключи;
- перед публикацией убедитесь, что в истории git нет секретов.

## Running the Project

Запуск CLI:

```bash
python assistant_api/app.py
```

Запуск с подробными логами:

```bash
python assistant_api/app.py --debug
```

Запуск с принудительной переиндексацией:

```bash
python assistant_api/app.py --reindex
```

## CLI Usage Example

Пример с общим поиском:

```text
💭 Ваш вопрос: Что такое RAG?
```

Пример с фильтром по Python:

```text
💭 Ваш вопрос: @python Что такое переменная?
```

Полезные команды внутри CLI:

- `stats` — статистика по индексу и кэшу;
- `clear` — очистка SQLite cache;
- `reindex` — принудительная переиндексация локальной базы знаний;
- `exit` / `quit` / `q` — выход.

## Logging

Логи пишутся:

- в консоль;
- в файл `app.log` в корне проекта.

По умолчанию используется уровень `INFO`.
Для более подробной диагностики используйте `--debug`.

## Local Data Storage

Проект хранит служебные данные в стабильных путях относительно корня репозитория:

- `chroma_db/` — локальная векторная база;
- `semantic_rag_cache_v2.db` — SQLite cache;
- `app.log` — лог-файл.

Это сделано так, чтобы поведение не зависело от текущей директории запуска.

## Limitations

- нужен действующий OpenAI API key;
- проект рассчитан на локальные текстовые файлы, а не на большие production-датасеты;
- качество ответа зависит от содержимого `docs.txt` и `python.txt`;
- evaluation через `ragas` не является полноценной production-валидацией;
- нет веб-интерфейса, только CLI;
- нет автоматической синхронизации индекса при изменении файлов: для обновления данных используйте `reindex`.

## Future Improvements

- добавить загрузку данных не только из `.txt`, но и из `.md`, `.pdf`, `.docx`;
- улучшить качество chunking для смешанных типов документов;
- добавить тесты на cache, indexing и search;
- расширить CLI отдельными командами для управления источниками;
- добавить API или веб-интерфейс;
- улучшить evaluation-сценарий с реальными ground truth данными;
- добавить экспорт метрик и более детальную observability.

## Publishing Checklist

Перед публикацией на GitHub проверьте:

- `.env` не попал в индекс git;
- реальные ключи удалены или перевыпущены;
- локальные файлы `chroma_db/`, `*.db`, `app.log`, виртуальные окружения не отслеживаются git;
- README соответствует текущему состоянию проекта;
- проект запускается командой `python assistant_api/app.py`.

## License

Проект распространяется под лицензией MIT. Подробности см. в файле `LICENSE`.
