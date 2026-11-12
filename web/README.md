# Веб-интерфейс юридического ассистента

Чат для юристов поверх ядра ассистента (`assistant.LegalAssistant`, контракт 1.1 — [docs/integration.md](../docs/integration.md)).
Ответы — тем же конвейером, что и сабмит на платформу; к каждому ответу — цитаты с подсветкой пунктов в документах,
предупреждения и состав контекста модели.

## Состав

| Компонент | Где | Что делает |
|---|---|---|
| Бэкенд | `web/server.py` | FastAPI: API для фронта, прогрев ядра, поток ответа (SSE), маршрутизация моделей |
| Фронтенд | `web/frontend/` | React + TypeScript + Vite + Tailwind; собирается в `web/frontend/dist`, отдаётся бэкендом |
| Ядро | корневые `*.py` | Поиск, промпт, генерация, проверки — общее с сабмитом |
| Эмбеддинги | сервис `emb` в `compose.yaml` | bge-m3 на llama.cpp; без него поиск по законам — только BM25 |

```
браузер ──HTTP──► web (FastAPI :8000) ──► LegalAssistant ──► /internal/llm/v1 ─┬─► провайдер LLM (DeepSeek V4.1 Flash)
                                                                              └─► emb (bge-m3, llama.cpp :8080)
```

Ядро умеет ходить во все модели только по одному адресу (`LLM_BASE_URL`). Если задан `EMB_BASE_URL`, бэкенд выдаёт
ядру свой внутренний адрес `/internal/llm/v1` (с одноразовым токеном) и раскладывает запросы: генерация — к провайдеру
с настоящим ключом, эмбеддинги — к локальному серверу, реранкер — к `RERANK_BASE_URL` или 404 (ядро отключает его само).

## Быстрый старт (Docker)

```bash
cp .env.example .env        # заполнить LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
docker compose up --build
```

Открыть http://localhost:8000.

- Первый запуск — несколько минут: контейнер `emb` скачивает модель bge-m3 (~0,6 ГБ) в том `emb-models`;
  веб дожидается её и только потом прогревает ядро. Повторные запуски — секунды.
- Лёгкий режим без эмбеддингов (поиск по законам только BM25, ничего не скачивается):
  `docker compose up --build --no-deps web`.
- Другой порт на хосте: `WEB_HOST_PORT=8080 docker compose up --build`.
- Остановить: `docker compose down` (модель остаётся в томе; удалить вместе с ней — `docker compose down -v`).

## Переменные окружения

Читаются из окружения или из `.env` в корне репозитория (окружение важнее). В Docker `.env` подключается через `env_file`.

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | — | Провайдер генерации (OpenAI-совместимый API) и модель — DeepSeek V4.1 Flash. **Обязательны** |
| `EMB_BASE_URL` | — (в Docker: `http://emb:8080/v1`) | OpenAI-совместимый сервер эмбеддингов bge-m3. Не задан — поиск по законам только BM25 |
| `EMB_LOCAL_MODEL` | `bge-m3` | Имя модели для сервера эмбеддингов |
| `EMB_WAIT_SECONDS` | `300` | Сколько ждать готовности сервера эмбеддингов при старте |
| `RERANK_BASE_URL` | — | Сервер реранкера с `/rerank` (необязательно) |
| `WEB_ANSWER_TIMEOUT` | `150` | Предельное время одного ответа, с |
| `WEB_PORT` | `8000` | Порт, на котором слушает бэкенд (нужен внутренней маршрутизации) |
| `WEB_HOST_PORT` | `8000` | Только Docker: порт на хосте |
| `DATA_DIR` | `data` (в Docker `/app/data`) | Папка с `knowledge-base.zip` |

Остальные переменные ядра — в [README](../README.md#переменные-окружения). Ключи только в `.env`, он в `.gitignore`.

## Порты

| Порт | Сервис | Снаружи |
|---|---|---|
| 8000 | веб-интерфейс и API | да (`WEB_HOST_PORT`) |
| 8080 | эмбеддинги (`emb`) | нет, только внутри сети compose |

## API

| Метод | Путь | Что делает |
|---|---|---|
| POST | `/api/chat` | `{messages: [{role, content}]}` → поток SSE: события `stage` (`search` → `generate` → `verify` → `done`), затем `result` — `Result.to_dict()` по контракту 1.1. Раз в 15 с — комментарий-пинг |
| GET | `/api/documents` | Каталог документов базы |
| GET | `/api/documents/{doc_id}` | Документ целиком: текст, пункты, пометки редакций, вырезанные инструкции |
| GET | `/api/examples` | Демо-вопросы (`data/demo_questions.csv`) для карточек-примеров |
| GET | `/api/health` | Прогрев ядра (`warming` / `ready` / `error`) и состояние моделей для индикаторов |
| GET | `/api/ready` | Readiness-проба (`200` после загрузки базы, `503` во время прогрева или ошибки — для Docker `HEALTHCHECK`) |

Интерактивная документация — http://localhost:8000/docs.

**Индикаторы моделей** (`/api/health` → `models`): зелёный — работает; жёлтый — загружается или запасной режим
(эмбеддингов нет — BM25); красный — недоступно. Генерация проверяется запросом списка моделей у провайдера — без
расхода токенов, результат кэшируется на 30 с.

## Основной сценарий проверки

1. Открыть http://localhost:8000, дождаться зелёного индикатора LLM.
2. Нажать пример «Договор на 12 млн рублей заключали в августе 2024 года…» — в ответе статус «Ответ по документам»,
   цитата п. 2.4 регламента ред. 2023 и пометка «Учтена редакция на дату».
3. Нажать на цитату — справа откроется регламент с подсвеченным пунктом.
4. Вкладка «Контекст» — какие документы получила модель; «не действовал на дату» у регламента 2025.
5. Пример «Почему… арендная плата 500 000 рублей?» — статус «Исправлена ошибка в вопросе» (по договору 450 000).
6. Вопрос вне базы («Какой размер пособия по беременности и родам?») — честный ответ «в базе нет».

## Разработка без Docker

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r web/requirements.txt
cd web/frontend && npm install && npm run build && cd ../..
.venv/bin/uvicorn web.server:app --port 8000
```

Для правок фронта с мгновенным обновлением — второй терминал: `cd web/frontend && npm run dev`, открыть
http://localhost:5173 (запросы `/api` проксируются на порт 8000).

Локальные эмбеддинги без Docker (macOS, `brew install llama.cpp`), в `.env` — `EMB_BASE_URL=http://127.0.0.1:8081/v1`:

```bash
llama-server -hf gpustack/bge-m3-GGUF:Q8_0 --embedding --pooling cls --host 127.0.0.1 --port 8081 -c 8192 -b 8192 -ub 8192
```

## Развёртывание на пустом сервере Ubuntu 22.04

Нужен сервер с публичным IPv4, от 2 vCPU и 4 ГБ RAM (контейнер эмбеддингов занимает ~1 ГБ памяти), ~5 ГБ диска.

### 1. Docker Engine и Compose

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER   # затем перелогиниться, чтобы docker работал без sudo
```

### 2. Код и ключ

```bash
git clone <адрес репозитория> psb && cd psb
git checkout web                 # ветка с веб-интерфейсом (до слияния в main)
cp .env.example .env
nano .env                        # LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
chmod 600 .env
```

Репозиторий приватный: для `git clone` нужен deploy key или персональный токен GitHub.

### 3. Запуск на порту 80

```bash
WEB_HOST_PORT=80 docker compose up -d --build
docker compose logs -f web       # ждать строку «[web] ядро: ready …», выйти — Ctrl+C
```

Контейнеры перезапускаются сами после сбоя и после перезагрузки сервера (`restart: unless-stopped`).
Порт сохраняется только на этот запуск команды — чтобы не указывать его каждый раз, добавьте `WEB_HOST_PORT=80` в `.env`.

### 4. Доступ по IPv4

Docker публикует порт на всех IPv4-адресах сервера (`0.0.0.0:80`), интерфейс открывается по `http://<IPv4 сервера>/`.

- Проверить, что порт слушается: `sudo ss -ltnp | grep ':80 '`; снаружи: `curl -I http://<IPv4 сервера>/`.
- Если у провайдера облака есть сетевой фаервол (Security Group и т. п.) — разрешите входящий TCP 80.
- Если включён `ufw`: `sudo ufw allow OpenSSH && sudo ufw allow 80/tcp && sudo ufw enable`.
  Учтите: опубликованные Docker порты обходят правила `ufw` (Docker сам пишет правила iptables), поэтому
  закрыть порт через `ufw deny` не получится — ограничивайте доступ фаерволом облака.

### 5. Безопасность

В интерфейсе нет авторизации: любой, кто знает адрес, тратит токены вашего ключа. Для публичной ссылки ограничьте
доступ по IP в фаерволе облака или поднимите готовый прокси Caddy из [`compose.prod.yaml`](../compose.prod.yaml)
с Basic Auth и HTTPS (в [`Caddyfile`](../Caddyfile) уже отключена буферизация для SSE):

```bash
docker run --rm caddy:2.8-alpine caddy hash-password --plaintext 'ВАШ_ПАРОЛЬ'
# добавить в .env: WEB_BIND_IP=127.0.0.1, CADDY_USER=jury, CADDY_PASS_HASH='$2a$14$...'
docker compose -f compose.yaml -f compose.prod.yaml up -d --build
```

### Проверка контейнера

```bash
.venv/bin/python tools/smoke_web.py --url http://127.0.0.1:8000 --wait 180
```

Без Docker (проверка роутов, статики и SSE-контракта через FastAPI TestClient): `.venv/bin/python tools/smoke_web.py --inprocess`.

### Обновление

```bash
git pull && docker compose up -d --build
```

## Известные ограничения

- **Нет авторизации** — см. «Безопасность».
- **История чатов** хранится в браузере (localStorage), на сервере её нет.
- **Очередь «модель занята»** работает в пределах одной вкладки браузера; запросы из разных вкладок и от разных
  пользователей бэкенд обрабатывает параллельно.
- **Реранкер** в веб-версии не подключён (у провайдера генерации нет `/rerank`, локально модель слишком тяжёлая).
- **Время ответа** зависит от провайдера: обычно 5–15 с, диалоговые уточнения с переформулированием (R7) — до минуты.
  Тайм-аут — 150 с, после него — карточка ошибки с кнопкой «Повторить».
