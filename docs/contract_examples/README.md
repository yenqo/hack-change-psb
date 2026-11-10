# Примеры JSON фасада

Сгенерированы `tools/make_contract_examples.py` (реальная модель через dev-прокси).
Статус вычисляется правилами по тексту ответа; точность правил будет измерена на golden set.

| файл | пример | статус | цитат |
|---|---|---|---|
| `answer_answer_fact.json` | answer | answer | 3 |
| `answer_no_info.json` | no_info | no_info | 0 |
| `answer_premise_error.json` | premise_error | premise_error | 2 |
| `answer_edition_on_date.json` | answer | premise_error | 2 |
| `answer_clarify_ambiguous.json` | clarify | clarify | 3 |
| `answer_dialog.json` | answer | answer | 1 |
| `answer_injection_in_question.json` | answer | partial | 2 |

Также: `list_documents.json`, `document_<doc_id>.json`, `health.json`.
