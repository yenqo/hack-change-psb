"""Поисковый индекс: BM25 со стеммером, dense-векторы bge-m3, гибрид через RRF; эмбеддинги вопросов."""

from __future__ import annotations

import math
import operator
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import TYPE_CHECKING

import runtime
from config import API_BASE, log, remaining
from textproc import expand_query, tokenize
from llm import embed

if TYPE_CHECKING:
    from kb import KB
    from pipeline import Prepared


class Index:
    def __init__(self, kb: KB) -> None:
        self.kb = kb
        self.k1, self.b = 1.5, 0.75
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.lens: list[int] = []
        for c in kb.chunks:
            toks = tokenize(c.search)
            self.lens.append(len(toks))
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t, f in tf.items():
                self.postings.setdefault(t, []).append((c.idx, f))
        n = max(1, len(kb.chunks))
        self.avgdl = (sum(self.lens) / n) if self.lens else 1.0
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        self.vecs: list[list[float]] | None = None

    def bm25(self, query: str) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for ci, f in self.postings[t]:
                dl = self.lens[ci]
                s = idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                scores[ci] = scores.get(ci, 0.0) + s
        return sorted(scores.items(), key=lambda x: -x[1])

    def build_dense(self, cap: float = 60.0) -> None:
        """Эмбеддинги базы через прокси. Не уложились в cap секунд — работаем без них."""
        texts = [c.search[:1200] for c in self.kb.chunks]
        if not texts or not API_BASE:
            return
        batches = [list(range(i, min(i + 16, len(texts)))) for i in range(0, len(texts), 16)]
        vecs: list[list[float] | None] = [None] * len(texts)

        def run(batch: list[int]) -> None:
            res = embed([texts[i] for i in batch])
            for i, v in zip(batch, res):
                vecs[i] = v

        t0 = time.monotonic()
        pool = ThreadPoolExecutor(max_workers=4)
        futs = [pool.submit(run, b) for b in batches]
        done, not_done = wait(futs, timeout=max(5.0, min(cap, remaining() - 120)))
        pool.shutdown(wait=False, cancel_futures=True)
        errors = [f.exception() for f in done if f.exception() is not None]
        if not_done or errors or any(v is None for v in vecs):
            reason = f"не уложились в {cap:.0f} c" if not_done else (f"{errors[0]}" if errors else "не все векторы")
            log(f"Эмбеддинги недоступны ({reason}), работаем на BM25")
            self.vecs = None
            return
        self.vecs = vecs  # type: ignore[assignment]
        runtime.FLAGS.dense = True
        log(f"Эмбеддинги базы готовы: {len(texts)} фрагментов за {time.monotonic() - t0:.1f} c")

    def dense(self, qvec: list[float] | None) -> list[tuple[int, float]]:
        if self.vecs is None or qvec is None:
            return []
        mul = operator.mul
        sc = [(i, sum(map(mul, qvec, v))) for i, v in enumerate(self.vecs)]
        return sorted(sc, key=lambda x: -x[1])

    def hybrid(self, query: str, qvec: list[float] | None, kinds: set[str]) -> list[int]:
        allowed = {c.idx for c in self.kb.chunks if self.kb.docs[c.doc].kind in kinds}
        lists = [[i for i, _ in self.bm25(expand_query(query)) if i in allowed][:80]]
        d = [i for i, _ in self.dense(qvec) if i in allowed][:80]
        if d:
            lists.append(d)
        rrf: dict[int, float] = {}
        for lst in lists:
            for r, i in enumerate(lst):
                rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + r)
        return [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])]


def embed_questions(preps: list[Prepared], cap: float = 25.0) -> None:
    qtexts = [(p.current + " " + " ".join(t for r, t in p.history if r == "user"))[:2000] for p in preps]
    batches = [list(range(i, min(i + 16, len(qtexts)))) for i in range(0, len(qtexts), 16)]
    out: list[list[float] | None] = [None] * len(qtexts)

    def run(batch: list[int]) -> None:
        for j, v in zip(batch, embed([qtexts[i] for i in batch])):
            out[j] = v

    pool = ThreadPoolExecutor(max_workers=4)
    futs = [pool.submit(run, b) for b in batches]
    done, not_done = wait(futs, timeout=max(3.0, min(cap, remaining() - 120)))
    pool.shutdown(wait=False, cancel_futures=True)
    if not_done or any(f.exception() for f in done) or any(v is None for v in out):
        log("Эмбеддинги вопросов не получены, поиск без них")
        return
    for p, v in zip(preps, out):
        p.qvec = v
