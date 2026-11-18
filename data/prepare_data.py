"""Download, cache, split, and format EMNIST and public-domain NLP worksheet data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import ssl
import sys
import tempfile
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import certifi
import httpx
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.datasets import EMNIST

if str(_ROOT := Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data._io_errors import describe_os_error
from data._io_errors import map_os_error as _map_os_error
from data.emnist_data import SPLIT_SEED, TRAIN_FRAC, compute_stratified_split
from data.preprocess import CANONICAL_36_CHARS, fix_emnist_orientation

EMNIST_ROOT, EMNIST_SPLITS_PATH, RAW_GUTENBERG_DIR, RAW_WIKIPEDIA_DIR = Path("data/emnist"), Path("data/emnist_splits.json"), Path("data/raw/gutenberg"), Path("data/raw/wikipedia")
DEV_WORKSHEETS_PATH, TEST_WORKSHEETS_PATH, SAMPLE_WORKSHEETS_DIR, CORPUS_SOURCES_PATH = Path("data/worksheets_dev_50.json"), Path("data/worksheets_test_300.json"), Path("data/sample_worksheets"), Path("results/corpus_sources.json")
EMNIST_URL, EMNIST_MD5, USER_AGENT = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip", "58c8d27c78d21e728a6bc7b3cc06412e", "SWE015StudentApp/1.0 (mailto:student@university.edu)"
DEFAULT_SEED, HTTP_TIMEOUT_S, EMNIST_TIMEOUT_S = 42, 30.0, 120.0
TRAIN_MIN_WORDS, DEV_MIN_WORDS, PPL_TEST_MIN_WORDS, MIN_LEXICON_TARGET, FALLBACK_LEXICON_SIZE = 500_000, 50_000, 50_000, 20_000, 22_000
DEV_WORKSHEET_COUNT, TEST_WORKSHEET_COUNT, SAMPLE_WORKSHEET_COUNT, DEV_NUMERIC_TARGET, TEST_NUMERIC_TARGET, MAX_PER_TARGET_WORD, MIN_WORD_LEN, MAX_WORD_LEN = 50, 300, 15, 4, 20, 2, 3, 10
MIN_SENT_WORDS, MAX_SENT_WORDS, CELL_PADDING_PX, MIN_CELL_SIZE, MAX_BYTE_VAL, STREAM_CHUNK_BYTES = 7, 22, 10, 24, 255, 65536
_TOKEN_RE, _LEX_WORD_RE = re.compile(r"[A-Z0-9]+"), re.compile(r"^(?:[A-Z]{2,}|A|I)$")
_GUT_START_RE = re.compile(r"\*{3}\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK[^\r\n*]*\*{3}", re.IGNORECASE)
_GUT_END_RE = re.compile(r"\*{3}\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK[^\r\n*]*\*{3}", re.IGNORECASE)
_WIKI_HEADING_RE, _CLEAN_SENT_RE, _WORD_SPAN_RE = re.compile(r"(?m)^\s*=+[^=\r\n]+=+\s*$"), re.compile(r"^[A-Z][A-Za-z0-9 ,]+\.$"), re.compile(r"[A-Za-z0-9]+")
_BAD_SENT_PREFIXES, _CONFUSABLE_CHARS = ("It ", "They ", "This ", "These ", "Those ", "He ", "She "), frozenset("OISBZAG")
_STOP_WORDS = frozenset({"THE", "AND", "FOR", "ARE", "WAS", "WERE", "THAT", "WITH", "FROM", "HAVE", "THIS", "WHICH", "ALSO", "VERY", "MORE", "SOME", "MANY", "MOST", "INTO", "ONLY", "THAN", "THEM", "BEEN", "WHEN", "WHAT", "WHERE", "WILL", "ABOUT", "AFTER", "CALLED", "KNOWN", "THEY", "THEIR", "THERE", "THESE", "THOSE", "OTHER", "SUCH", "BECAUSE", "COULD", "WOULD", "SHOULD", "USED", "OFTEN", "BETWEEN", "THROUGH", "DURING", "BEFORE", "UNDER", "WHILE", "SINCE", "BEING", "MAKE", "MADE"})
_NUM_PREV_CUES = frozenset({"in", "around", "about", "nearly", "over", "after", "before", "since", "by", "until", "during", "between"})
_NUM_NEXT_CUES = frozenset({"days", "years", "bones", "degrees", "percent", "hours", "minutes", "seconds", "meters", "kilometers", "miles", "million", "billion", "thousand", "hundred", "kilograms", "grams", "tons", "centuries", "months", "weeks", "planets", "elements", "teeth", "countries", "people", "species"})
_GUTENBERG_BOOKS: tuple[tuple[str, int, str], ...] = (
    ("train", 1342, "Pride and Prejudice"), ("train", 84, "Frankenstein"), ("train", 1661, "The Adventures of Sherlock Holmes"),
    ("train", 11, "Alice's Adventures in Wonderland"), ("train", 1228, "On the Origin of Species"), ("train", 345, "Dracula"),
    ("dev", 43, "The Strange Case of Dr Jekyll and Mr Hyde"), ("dev", 215, "The Call of the Wild"),
    ("ppl_test", 35, "The Time Machine"), ("ppl_test", 36, "The War of the Worlds"),
)
_WIKI_POOLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "train": ("en", ("Science", "Physics", "Chemistry", "Biology", "Astronomy", "Solar System", "Gravity", "Oxygen", "Photosynthesis", "Electricity", "Atom", "Water", "Earth", "Atmosphere", "DNA", "Periodic table", "World War II", "Mathematics", "Light", "Energy", "Evolution", "Volcano", "Ocean", "Climate")),
    "dev": ("en", ("Thermodynamics", "Magnetism", "Cell (biology)", "Plate tectonics", "Sound", "Glacier", "Ecosystem", "Rocket")),
    "ppl_test": ("en", ("Quantum mechanics", "Genetics", "Electromagnetism", "Mineral", "Neuron", "Fossil", "Telescope", "Microscope")),
    "worksheet_dev": ("simple", ("Plant", "Animal", "Sun", "Moon", "Star", "Rain", "River", "Mountain", "Metal", "Glass", "Salt", "Sugar", "Heart", "Brain", "Blood", "Bone", "Tree", "Flower", "Bird", "Fish", "Cloud", "Wind")),
    "worksheet_test": ("simple", ("Planet", "Galaxy", "Universe", "Earthquake", "Desert", "Forest", "Ice", "Fire", "Heat", "Force", "Magnet", "Battery", "Computer", "Internet", "Airplane", "Train", "Bridge", "Clock", "Calendar", "Year", "Century", "History", "Geography", "Map", "Continent", "Europe", "Asia", "Africa", "Human skeleton", "Muscle", "Lung", "Stomach", "Eye", "Ear", "Skin", "Tooth", "Seed", "Leaf", "Root", "Soil", "Rock", "Gold", "Iron", "Copper", "Silver", "Carbon", "Nitrogen", "Hydrogen", "Acid", "Temperature", "Thermometer", "Compass", "Wheel", "Paper", "Book", "School", "Library", "Hospital", "Medicine", "Vaccine", "Virus", "Bacteria", "Boiling point", "Freezing point", "Triangle", "Circle", "World War I", "Apollo 11", "Steam engine", "Printing press", "Telephone", "Radio", "Television", "Bicycle", "Ship", "Pyramid", "Ancient Egypt", "Ancient Rome", "Industrial Revolution", "Renaissance", "Democracy", "Alphabet", "Dictionary", "Square", "Geometry", "Algebra", "Weather", "Lightning", "Snow", "Lake", "Island", "Valley")),
}

def _tls_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())

def _validate_int(val: Any, name: str, minimum: int = 0) -> int:
    if isinstance(val, bool) or not isinstance(val, int) or val < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return val

def read_data(path: Path, as_json: bool = False) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as err:
        raise _map_os_error(err, path, "reading file") from err
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")
    try:
        return json.loads(text) if as_json else text
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {os.fspath(path)}: {exc.msg}.") from exc

def write_data(path: Path, payload: Any, as_json: bool = False) -> None:
    text = (json.dumps(payload, indent=2) + "\n") if as_json else str(payload)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    except OSError as err:
        raise _map_os_error(err, path, "writing file") from err

def _download_emnist_balanced(root: Path) -> None:
    raw_dir, tmp_path = root / "EMNIST" / "raw", None
    raw_resolved = raw_dir.resolve()
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        md5 = hashlib.md5()
        with httpx.Client(timeout=EMNIST_TIMEOUT_S, verify=_tls_context(), follow_redirects=True) as client, client.stream("GET", EMNIST_URL) as resp:
            resp.raise_for_status()
            with tmp_path.open("wb") as handle:
                for chunk in resp.iter_bytes(chunk_size=STREAM_CHUNK_BYTES):
                    handle.write(chunk); md5.update(chunk)
        if md5.hexdigest() != EMNIST_MD5:
            raise RuntimeError(f"EMNIST MD5 mismatch: expected {EMNIST_MD5}, got {md5.hexdigest()}.")
        with zipfile.ZipFile(tmp_path, "r") as zf:
            for info in zf.infolist():
                posix = PurePosixPath(info.filename.replace("\\", "/"))
                if posix.is_absolute() or ".." in posix.parts or ":" in info.filename or not (raw_dir / posix).resolve().is_relative_to(raw_resolved):
                    raise RuntimeError(f"Unsafe path in EMNIST archive: {info.filename}.")
                if info.is_dir() or not posix.name.startswith("emnist-balanced-"):
                    continue
                out_path = raw_dir / posix.name.removesuffix(".gz")
                if not out_path.resolve().is_relative_to(raw_resolved):
                    raise RuntimeError(f"Unsafe destination path in EMNIST archive: {out_path.name}.")
                with zf.open(info, "r") as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(gzip.GzipFile(fileobj=src) if posix.name.endswith(".gz") else src, dst)
    except OSError as err:
        raise _map_os_error(err, raw_dir, "downloading EMNIST data") from err
    except (httpx.HTTPError, zipfile.BadZipFile, gzip.BadGzipFile, EOFError) as exc:
        raise RuntimeError(f"Failed to download/extract EMNIST from {EMNIST_URL}: {' '.join(str(exc).splitlines())}.") from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

def load_emnist_dataset(root: Path, train: bool) -> Any:
    try:
        if bool((ds := EMNIST(root=str(root), split="balanced", train=train, download=False))._check_exists()):
            return ds
    except RuntimeError:
        pass
    _download_emnist_balanced(root)
    ds = EMNIST(root=str(root), split="balanced", train=train, download=False)
    if not bool(ds._check_exists()):
        raise RuntimeError(f"EMNIST balanced dataset incomplete under {root}.")
    return ds

def build_emnist_splits(root: str | Path = "data/emnist", out_path: str | Path = "data/emnist_splits.json") -> dict[str, Any]:
    """Write the exact train/val split the vision models were trained with (scripts.train_vision.compute_stratified_split)."""
    if not isinstance(root, (str, os.PathLike)) or not isinstance(out_path, (str, os.PathLike)):
        raise TypeError("root and out_path must be strings or path-like objects.")
    ds = load_emnist_dataset(Path(root), train=True)
    train_idx, val_idx = compute_stratified_split(ds.targets, train_frac=TRAIN_FRAC, seed=SPLIT_SEED)
    payload = {"seed": SPLIT_SEED, "train_indices": sorted(int(i) for i in train_idx), "val_indices": sorted(int(i) for i in val_idx)}
    write_data(Path(out_path), payload, as_json=True)
    return {"seed": payload["seed"], "train_indices": list(payload["train_indices"]), "val_indices": list(payload["val_indices"])}

def load_emnist_splits(path: str | Path = "data/emnist_splits.json") -> dict[str, Any]:
    """Load and validate EMNIST train/val index splits JSON with explicit OS error mapping."""
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be a string or path-like object.")
    data = read_data(Path(path), as_json=True)
    if not isinstance(data, Mapping):
        raise TypeError("EMNIST splits file must contain a JSON object.")
    seed_val, train_raw, val_raw = _validate_int(data.get("seed"), "seed"), data.get("train_indices"), data.get("val_indices")
    if not isinstance(train_raw, list) or not isinstance(val_raw, list) or not train_raw or not val_raw:
        raise ValueError("EMNIST splits must contain non-empty train_indices and val_indices lists.")
    train_idx, val_idx = [_validate_int(i, "train_indices item") for i in train_raw], [_validate_int(i, "val_indices item") for i in val_raw]
    if not set(train_idx).isdisjoint(val_idx):
        raise ValueError("EMNIST train_indices and val_indices must be disjoint.")
    return {"seed": seed_val, "train_indices": train_idx, "val_indices": val_idx}

def _fetch_gutenberg_book(client: httpx.Client, bid: int, cache_dir: Path, force_download: bool) -> tuple[str, str]:
    dp = "/".join(list(str(bid)[:-1])) or "0"
    urls = [f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt", f"https://gutenberg.pglaf.org/{dp}/{bid}/{bid}-0.txt", f"https://gutenberg.pglaf.org/{dp}/{bid}/{bid}.txt"]
    cache_path, url_path, raw, used_url = cache_dir / f"pg{bid}.txt", cache_dir / f"pg{bid}.url", "", urls[0]
    if not force_download:
        try:
            raw = read_data(cache_path)
            used_url = read_data(url_path).strip() if url_path.is_file() else urls[0]
        except FileNotFoundError:
            pass
    if not raw.strip():
        for url in urls:
            try:
                resp = client.get(url); resp.raise_for_status()
                if (body := resp.content.decode("utf-8-sig", errors="replace")).strip():
                    raw, used_url = body, url; write_data(cache_path, raw); write_data(url_path, url); break
            except httpx.HTTPError:
                continue
    if not raw.strip():
        raise RuntimeError(f"Failed to download Gutenberg book {bid} from {urls[-1]}.")
    trimmed = raw[s.end():] if (s := _GUT_START_RE.search(raw)) else raw
    return (trimmed[:e.start()] if (e := _GUT_END_RE.search(trimmed)) else trimmed).strip(), used_url

def _fetch_wiki_article(client: httpx.Client, lang: str, title: str, cache_dir: Path, force_download: bool) -> dict[str, str]:
    cache_path = cache_dir / f"{lang}__{re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')}.json"
    if not force_download:
        try:
            cached = read_data(cache_path, as_json=True)
            if isinstance(cached, Mapping) and str(cached.get("text", "")).strip() and cached.get("title") and cached.get("url"):
                return {"title": str(cached["title"]), "url": str(cached["url"]), "text": _WIKI_HEADING_RE.sub("", str(cached["text"])).strip()}
        except FileNotFoundError:
            pass
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    try:
        resp = client.get(api_url, params={"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "format": "json", "titles": title})
        resp.raise_for_status(); pages = resp.json().get("query", {}).get("pages", {})
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(f"Failed to fetch Wikipedia article '{title}' from {api_url}: {exc}.") from exc
    page = next(iter(pages.values()), {})
    if not pages or "missing" in page or not (raw_extract := str(page.get("extract", ""))).strip():
        raise RuntimeError(f"Missing Wikipedia extract for '{title}' at {api_url}.")
    resolved = str(page.get("title", title))
    page_url = f"https://{lang}.wikipedia.org/wiki/{quote(resolved.replace(' ', '_'))}"
    write_data(cache_path, {"title": resolved, "requested_title": title, "url": page_url, "text": raw_extract}, as_json=True)
    return {"title": resolved, "url": page_url, "text": _WIKI_HEADING_RE.sub("", raw_extract).strip()}

def prepare_corpus(force_download: bool = False) -> dict[str, Any]:
    """Download/cache Gutenberg + Wikipedia corpus and enforce disjoint splits; writes only the raw-text cache."""
    if not isinstance(force_download, bool):
        raise TypeError("force_download must be a bool.")
    seen_req: dict[str, str] = {}
    for pool, (_, titles) in _WIKI_POOLS.items():
        for t in titles:
            if (k := t.strip().casefold()) in seen_req:
                raise ValueError(f"Overlapping Wikipedia title '{t}' in {pool} and {seen_req[k]}.")
            seen_req[k] = pool
    buckets: dict[str, list[Any]] = {k: [] for k in ("train", "dev", "ppl_test", "worksheet_dev", "worksheet_test")}
    word_counts, sources, seen_res = dict.fromkeys(buckets, 0), [], {}
    with httpx.Client(timeout=HTTP_TIMEOUT_S, headers={"User-Agent": USER_AGENT}, verify=_tls_context(), follow_redirects=True) as client:
        for split, bid, title in _GUTENBERG_BOOKS:
            text, url = _fetch_gutenberg_book(client, bid, RAW_GUTENBERG_DIR, force_download)
            wc = len(_TOKEN_RE.findall(text.upper())); buckets[split].append(text); word_counts[split] += wc
            sources.append({"split": split, "source_type": "gutenberg", "title": title, "url": url, "word_count": wc, "license": "Public Domain (Project Gutenberg)"})
        for split, (lang, titles) in _WIKI_POOLS.items():
            for title in titles:
                doc = _fetch_wiki_article(client, lang, title, RAW_WIKIPEDIA_DIR, force_download)
                if (rk := doc["title"].strip().casefold()) in seen_res and seen_res[rk] != split:
                    raise ValueError(f"Resolved Wikipedia title '{doc['title']}' overlaps across {seen_res[rk]} and {split}.")
                seen_res[rk] = split; wc = len(_TOKEN_RE.findall(doc["text"].upper())); word_counts[split] += wc
                buckets[split].append({**doc, "word_count": wc} if split.startswith("worksheet") else doc["text"])
                sources.append({"split": split, "source_type": f"{lang}.wikipedia", "title": doc["title"], "url": doc["url"], "word_count": wc, "license": "CC BY-SA 4.0 (Wikipedia)"})
    if word_counts["train"] < TRAIN_MIN_WORDS or word_counts["dev"] < DEV_MIN_WORDS or word_counts["ppl_test"] < PPL_TEST_MIN_WORDS:
        raise RuntimeError(f"Corpus word counts below minimum thresholds: {word_counts}.")
    stats = {"word_counts": word_counts, "doc_counts": {k: len(v) for k, v in buckets.items()}}
    return {"train_texts": buckets["train"], "dev_texts": buckets["dev"], "ppl_test_texts": buckets["ppl_test"], "worksheet_dev_docs": buckets["worksheet_dev"], "worksheet_test_docs": buckets["worksheet_test"], "sources": sources, "stats": stats}

def build_lexicon(train_tokens: Sequence[str], max_words: int = 24000, min_count: int = 2) -> dict[str, int]:
    """Build canonical uppercase word lexicon strictly from training tokens."""
    if isinstance(train_tokens, (str, bytes)) or not isinstance(train_tokens, Sequence):
        raise TypeError("train_tokens must be a sequence of token strings.")
    max_w, min_c, counts = _validate_int(max_words, "max_words", 1), _validate_int(min_count, "min_count", 1), Counter()
    for tok in train_tokens:
        if not isinstance(tok, str):
            raise TypeError("Each token in train_tokens must be a string.")
        if _LEX_WORD_RE.fullmatch(tok):
            counts[tok] += 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    filtered = [(w, c) for w, c in ordered if c >= min_c]
    use_fb = min_c == 2 and len(filtered) < MIN_LEXICON_TARGET and len(ordered) >= MIN_LEXICON_TARGET
    return dict(ordered[: min(max_w, FALLBACK_LEXICON_SIZE)] if use_fb else filtered[:max_w])

def _candidate_targets(doc: Mapping[str, Any]) -> list[tuple[str, str, str, int, int, str, str]]:
    cands: list[tuple[str, str, str, int, int, str, str]] = []
    for raw in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", str(doc["text"])).strip()):
        sent = raw.strip()
        if not sent.isascii() or not _CLEAN_SENT_RE.fullmatch(sent) or sent.startswith(_BAD_SENT_PREFIXES) or "[MASK]" in sent:
            continue
        if not (MIN_SENT_WORDS <= len(matches := list(_WORD_SPAN_RE.finditer(sent))) <= MAX_SENT_WORDS):
            continue
        toks = [m.group(0) for m in matches]
        for idx in range(1, len(matches)):
            word, prev_w, next_w = toks[idx].upper(), toks[idx - 1].lower(), (toks[idx + 1].lower() if idx + 1 < len(toks) else "")
            if not (MIN_WORD_LEN <= len(word) <= MAX_WORD_LEN):
                continue
            if word.isdigit() and len(word) in (3, 4) and (prev_w in _NUM_PREV_CUES or next_w in _NUM_NEXT_CUES):
                diff = "numeric"
            elif word.isalpha() and word not in _STOP_WORDS:
                diff = "confusable" if any(ch in _CONFUSABLE_CHARS for ch in word) else "easy"
            else:
                continue
            cands.append((diff, sent, word, matches[idx].start(), matches[idx].end(), str(doc["title"]), str(doc["url"])))
    return cands

def _select_worksheet_items(docs: Sequence[Mapping[str, Any]], total: int, numeric_target: int, prefix: str, emnist_split: str, allowed_indices: Sequence[int], targets: Any, lexicon: Mapping[str, int], seed: int) -> list[dict[str, Any]]:
    rng, arr = np.random.default_rng(seed), targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    char_pool: dict[str, list[int]] = {ch: [] for ch in CANONICAL_36_CHARS}
    for raw_idx in allowed_indices:
        if 0 <= (lbl := int(arr[idx := int(raw_idx)])) < len(CANONICAL_36_CHARS):
            char_pool[CANONICAL_36_CHARS[lbl]].append(idx)
    by_diff: dict[str, list[tuple[str, str, int, int, str, str]]] = {"numeric": [], "confusable": [], "easy": []}
    for doc in sorted(docs, key=lambda d: str(d["title"])):
        for diff, sent, word, s0, s1, title, url in _candidate_targets(doc):
            by_diff[diff].append((sent, word, s0, s1, title, url))
    for key, val in by_diff.items():
        by_diff[key] = [val[int(i)] for i in rng.permutation(len(val))]
    conf_t, used_sents, word_counts, chosen = (total - numeric_target) // 2, set(), Counter(), []
    for diff, quota in [("numeric", numeric_target), ("confusable", conf_t), ("easy", total - numeric_target - conf_t), ("confusable", total), ("easy", total)]:
        added = 0
        for sent, word, s0, s1, title, url in by_diff[diff]:
            if len(chosen) >= total or added >= quota:
                break
            if sent not in used_sents and word_counts[word] < MAX_PER_TARGET_WORD and (masked := f"{sent[:s0]}[MASK]{sent[s1:]}").count("[MASK]") == 1:
                used_sents.add(sent); word_counts[word] += 1; chosen.append((masked, word, diff, title, url)); added += 1
    if len(chosen) != total:
        raise RuntimeError(f"Could only extract {len(chosen)} of {total} clean worksheet items for {prefix}.")
    items: list[dict[str, Any]] = []
    for idx, (masked, word, diff, title, url) in enumerate(chosen, 1):
        emnist_ids = [int(char_pool[ch][int(rng.integers(len(char_pool[ch])))]) for ch in word]
        if any(int(arr[eidx]) != CANONICAL_36_CHARS.index(ch) for ch, eidx in zip(word, emnist_ids)):
            raise RuntimeError(f"EMNIST index label mismatch while assembling {prefix}_{idx:03d}.")
        items.append({"id": f"{prefix}_{idx:03d}", "sentence": masked, "target_word": word, "difficulty": diff, "emnist_split": emnist_split, "emnist_indices": emnist_ids, "char_labels": list(word), "source_title": title, "source_url": url, "in_lexicon": word in lexicon})
    return items

def render_emnist_word_strip(raw_emnist_chars: Sequence[torch.Tensor | np.ndarray], cell_size: int = 64, border_width: int = 2) -> Image.Image:
    """Render raw EMNIST character tensors/arrays into an upright dark-on-white boxed strip."""
    if isinstance(raw_emnist_chars, (str, bytes)) or not isinstance(raw_emnist_chars, Sequence) or not raw_emnist_chars:
        raise ValueError("raw_emnist_chars must be a non-empty sequence of 2D EMNIST tensors or arrays.")
    csize, bwidth = _validate_int(cell_size, "cell_size", MIN_CELL_SIZE), _validate_int(border_width, "border_width", 1)
    if bwidth * 2 >= csize or csize - 2 * CELL_PADDING_PX < 4:
        raise ValueError("border_width and padding must leave positive inner glyph area.")
    inner, width = csize - 2 * CELL_PADDING_PX, len(raw_emnist_chars) * csize
    strip = Image.new("L", (width, csize), MAX_BYTE_VAL)
    for idx, raw_ch in enumerate(raw_emnist_chars):
        upright = fix_emnist_orientation(raw_ch)
        arr2d = np.squeeze(upright.detach().cpu().numpy() if isinstance(upright, torch.Tensor) else np.asarray(upright))
        if arr2d.ndim != 2 or not np.all(np.isfinite(arr2d)):
            raise ValueError("Each character tensor/array must be finite with 2D spatial dimensions.")
        if np.issubdtype(arr2d.dtype, np.floating) and float(arr2d.max()) <= 1.0 and float(arr2d.min()) >= 0.0:
            arr2d = np.rint(arr2d * MAX_BYTE_VAL)
        dark = (MAX_BYTE_VAL - np.clip(arr2d, 0, MAX_BYTE_VAL).astype(np.uint8)).astype(np.uint8)
        with Image.fromarray(dark).resize((inner, inner), Image.Resampling.BILINEAR) as glyph:
            strip.paste(glyph, (idx * csize + CELL_PADDING_PX, CELL_PADDING_PX))
    draw = ImageDraw.Draw(strip)
    draw.rectangle([0, 0, width - 1, csize - 1], outline=0, width=bwidth)
    for idx in range(1, len(raw_emnist_chars)):
        draw.line([(idx * csize, 0), (idx * csize, csize - 1)], fill=0, width=bwidth)
    return strip

def render_worksheet_item_strip(item: Mapping[str, Any], emnist_dataset: Any, cell_size: int = 64) -> Image.Image:
    """Render the horizontal boxed strip image for one worksheet item from an EMNIST dataset."""
    if not isinstance(item, Mapping):
        raise TypeError("item must be a mapping containing emnist_indices.")
    indices = item.get("emnist_indices")
    if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)) or not indices:
        raise ValueError("item['emnist_indices'] must be a non-empty sequence of indices.")
    has_data, chars = hasattr(emnist_dataset, "data"), []
    for raw_idx in indices:
        idx = _validate_int(raw_idx, "emnist_indices element")
        sample = emnist_dataset.data[idx] if has_data else emnist_dataset[idx]
        chars.append(sample[0] if (not has_data and isinstance(sample, tuple)) else sample)
    return render_emnist_word_strip(chars, cell_size=cell_size)

def _export_sample_worksheets(test_items: Sequence[Mapping[str, Any]], test_ds: Any, out_dir: Path) -> list[dict[str, Any]]:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise _map_os_error(err, out_dir, "creating sample worksheets directory") from err
    picked, picked_ids = [], set()
    for ch in ("0", "1", "5", "8", "O", "I", "S", "B"):
        if (hit := next((it for it in test_items if it["id"] not in picked_ids and ch in it["target_word"]), None)) is not None:
            picked.append(hit); picked_ids.add(str(hit["id"]))
    for diff in ("numeric", "confusable", "easy", ""):
        for it in test_items:
            if len(picked) >= SAMPLE_WORKSHEET_COUNT or (diff in ("numeric", "confusable") and sum(1 for p in picked if p["difficulty"] == diff) >= 5):
                break
            if it["id"] not in picked_ids and (not diff or it["difficulty"] == diff):
                picked.append(it); picked_ids.add(str(it["id"]))
    manifest: list[dict[str, Any]] = []
    for idx, it in enumerate(picked[:SAMPLE_WORKSHEET_COUNT], 1):
        fname, img_path = f"sample_{idx:02d}.png", out_dir / f"sample_{idx:02d}.png"
        try:
            with render_worksheet_item_strip(it, test_ds) as img:
                img.save(img_path, format="PNG")
        except OSError as err:
            raise _map_os_error(err, img_path, "writing sample PNG") from err
        manifest.append({"id": it["id"], "filename": fname, "sentence": it["sentence"], "target_word": it["target_word"], "difficulty": it["difficulty"]})
    write_data(out_dir / "manifest.json", manifest, as_json=True)
    return manifest

def prepare_all(force_download: bool = False, emnist_root: str | Path = EMNIST_ROOT) -> dict[str, Any]:
    """Run full EMNIST split, corpus fetch, lexicon build, worksheet generation, and sample export."""
    if not isinstance(force_download, bool):
        raise TypeError("force_download must be a bool.")
    splits, corpus = build_emnist_splits(root=emnist_root, out_path=EMNIST_SPLITS_PATH), prepare_corpus(force_download=force_download)
    lexicon = build_lexicon([tok for text in corpus["train_texts"] for tok in _TOKEN_RE.findall(text.upper())])
    train_ds, test_ds = load_emnist_dataset(Path(emnist_root), train=True), load_emnist_dataset(Path(emnist_root), train=False)
    dev_items = _select_worksheet_items(corpus["worksheet_dev_docs"], DEV_WORKSHEET_COUNT, DEV_NUMERIC_TARGET, "dev", "val", splits["val_indices"], train_ds.targets, lexicon, DEFAULT_SEED)
    test_items = _select_worksheet_items(corpus["worksheet_test_docs"], TEST_WORKSHEET_COUNT, TEST_NUMERIC_TARGET, "test", "test", range(len(test_ds)), test_ds.targets, lexicon, DEFAULT_SEED)
    write_data(DEV_WORKSHEETS_PATH, dev_items, as_json=True); write_data(TEST_WORKSHEETS_PATH, test_items, as_json=True)
    samples = _export_sample_worksheets(test_items, test_ds, SAMPLE_WORKSHEETS_DIR)
    test_oov_rate, dev_oov_rate = sum(1 for it in test_items if not it["in_lexicon"]) / len(test_items), sum(1 for it in dev_items if not it["in_lexicon"]) / len(dev_items)
    stats = {**corpus["stats"], "lexicon_size": len(lexicon), "dev_oov_rate": dev_oov_rate, "test_oov_rate": test_oov_rate}
    write_data(CORPUS_SOURCES_PATH, {"sources": corpus["sources"], "stats": stats}, as_json=True)
    return {"corpus": corpus, "splits": splits, "lexicon": lexicon, "dev_items": dev_items, "test_items": test_items, "samples": samples, "test_oov_rate": test_oov_rate, "sources": corpus["sources"], "stats": stats}

def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: regenerate splits, worksheets, samples, and corpus citations; friendly one-line errors."""
    parser = argparse.ArgumentParser(description="Prepare EMNIST splits, corpus citations, and worksheet datasets.")
    parser.add_argument("--force-download", action="store_true", help="Re-download corpus sources instead of using the cache.")
    parser.add_argument("--emnist-root", type=Path, default=EMNIST_ROOT, help="EMNIST dataset root directory.")
    args = parser.parse_args(argv)
    try:
        prepare_all(force_download=args.force_download, emnist_root=args.emnist_root)
    except OSError as exc:
        sys.stderr.write(f"File system error: {describe_os_error(exc)}\n")
        return 1
    except (RuntimeError, ValueError, TypeError) as exc:
        sys.stderr.write(f"Error: {' '.join(str(exc).splitlines()).strip() or 'data preparation failed.'}\n")
        return 1
    sys.stdout.write("Data preparation complete.\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
