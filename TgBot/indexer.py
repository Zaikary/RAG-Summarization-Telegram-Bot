import os
import logging

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional

from rank_bm25 import BM25Okapi
import re

load_dotenv()
logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBED_MODEL")
CHUNK_SIZE = 80
CHUNK_OVERLAP = 0
TASK_PREFIX_DOC = "search_document"
TASK_PREFIX_QUERY = "search_query"
INDEX_ROOT = os.path.join(os.getcwd(), "faiss_index")  # ./faiss_index/<collection>
_BM25_CACHE = {}

_SENTENCE_MODEL = None
def _get_sentence_model(name: str = EMBED_MODEL):
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        _SENTENCE_MODEL = SentenceTransformer(name)
    return _SENTENCE_MODEL


class RosBERTaEmbeddings:
    def __init__(self, model_name: str = EMBED_MODEL):
       self.model = _get_sentence_model(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"{TASK_PREFIX_DOC}: {text}" for text in texts]
        vecs = self.model.encode(prefixed, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        prefix = f"{TASK_PREFIX_QUERY}: {text}"
        v = self.model.encode(prefix, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return v.tolist()

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

def _collection_dir(collection_name: str):
    return os.path.join(INDEX_ROOT, collection_name)

def _ensure_collection_dir(collection_name: str):
    collection_dir = _collection_dir(collection_name)
    os.makedirs(collection_dir, exist_ok=True)
    return collection_dir

def _load_text(path: str) -> str:
    loader = TextLoader(path, encoding="utf8")
    pages = loader.load()
    return "\n".join([page.page_content or "" for page in pages])

def _load_pdf(path: str) -> str:
    loader = PyPDFLoader(path)
    pages = loader.load()
    texts = [page.page_content or "" for page in pages]
    return "\n".join(texts)

def _make_chunks(text: str, source: Optional[str] = None) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n"]
    )
    doc = Document(page_content=text, metadata={"source": source} if source else {})
    chunks = splitter.split_documents([doc])

    for chunk in chunks:
        error_codes = re.findall(r'PCBUD\d{4}', chunk.page_content)
        if error_codes:
            chunk.metadata["error_codes"] = error_codes
            chunk.page_content = ' '.join(error_codes) + ' ' + chunk.page_content
    return chunks

def index_file(path: str, collection: str) -> int:
    logger.info(f"Indexing file {path} into collection {collection}")
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".pdf":
        text = _load_pdf(path)
    elif ext == ".txt":
        text = _load_text(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    splits = _make_chunks(text, source=path)
    if not splits:
        logger.warning(f"No content found in file {path}")
        raise ValueError(f"No content found in file {path}")

    embeddings = RosBERTaEmbeddings()
    vs = FAISS.from_documents(splits, embeddings)
    col_dir = _collection_dir(collection)
    _ensure_collection_dir(collection)
    vs.save_local(col_dir)
    logger.info(f"Indexed {len(splits)} chunks from file {path} into collection {collection}")
    return len(splits)

def index_folder(folder: str, collection: str = "default") -> int:
    """Index all .txt and .pdf files in folder (sorted). Returns total chunks added (sum of files)."""
    if not os.path.isdir(folder):
        raise NotADirectoryError(folder)
    total = 0
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".txt", ".pdf")):
            continue
        path = os.path.join(folder, fname)
        try:
            total += index_file(path, collection=collection)
        except Exception as e:
            logger.exception("Failed to index %s: %s", path, e)
    logger.info("Indexed total %d chunks from folder %s into %s", total, folder, collection)
    return total

def load_vectorstore(collection: str="default") -> FAISS:
    embeddings = RosBERTaEmbeddings()
    col_dir = _collection_dir(collection)
    if not os.path.isdir(col_dir):
        raise FileNotFoundError(f"No saved collection at {col_dir}")
    vs = FAISS.load_local(col_dir, embeddings, allow_dangerous_deserialization=True)
    return vs

def _get_bm25_index(collection: str):
    if collection in _BM25_CACHE:
        return _BM25_CACHE[collection]

    vs = load_vectorstore(collection)
    docs = vs.docstore._dict.values()
    corpus = [doc.page_content for doc in docs]
    tokenized = [doc.lower().split() for doc in corpus]

    bm25 = BM25Okapi(tokenized)
    _BM25_CACHE[collection] = (bm25, list(docs))
    return bm25, list(docs)

def search_hybrid(query: str, collection: str="default", k: int=3) -> List[Dict]:
    is_error_code = bool(re.match(r'^([PCBUD])\d{4}$', query.upper().strip()))

    if is_error_code:
        query_normalized = query.upper().strip()
        logger.info(f"Using BM25 search for error code query: {query_normalized}")
        try:
            bm25, docs = _get_bm25_index(collection)
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return []

        tokenized_query = query_normalized.lower().split()
        scores = bm25.get_scores(tokenized_query)
        results = []
        for idx, score in enumerate(scores):
            doc_text = docs[idx].page_content.upper()
            # Точная проверка наличия кода в тексте
            if query_normalized in doc_text and score > 0:
                results.append({
                    "score": float(score),
                    "content": docs[idx].page_content,
                    "metadata": docs[idx].metadata
                })

        # Сортируем по score и берём топ-k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:1]
    else:
        logger.info(f"Using vector search for query: {query}")
        results = search(query, collection, k)

        RELEVANCE_THRESHOLD = 1.4  # Подбор под модель

        filtered = [r for r in results if r['score'] < RELEVANCE_THRESHOLD]

        if not filtered:
            logger.info(f"No relevant results found for query: {query} (best score: {results[0]['score'] if results else 'N/A'})")

        return filtered[:1]

def search(query: str, collection: str="default", k: int=5) -> List[Dict]:
    vs = load_vectorstore(collection)
    hits = vs.similarity_search_with_score(query, k=k)
    results: List[Dict] = []
    for doc, score in hits:
        results.append({"score": float(score), "content": doc.page_content, "metadata": getattr(doc, "metadata", {})})
    return results

def clear_collection(collection: str = "default") -> bool:
    """Удалить сохранённый индекс (файлы) для коллекции. Возвращает True если что-то удалено."""
    d = _collection_dir(collection)
    if os.path.isdir(d):
        for fname in os.listdir(d):
            try:
                os.remove(os.path.join(d, fname))
            except Exception:
                pass
        logger.info("Cleared collection %s", collection)
        return True
    return False
