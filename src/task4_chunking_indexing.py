"""
Task 4 — Chunking & Indexing vào Vector Store (kèm metadata customer_role).

Luồng xử lý:
    data/standardized/**.md  →  load_documents()   → documents (kèm customer_role)
                             →  chunk_documents()  → chunks 800 ký tự, overlap 100
                             →  embed_chunks()     → vector 1024 chiều (BAAI/bge-m3)
                             →  index_to_vectorstore() → ChromaDB (cosine) tại chroma_db/

Chạy:
    python -m src.task4_chunking_indexing
    (hoặc: python src/task4_chunking_indexing.py)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu) hoặc đổi
EMBEDDING_PROVIDER, phải XÓA chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và
mới sẽ tồn tại lẫn lộn trong cùng collection (hoặc lệch số chiều vector 1024/768/1536)
và retrieval sẽ trả về kết quả rác.
"""

import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# Chunking: RecursiveCharacterTextSplitter
#   - CHUNK_SIZE = 800: đủ dài để giữ trọn 1 ý/1 mục trong cẩm nang (vài đoạn văn),
#     nhưng vẫn đủ ngắn để không nhồi cả file 50 trang vào prompt LLM (tốn chi phí +
#     loãng thông tin, LLM dễ bỏ sót chi tiết ở giữa context).
#   - CHUNK_OVERLAP = 100 (12.5% của 800): phần đuôi chunk trước được lặp lại ở đầu
#     chunk sau, tránh việc một câu quan trọng (VD: "giá vé vào cửa là ...") bị cắt
#     đôi đúng ranh giới và không chunk nào còn đủ ngữ cảnh để trả lời.
#   - "recursive": cắt ưu tiên theo \n\n → \n → ". " → " " nên hạn chế cắt giữa câu.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Embedding: BAAI/bge-m3 — multilingual (hỗ trợ tốt tiếng Việt có dấu), chạy local
# nên không cần API key và không giới hạn quota; vector 1024 chiều.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Cho phép cả nhóm đổi provider qua .env mà không phải sửa code (nhớ xoá chroma_db/
# khi đổi vì số chiều vector khác nhau: 1024 / 768 / 1536).
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")

# Vector store: ChromaDB — local persistent, không cần Docker, truy vấn cosine
# similarity trong vài mili-giây thay vì quét thủ công từng file text.
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "holiday_support_docs"

# Batch size khi encode để không ngốn hết RAM với file lớn (~70KB markdown).
EMBED_BATCH_SIZE = 16


# =============================================================================
# CUSTOMER ROLE — nhãn metadata để lọc đúng nhóm đối tượng khi retrieval
# =============================================================================
#
# Vì sao cần customer_role? Hai nhóm đối tượng đọc cùng một corpus nhưng cần thông
# tin khác nhau:
#   - "buyer"  : người dùng cuối / du khách — quan tâm giá vé, đặt phòng, lịch trình,
#                chi phí, hoàn tiền, cách di chuyển.
#   - "seller" : bên cung cấp dịch vụ — công ty lữ hành, khách sạn, hướng dẫn viên —
#                quan tâm quy định kinh doanh, phí hoa hồng/phí sàn, điều kiện hợp tác.
#   - "both"   : nội dung mô tả chung (giới thiệu điểm đến, ẩm thực, văn hoá) hai bên
#                đều dùng được — đây là mặc định khi không đủ tín hiệu phân loại.
#
# Cách phân loại: chấm điểm từ khoá (rule-based, không tốn API). Bên nào ăn điểm
# vượt trội thì gán nhãn bên đó; hoà hoặc không có tín hiệu → "both" (an toàn, vì
# "both" luôn được trả về cho mọi vai trò khi lọc).

BUYER_KEYWORDS = [
    # du lịch — góc nhìn người mua/du khách
    "du khách", "khách du lịch", "người mua", "khách hàng", "đặt phòng", "đặt tour",
    "đặt vé", "giá vé", "vé vào cửa", "chi phí", "lịch trình", "gợi ý cho bạn",
    "bạn nên", "kinh nghiệm", "nên đi", "di chuyển", "lưu trú", "ăn gì", "chơi đâu",
    "mua gì làm quà", "tự túc", "phượt", "review",
    # thương mại điện tử — góc nhìn người mua
    "hoàn tiền", "trả hàng", "phí vận chuyển", "theo dõi đơn hàng", "thanh toán",
]

SELLER_KEYWORDS = [
    # du lịch — góc nhìn bên cung cấp dịch vụ
    "nhà cung cấp", "đơn vị lữ hành", "công ty lữ hành", "doanh nghiệp", "đối tác",
    "chủ khách sạn", "chủ homestay", "hướng dẫn viên", "kinh doanh", "giấy phép",
    "hợp đồng", "hoa hồng", "đăng ký dịch vụ", "quản lý điểm đến", "vận hành tour",
    # thương mại điện tử — góc nhìn người bán
    "người bán", "nhà bán hàng", "gian hàng", "phí sàn", "đăng bán", "sản phẩm cấm",
    "quy định đăng bán", "shop",
]


def detect_customer_role(text: str) -> str:
    """
    Gán nhãn customer_role cho một đoạn text: "buyer" | "seller" | "both".

    Dùng đếm số lần xuất hiện từ khoá (không phân biệt hoa/thường). Chênh lệch phải
    đủ rõ (>= 2 điểm) mới gán nhãn riêng, tránh gán nhầm chỉ vì 1 từ khoá lạc.
    """
    lowered = text.lower()
    buyer_score = sum(lowered.count(kw) for kw in BUYER_KEYWORDS)
    seller_score = sum(lowered.count(kw) for kw in SELLER_KEYWORDS)

    if buyer_score - seller_score >= 2:
        return "buyer"
    if seller_score - buyer_score >= 2:
        return "seller"
    return "both"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source', 'type', 'doc_type',
                                              'customer_role'}}
    """
    documents = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            print(f"  [SKIP] File rỗng: {md_file.name}")
            continue

        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,       # key dùng ở app.py / task10
                "doc_type": doc_type,   # alias cho task5/task9
                # Nhãn ở mức tài liệu (tham khảo/thống kê); chunk sẽ tự phân loại lại.
                "customer_role": detect_customer_role(content),
            },
        })

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter (size=800, overlap=100).

    Mỗi chunk được gán lại customer_role riêng: một cẩm nang có thể vừa có phần cho
    du khách vừa có phần cho bên cung cấp dịch vụ, nên nhãn ở mức chunk chính xác
    hơn nhãn ở mức file.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            # Bỏ mảnh vụn (menu, dòng phân cách) — không đủ ngữ nghĩa để retrieval.
            if len(chunk_text) < 50:
                continue

            # Cố tình KHÔNG kế thừa nhãn của cả tài liệu: chunk không có tín hiệu
            # riêng (mô tả điểm đến, ẩm thực...) để "both" thì cả buyer lẫn seller
            # đều truy xuất được; ép theo nhãn tài liệu sẽ giấu mất chunk đó khỏi
            # nhóm còn lại.
            role = detect_customer_role(chunk_text)

            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                    "customer_role": role,
                    "char_count": len(chunk_text),
                },
            })

    return chunks


def get_embedding_model():
    """
    Trả về SentenceTransformer đã load (cache lại để không load model 2.2GB nhiều lần).
    """
    if not hasattr(get_embedding_model, "_model"):
        from sentence_transformers import SentenceTransformer

        print(f"  Đang load embedding model: {EMBEDDING_MODEL} (lần đầu sẽ tải ~2GB)...")
        get_embedding_model._model = SentenceTransformer(EMBEDDING_MODEL)
    return get_embedding_model._model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed danh sách text theo EMBEDDING_PROVIDER trong .env.

    Task 5 (semantic_search) gọi lại đúng hàm này để embed query, đảm bảo query và
    document luôn nằm cùng một không gian vector.
    """
    provider = EMBEDDING_PROVIDER.lower()

    if provider == "google":
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=texts,
            task_type="retrieval_document",
        )
        return result["embedding"]

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]

    # Mặc định: sentence-transformers (local, không cần API key)
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # chuẩn hoá để cosine similarity ổn định
    )
    return [emb.tolist() for emb in embeddings]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks; mỗi chunk được thêm key 'embedding': list[float].
    """
    if not chunks:
        return chunks

    embeddings = embed_texts([c["content"] for c in chunks])

    dim = len(embeddings[0])
    if dim != EMBEDDING_DIM:
        print(f"  [WARN] Vector {dim} chiều, khác EMBEDDING_DIM={EMBEDDING_DIM} "
              f"(provider={EMBEDDING_PROVIDER}). Nhớ xoá chroma_db/ cũ nếu vừa đổi provider.")

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


def get_collection():
    """
    Trả về ChromaDB collection (tạo mới nếu chưa có), dùng khoảng cách cosine.

    Task 5 import hàm này để query, nhờ đó Task 4 và Task 5 luôn trỏ về cùng một
    collection và cùng một cấu hình khoảng cách.
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection():
    """Xoá chroma_db/ cũ trước khi index lại để không lẫn chunk của corpus cũ."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print(f"  Đã xoá vector store cũ: {CHROMA_DIR}")


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks (content + embedding + metadata) vào ChromaDB.
    """
    if not chunks:
        print("  [WARN] Không có chunk nào để index.")
        return

    collection = get_collection()

    ids, documents, embeddings, metadatas = [], [], [], []
    for c in chunks:
        meta = c["metadata"]
        # id phải duy nhất: cùng file có nhiều chunk → ghép source + chunk_index
        safe_source = re.sub(r"[^A-Za-z0-9_.-]", "_", meta["source"])
        ids.append(f"{meta['type']}_{safe_source}_chunk_{meta['chunk_index']}")
        documents.append(c["content"])
        embeddings.append(c["embedding"])
        metadatas.append(meta)

    # upsert theo lô 100 để tránh vượt giới hạn payload của Chroma
    for start in range(0, len(ids), 100):
        end = start + 100
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"  Collection '{COLLECTION_NAME}' hiện có {collection.count()} chunks")


def run_pipeline(reset: bool = True):
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking     : {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding    : {EMBEDDING_MODEL} (dim={EMBEDDING_DIM}, provider={EMBEDDING_PROVIDER})")
    print(f"  Vector Store : {VECTOR_STORE} → {CHROMA_DIR}")
    print("=" * 60)

    docs = load_documents()
    print(f"\n[1/4] Loaded {len(docs)} documents")
    for d in docs:
        m = d["metadata"]
        print(f"      - {m['source']:<40} type={m['type']:<6} role={m['customer_role']}")

    chunks = chunk_documents(docs)
    print(f"\n[2/4] Created {len(chunks)} chunks")
    if chunks:
        sizes = [len(c["content"]) for c in chunks]
        print(f"      Độ dài chunk: min={min(sizes)}, max={max(sizes)}, "
              f"trung bình={sum(sizes) // len(sizes)} ký tự")
        roles = {}
        for c in chunks:
            role = c["metadata"]["customer_role"]
            roles[role] = roles.get(role, 0) + 1
        print(f"      Phân bố customer_role: {roles}")

    chunks = embed_chunks(chunks)
    print(f"\n[3/4] Embedded {len(chunks)} chunks "
          f"(vector {len(chunks[0]['embedding']) if chunks else 0} chiều)")

    print("\n[4/4] Indexing vào ChromaDB...")
    if reset:
        reset_collection()
    index_to_vectorstore(chunks)

    print("\n[OK] Task 4 hoàn tất. Vector store sẵn sàng cho Task 5 (semantic search).")


if __name__ == "__main__":
    run_pipeline()
