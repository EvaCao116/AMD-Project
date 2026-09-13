"""
Run google/embeddinggemma-300m locally without FastAPI or Uvicorn.

Uses:
  - sentence-transformers for loading / running the embedding model
  - Python stdlib http.server for the local API server

Install:
    pip install -U sentence-transformers transformers torch

Serve:
    python embeddinggemma_no_fastapi.py serve --host 127.0.0.1 --port 8000

OpenAI-compatible embeddings endpoint:
    curl http://127.0.0.1:8000/v1/embeddings \\
      -H "Content-Type: application/json" \\
      -d '{"input": ["hello world", "another sentence"]}'

Simple endpoint:
    curl http://127.0.0.1:8000/embed \\
      -H "Content-Type: application/json" \\
      -d '{"text": ["hello world"], "mode": "auto", "normalize": true}'

CLI:
    python embeddinggemma_no_fastapi.py encode --text "hello world"

Use local model directory:
    python embeddinggemma_no_fastapi.py serve --model ./embeddinggemma-300m
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, List, Literal, Optional, Union
from urllib.parse import urlparse


#DEFAULT_MODEL = str(
  #  Path(__file__).resolve().parent.parent.parent.parent / "/models/embedders/google/embeddinggemma300m"
#)
#
DEFAULT_MODEL = str(
    Path(__file__).resolve().parents[1]
    / "models"
    / "embedders"
    / "google"
    / "embeddinggemma-300m"
)
print(f"Using default model path: {DEFAULT_MODEL}", file=sys.stderr)

def load_model(
    model_name_or_path: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    max_seq_length: Optional[int] = None,
    local_files_only: bool = False,
):
    """
    Load EmbeddingGemma with SentenceTransformers.
    """

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sentence-transformers\n\n"
            "Install with:\n"
            "    pip install -U sentence-transformers transformers torch"
        ) from exc

    # Only validate absolute local paths.
    # Do not validate Hugging Face model IDs like google/embeddinggemma-300m here.
    model_path = Path(model_name_or_path)

    if model_path.is_absolute() and not model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {model_name_or_path}\n\n"
            "This is an absolute path because it starts with '/'.\n"
            "Use either a Hugging Face model ID:\n"
            "  --model google/embeddinggemma-300m\n\n"
            "or a valid local path like:\n"
            "  --model models/embedders/google/embeddinggemma-300m"
        )

    kwargs = {}

    if device:
        kwargs["device"] = device

    if cache_dir:
        kwargs["cache_folder"] = cache_dir

    print(f"Loading model: {model_name_or_path}", file=sys.stderr)

    model = SentenceTransformer(
        model_name_or_path,
        local_files_only=local_files_only,
        **kwargs,
    )

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
        print(f"Set max_seq_length={max_seq_length}", file=sys.stderr)

    print("Model loaded.", file=sys.stderr)
    return model


def encode_texts(
    model,
    texts: List[str],
    mode: Literal["auto", "query", "document"] = "auto",
    normalize: bool = True,
    batch_size: int = 32,
    show_progress: bool = False,
) -> List[List[float]]:
    """
    Encode texts into embeddings.

    mode:
      auto      -> model.encode()
      query     -> model.encode_query() if available, else model.encode()
      document  -> model.encode_document() if available, else model.encode()

    normalize:
      True is recommended for cosine similarity / semantic search.
    """
    if not texts:
        return []

    encode_kwargs = {
        "batch_size": batch_size,
        "normalize_embeddings": normalize,
        "convert_to_numpy": True,
        "show_progress_bar": show_progress,
    }

    if mode == "query" and hasattr(model, "encode_query"):
        embeddings = model.encode_query(texts, **encode_kwargs)
    elif mode == "document" and hasattr(model, "encode_document"):
        embeddings = model.encode_document(texts, **encode_kwargs)
    else:
        embeddings = model.encode(texts, **encode_kwargs)

    return embeddings.astype("float32").tolist()


def parse_text_input(value: Any, field_name: str = "input") -> List[str]:
    """
    Accept a string or list of strings.
    """
    if isinstance(value, str):
        return [value]

    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value

    raise ValueError(f"'{field_name}' must be a string or a list of strings.")


def read_json_body(handler: BaseHTTPRequestHandler, max_body_bytes: int) -> Any:
    content_length_raw = handler.headers.get("Content-Length")

    if not content_length_raw:
        raise ValueError("Missing Content-Length header.")

    try:
        content_length = int(content_length_raw)
    except ValueError:
        raise ValueError("Invalid Content-Length header.")

    if content_length < 0:
        raise ValueError("Invalid Content-Length header.")

    if content_length > max_body_bytes:
        raise ValueError(f"Request body too large. Limit is {max_body_bytes} bytes.")

    raw = handler.rfile.read(content_length)

    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON body: {exc}") from exc


def make_handler(
    model,
    model_name: str,
    api_key: Optional[str],
    default_batch_size: int,
    default_normalize: bool,
    max_body_bytes: int,
):
    """
    Create a request handler class with access to loaded model.
    """
    encode_lock = threading.Lock()

    class EmbeddingGemmaHandler(BaseHTTPRequestHandler):
        server_version = "EmbeddingGemmaHTTP/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write(
                "%s - - [%s] %s\n"
                % (self.address_string(), self.log_date_time_string(), fmt % args)
            )

        def send_json(self, status_code: int, payload: Any) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(data)

        def send_error_json(self, status_code: int, message: str) -> None:
            self.send_json(
                status_code,
                {
                    "error": {
                        "message": message,
                        "type": "request_error",
                    }
                },
            )

        def require_auth(self) -> bool:
            if not api_key:
                return True

            auth = self.headers.get("Authorization", "")
            expected = f"Bearer {api_key}"

            if auth != expected:
                self.send_error_json(401, "Unauthorized.")
                return False

            return True

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self) -> None:
            path = urlparse(self.path).path

            if path == "/" or path == "":
                self.send_json(
                    200,
                    {
                        "status": "ok",
                        "model": model_name,
                        "endpoints": [
                            "/health",
                            "/embed",
                            "/v1/embeddings",
                        ],
                    },
                )
                return

            if path == "/health":
                self.send_json(
                    200,
                    {
                        "status": "ok",
                        "model": model_name,
                    },
                )
                return

            self.send_error_json(404, f"Unknown endpoint: {path}")

        def do_POST(self) -> None:
            path = urlparse(self.path).path

            if not self.require_auth():
                return

            if path == "/embed":
                self.handle_embed()
                return

            if path == "/v1/embeddings":
                self.handle_openai_embeddings()
                return

            self.send_error_json(404, f"Unknown endpoint: {path}")

        def handle_embed(self) -> None:
            """
            Custom endpoint.

            Request:
                {
                  "text": "hello"
                }

            or:
                {
                  "text": ["hello", "world"],
                  "mode": "query",
                  "normalize": true,
                  "batch_size": 32
                }
            """
            try:
                body = read_json_body(self, max_body_bytes=max_body_bytes)

                if not isinstance(body, dict):
                    raise ValueError("JSON body must be an object.")

                if "text" not in body:
                    raise ValueError("Missing required field: 'text'.")

                texts = parse_text_input(body["text"], field_name="text")

                mode = body.get("mode", "auto")
                if mode not in ("auto", "query", "document"):
                    raise ValueError("'mode' must be one of: auto, query, document.")

                normalize = body.get("normalize", default_normalize)
                if not isinstance(normalize, bool):
                    raise ValueError("'normalize' must be a boolean.")

                batch_size = body.get("batch_size", default_batch_size)
                if not isinstance(batch_size, int) or batch_size < 1:
                    raise ValueError("'batch_size' must be a positive integer.")

                with encode_lock:
                    embeddings = encode_texts(
                        model=model,
                        texts=texts,
                        mode=mode,
                        normalize=normalize,
                        batch_size=batch_size,
                        show_progress=False,
                    )

                self.send_json(
                    200,
                    {
                        "model": model_name,
                        "normalized": normalize,
                        "mode": mode,
                        "count": len(embeddings),
                        "data": [
                            {
                                "index": i,
                                "embedding": embedding,
                            }
                            for i, embedding in enumerate(embeddings)
                        ],
                    },
                )

            except Exception as exc:
                self.send_error_json(400, str(exc))

        def handle_openai_embeddings(self) -> None:
            """
            OpenAI-compatible endpoint.

            Request:
                {
                  "input": "hello world"
                }

            or:
                {
                  "input": ["hello", "world"],
                  "model": "optional"
                }

            Response shape is compatible with /v1/embeddings.
            """
            try:
                body = read_json_body(self, max_body_bytes=max_body_bytes)

                if not isinstance(body, dict):
                    raise ValueError("JSON body must be an object.")

                if "input" not in body:
                    raise ValueError("Missing required field: 'input'.")

                encoding_format = body.get("encoding_format", "float")
                if encoding_format not in (None, "float"):
                    raise ValueError("Only encoding_format='float' is supported.")

                texts = parse_text_input(body["input"], field_name="input")

                with encode_lock:
                    embeddings = encode_texts(
                        model=model,
                        texts=texts,
                        mode="auto",
                        normalize=default_normalize,
                        batch_size=default_batch_size,
                        show_progress=False,
                    )

                estimated_tokens = sum(len(text.split()) for text in texts)

                self.send_json(
                    200,
                    {
                        "object": "list",
                        "model": model_name,
                        "data": [
                            {
                                "object": "embedding",
                                "index": i,
                                "embedding": embedding,
                            }
                            for i, embedding in enumerate(embeddings)
                        ],
                        "usage": {
                            "prompt_tokens": estimated_tokens,
                            "total_tokens": estimated_tokens,
                        },
                    },
                )

            except Exception as exc:
                self.send_error_json(400, str(exc))

    return EmbeddingGemmaHandler


def run_server(args: argparse.Namespace) -> None:
    model = load_model(
        model_name_or_path=args.model,
        device=args.device,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
        local_files_only=args.local_files_only,
    )

    api_key = args.api_key or os.environ.get("EMBEDDINGS_API_KEY")

    handler_class = make_handler(
        model=model,
        model_name=args.model,
        api_key=api_key,
        default_batch_size=args.batch_size,
        default_normalize=not args.no_normalize,
        max_body_bytes=args.max_body_mb * 1024 * 1024,
    )

    server = ThreadingHTTPServer((args.host, args.port), handler_class)

    print(f"Server running at http://{args.host}:{args.port}", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Auth enabled: {bool(api_key)}", file=sys.stderr)
    print("Press Ctrl+C to stop.", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.", file=sys.stderr)
    finally:
        server.server_close()


def parse_input_file(path: str) -> List[str]:
    """
    Accept:
      - plain text file, one non-empty item per line
      - JSON list of strings
      - JSON object with {"input": "..."} or {"input": ["...", "..."]}
      - JSON object with {"text": "..."} or {"text": ["...", "..."]}
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")

    if p.suffix.lower() == ".json":
        data = json.loads(raw)

        if isinstance(data, list):
            return parse_text_input(data, field_name="input")

        if isinstance(data, dict):
            if "input" in data:
                return parse_text_input(data["input"], field_name="input")
            if "text" in data:
                return parse_text_input(data["text"], field_name="text")

        raise ValueError(
            "JSON file must be a list of strings or an object with 'input' or 'text'."
        )

    return [line.strip() for line in raw.splitlines() if line.strip()]


def run_encode(args: argparse.Namespace) -> None:
    model = load_model(
        model_name_or_path=args.model,
        device=args.device,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
        local_files_only=args.local_files_only,
    )

    texts: List[str] = []

    if args.text:
        texts.extend(args.text)

    if args.input_file:
        texts.extend(parse_input_file(args.input_file))

    if not texts:
        raise SystemExit("No input text provided. Use --text or --input-file.")

    embeddings = encode_texts(
        model=model,
        texts=texts,
        mode=args.mode,
        normalize=not args.no_normalize,
        batch_size=args.batch_size,
        show_progress=args.show_progress,
    )

    output = {
        "model": args.model,
        "normalized": not args.no_normalize,
        "mode": args.mode,
        "count": len(embeddings),
        "data": [
            {
                "index": i,
                "text": text,
                "embedding": embedding,
            }
            for i, (text, embedding) in enumerate(zip(texts, embeddings))
        ],
    }

    json_output = json.dumps(output, ensure_ascii=False)

    if args.output_file:
        Path(args.output_file).write_text(json_output, encoding="utf-8")
        print(f"Wrote embeddings to {args.output_file}", file=sys.stderr)
    else:
        print(json_output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run google/embeddinggemma-300m locally without FastAPI or Uvicorn."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)

    common.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model ID or local path. Default: {DEFAULT_MODEL}",
    )

    common.add_argument(
        "--device",
        default=None,
        help="Device to use: cpu, cuda, cuda:0, mps. Default: auto.",
    )

    common.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )

    common.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Optional max sequence length override.",
    )

    common.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size. Default: 32.",
    )

    common.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable embedding normalization.",
    )

    common.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load model files already present locally.",
    )

    serve = subparsers.add_parser(
        "serve",
        parents=[common],
        help="Run local HTTP embeddings server using stdlib http.server.",
    )

    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind. Default: 127.0.0.1",
    )

    serve.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind. Default: 8000",
    )

    serve.add_argument(
        "--api-key",
        default=None,
        help=(
            "Optional bearer token. Can also be set via EMBEDDINGS_API_KEY. "
            "If set, requests must include Authorization: Bearer <key>."
        ),
    )

    serve.add_argument(
        "--max-body-mb",
        type=int,
        default=20,
        help="Maximum JSON request body size in MB. Default: 20.",
    )

    serve.set_defaults(func=run_server)

    encode = subparsers.add_parser(
        "encode",
        parents=[common],
        help="Encode text from the command line.",
    )

    encode.add_argument(
        "--text",
        action="append",
        help="Text to embed. Can be passed multiple times.",
    )

    encode.add_argument(
        "--input-file",
        default=None,
        help="Plain text file, JSON list, or JSON object with input/text field.",
    )

    encode.add_argument(
        "--output-file",
        default=None,
        help="Optional JSON output file.",
    )

    encode.add_argument(
        "--mode",
        choices=["auto", "query", "document"],
        default="auto",
        help="Embedding mode. Default: auto.",
    )

    encode.add_argument(
        "--show-progress",
        action="store_true",
        help="Show SentenceTransformers progress bar.",
    )

    encode.set_defaults(func=run_encode)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
