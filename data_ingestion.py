# data_ingestion.py

import os
import mimetypes
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import csv
import json
import xml.etree.ElementTree as ET
import markdown

class FileHandler:
    def read_file(self, file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)

        if mime_type is None:
            raise ValueError(f"Unknown file type: {file_path}")

        if mime_type.startswith("text"):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        elif mime_type.endswith("xml"):
            return self._read_xml(file_path)
        elif mime_type.endswith("json"):
            return self._read_json(file_path)
        elif mime_type.endswith("csv"):
            return self._read_csv(file_path)
        elif mime_type.endswith("md"):
            return self._read_markdown(file_path)
        elif mime_type.endswith("py"):
            return self._read_python(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _read_xml(self, file_path: str) -> str:
        tree = ET.parse(file_path)
        return ET.tostring(tree.getroot(), encoding='unicode')

    def _read_json(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.dumps(json.load(file), indent=2)

    def _read_csv(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            return '\n'.join(','.join(row) for row in reader)

    def _read_markdown(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return markdown.markdown(content)
    
    def _read_python(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

class TextChunker:
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

class MetadataGenerator:
    def generate_metadata(self, file_path: str) -> Dict[str, Any]:
        file_stats = os.stat(file_path)
        return {
            "file_name": os.path.basename(file_path),
            "file_size": file_stats.st_size,
            "last_modified": file_stats.st_mtime,
        }

    def generate_ids(self, chunks: List[str], file_path: str) -> List[str]:
        return [hashlib.md5(f"{file_path}_{i}".encode()).hexdigest() for i in range(len(chunks))]
