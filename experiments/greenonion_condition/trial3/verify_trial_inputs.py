"""Strict, read-only preflight for a green-onion trial.

Run with --photos CSV --root DIR --cache NPZ --model PTH --extractor PY.
CSV columns: path, sha256, condition, group. The frozen CSV must contain no test rows.
NPZ fields: ids, X, b, sha256 (aligned with ids), list_sha256, model_sha256,
extractor_sha256. Old caches without provenance intentionally fail: regenerate
them from the current source photos; never add provenance to an old cache.
This validates provenance records, not the truth of image labels or licensing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(photos: Path, root: Path, cache: Path, model: Path, extractor: Path) -> dict:
    errors: list[str] = []
    root = root.resolve()
    with photos.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'path', 'sha256', 'condition', 'group'}
        if not required.issubset(reader.fieldnames or []):
            return {'ok': False, 'errors': ['CSV requires path, sha256, condition, group']}
        rows = list(reader)
    if not rows:
        errors.append('photo list is empty')
    expected: dict[str, str] = {}
    seen_paths: set[str] = set()
    for row in rows:
        name = row['path']
        target = (root / name).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            errors.append(f'outside root: {name}')
            continue
        key = str(target).casefold()
        if key in seen_paths:
            errors.append(f'duplicate photo path: {name}')
        seen_paths.add(key)
        if row['condition'] not in {'fresh', 'spoiled'} or not row['group'].strip():
            errors.append(f'invalid condition/group: {name}')
        split = row.get('split', '').casefold()
        origin = row.get('origin', '').casefold()
        if (split.startswith('test') or origin.startswith(('test', 'user_test'))
                or any(part.casefold().startswith('test') for part in Path(name).parts)):
            errors.append(f'test photo in development list: {name}')
        sha = row['sha256'].lower()
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            errors.append(f'invalid SHA256: {name}')
        expected[name] = sha
        if not target.is_file():
            errors.append(f'missing photo: {name}')
        elif digest(target) != sha:
            errors.append(f'photo changed: {name}')
    with np.load(cache, allow_pickle=False) as features:
        required = {'ids', 'X', 'b', 'sha256', 'list_sha256', 'model_sha256', 'extractor_sha256'}
        missing = required - set(features.files)
        if missing:
            errors.append('cache missing provenance/arrays: ' + ', '.join(sorted(missing)))
        else:
            for key, path in [('list_sha256', photos), ('model_sha256', model), ('extractor_sha256', extractor)]:
                value = features[key]
                if value.size != 1 or str(value.reshape(-1)[0]).lower() != digest(path):
                    errors.append(f'cache {key} mismatch')
            ids_array, hashes = features['ids'], features['sha256']
            if ids_array.ndim != 1 or hashes.ndim != 1:
                errors.append('cache ids/sha256 must be one-dimensional')
            else:
                ids = [str(x) for x in ids_array]
                if len(ids) != len(set(ids)):
                    errors.append('duplicate cache ids')
                if set(ids) != set(expected):
                    errors.append('cache/list photo sets differ')
                if len(hashes) != len(ids):
                    errors.append('cache sha256 length mismatch')
                else:
                    for name, sha in zip(ids, hashes):
                        if str(sha).lower() != expected.get(name):
                            errors.append(f'cache photo SHA256 mismatch: {name}')
                for key in ('X', 'b'):
                    array = features[key]
                    if (array.ndim != 2 or array.shape[0] != len(ids)
                            or array.shape[1] == 0 or not np.issubdtype(array.dtype, np.number)
                            or not np.isfinite(array).all()):
                        errors.append(f'invalid feature array: {key}')
    return {'ok': not errors, 'n_photos': len(rows), 'errors': errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('photos', 'root', 'cache', 'model', 'extractor'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(**vars(args))
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {'ok': False, 'errors': [f'{type(error).__name__}: {error}']}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
