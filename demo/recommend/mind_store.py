"""Serve MIND news and rank the saved representations from the MIND run.

Only load trusted, local pickle artifacts produced by this project.
"""
import csv
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from src.recommendation.RecommendationEngine import RecommendationEngine
from src.urv.URV import URV


class MindStore:
    def __init__(self, data_dir=None, vector_dir=None):
        root = Path(__file__).resolve().parents[2]
        data = Path(data_dir or os.getenv('MIND_DATA_DIR', root / 'data/raw'))
        vectors = Path(vector_dir or os.getenv('MIND_VECTOR_DIR', root / 'vectors/mind_small'))
        self.verification = None
        verification_file = os.getenv('MIND_VERIFICATION_FILE')
        if verification_file:
            report = json.loads(Path(verification_file).read_text())
            checkpoint = Path(os.getenv('MIND_MODEL_FILE', root / report['model']))
            for path, key in [(checkpoint, 'model_sha256'),
                              (vectors / 'represent_vectors.pkl', 'vectors_sha256')]:
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                if digest.hexdigest() != report[key] or not report['matched']:
                    raise ValueError('MIND artifact does not match checkpoint verification report')
            self.verification = report
        self.manifest = json.loads((vectors / 'manifest.json').read_text())
        digest = hashlib.sha256((data / 'news.tsv').read_bytes()).hexdigest()
        self.split = next((split for split in ('train', 'dev')
                           if digest == self.manifest.get(f'{split}_news_sha256')), None)
        if not self.split:
            raise ValueError('news.tsv does not match the MIND vector manifest')
        self.news = {}
        with (data / 'news.tsv').open(encoding='utf-8') as stream:
            for row in csv.reader(stream, delimiter='\t', quoting=csv.QUOTE_NONE):
                if len(row) < 8:
                    raise ValueError('Invalid MIND news row')
                key, category, subcategory, title, abstract, url = row[:6]
                self.news[key] = dict(news_id=key, category=category, subcategory=subcategory,
                                      title=title, abstract=abstract, url=url)
        with (vectors / 'represent_vectors.pkl').open('rb') as stream:
            self.vectors = pickle.load(stream)
        for key, article in self.news.items():
            vector = self.vectors.get(key)
            # The training TSV parser normalizes some quoted titles. IDs and the
            # source file SHA-256 identify the original articles without rewriting them.
            if vector is None:
                raise ValueError(f'Missing or mismatched representation: {key}')
            for field, size in [('semantic', 384), ('topic_distribution', self.manifest['topic_dimensions'])]:
                value = np.asarray(vector[field])
                if value.shape != (size,) or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
                    raise ValueError(f'Invalid {field} vector: {key}')
        self.impressions = {}
        self.skipped = 0
        with (data / 'behaviors.tsv').open(encoding='utf-8') as stream:
            for row in csv.reader(stream, delimiter='\t', quoting=csv.QUOTE_NONE):
                if len(row) != 5:
                    raise ValueError('Invalid MIND behavior row')
                key, user, timestamp, history, candidates = row
                history_ids = history.split()
                parsed = [item.rsplit('-', 1) for item in candidates.split()]
                if (not history_ids or not parsed or
                        any(len(pair) != 2 or pair[1] not in ('0', '1') for pair in parsed) or
                        any(n not in self.news for n in history_ids) or
                        any(pair[0] not in self.news for pair in parsed)):
                    self.skipped += 1
                    continue
                self.impressions[key] = dict(impression_id=key, user_id=user, timestamp=timestamp,
                    history_ids=history_ids, candidates=[(n, int(label)) for n, label in parsed])
        self.rows = list(self.impressions.values())
        self.categories = sorted({n['category'] for n in self.news.values()})
        if not self.rows:
            raise ValueError('No complete MIND impressions with nonempty history')

    def metadata(self):
        return dict(dataset='MIND Small', split=self.split, news_count=len(self.news),
                    impression_count=len(self.rows), skipped_impressions=self.skipped,
                    vector_count=len(self.vectors), categories=self.categories,
                    sentence_model=self.manifest['sentence_model'],
                    topic_dimensions=self.manifest['topic_dimensions'], alpha=0.75,
                    representation=self.manifest['topic_representation'],
                    scoring_mode='saved_mind_vectors',
                    checkpoint_verification=self.verification,
                    source='vectors/mind_small/represent_vectors.pkl')

    def impressions_page(self, offset, limit, user=''):
        rows = self.rows if not user else [r for r in self.rows if r['user_id'].lower() == user.lower()]
        return dict(total=len(rows), offset=offset, items=[dict(
            impression_id=r['impression_id'], user_id=r['user_id'], timestamp=r['timestamp'],
            history_count=len(r['history_ids']), candidate_count=len(r['candidates']))
            for r in rows[offset:offset + limit]])

    def impression(self, key):
        row = self.impressions[key]
        return dict(impression_id=key, user_id=row['user_id'], timestamp=row['timestamp'],
            history=[self.news[n] for n in row['history_ids']],
            candidates=[dict(self.news[n], clicked=label) for n, label in row['candidates']])

    def rank(self, key):
        row = self.impressions[key]
        urv = URV().getURVFromVector([self.vectors[n] for n in row['history_ids']])
        # Labels are joined only after scoring; they never enter the model.
        candidates = [dict(self.vectors[n], title=n) for n, _ in row['candidates']]
        scores = RecommendationEngine(alpha=0.75).recommended_from_vectors(urv, candidates)
        labels = dict(row['candidates'])
        return dict(impression_id=key, results=[dict(self.news[r['news_id']], **{
            k: v for k, v in r.items() if k != 'news_id'}, clicked=labels[r['news_id']]) for r in scores])
