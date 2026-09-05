"""Read-only local fixture viewer; never connects to a remote database.

streamlit run scripts/view_forward_test_fixture.py -- --database /path/fixture.sqlite
"""
import argparse
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from range_finder.forward_test.store import Store
from range_finder.forward_test.results import load_results
from ui_forward_test import render_forward_test

parser = argparse.ArgumentParser()
parser.add_argument('--database', required=True)
args = parser.parse_args()
path = Path(args.database).resolve(strict=True)
store = Store(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True))
try:
    rows = load_results(store, 'DEMO_ONLY')
    if not rows or any(r['cohort'] != 'deterministic_fixture' for r in rows):
        raise ValueError('This viewer requires the deterministic DEMO_ONLY fixture')
    runs = store.query('SELECT * FROM ft_runs ORDER BY started_at DESC')
finally:
    store.close()
st.set_page_config(page_title='Gamma Lens fixture', layout='wide')
st.info('Deterministic fixture data. Synthetic prices and outcomes; excluded from the prospective study.')
render_forward_test(rows, runs)
