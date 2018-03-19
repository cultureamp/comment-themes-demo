from flask import Flask, render_template
from os import path
import json


class DEFAULTS:
    CLUSTER_FILE_ROOT = './data/clusters'
    LABEL_WHITELIST = {
        'most_common_05'
        'highest_pmi_05',
        'relevance_0_600',
        'summ_basic',
        'summ_luhn',
        'summ_lexrank',
        'summ_textrank',
        'total_words',
        'num_comments',
        'sentiment'
    }


app = Flask(__name__)

app.config.from_object(DEFAULTS)
app.config.from_pyfile(path.join(path.split(__file__)[0], 'localconfig.py'), silent=False)

from collections import defaultdict


def cluster_file_root():
    return app.config['CLUSTER_FILE_ROOT']


def label_whitelist():
    return app.config['LABEL_WHITELIST']


class Theme:
    label_whitelist = label_whitelist()

    def __init__(self, id, documents, labels):
        self.id = id
        self.documents = documents
        self.labels = {name: labels[name] for name in labels if name in self.label_whitelist}


class ThemeDoc:
    def __init__(self, id, text, closeness):
        self.id = id
        self.text = text
        self.closeness = closeness


class ThemeResult:
    def __init__(self, themes):
        self.themes = themes


def _read_json(json_path):
    with open(json_path) as f:
        return json.load(f)


def theme_result_from_cluster_json(clustered_doc_json_path, label_json_path):
    documents = _read_json(clustered_doc_json_path)
    label_mapping = _read_json(label_json_path)
    docs_by_theme_id = defaultdict(list)
    for doc in documents:
        theme_doc = ThemeDoc(doc['id'], doc['text_val_original'], 1.0 - doc['cluster_dist'])
        docs_by_theme_id[doc['cluster_id']].append(theme_doc)
    sorted_theme_ids = sorted(docs_by_theme_id)
    themes = [Theme(tid, docs_by_theme_id[tid], label_mapping[str(tid)]) for tid in sorted_theme_ids]
    return ThemeResult(themes)


@app.route('/themes/<configset>/<company>/<survey_id>')
def show_survey_theme(configset, company, survey_id):
    parent = path.join(cluster_file_root(), configset, company)
    cluster_json = path.join(parent, f"{survey_id}-clusters.json")
    label_json= path.join(parent, f"{survey_id}-cluster_labels.json")
    theme_result = theme_result_from_cluster_json(cluster_json, label_json)
    return render_template("show-themes.html", theme_result=theme_result)


@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
