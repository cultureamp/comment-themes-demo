import glob
import json
import pickle
import pandas as pd

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from os import path
from typing import NamedTuple, List

from flask import Flask, render_template, request, g, session, redirect, url_for
from flask_simpleldap import LDAP
from sklearn.metrics import adjusted_mutual_info_score

CLUSTER_JSON_SUFF = "-clusters.json"
TOPIC_PICKLE_SUFF = "-topicmodel.pickle"

class DEFAULTS:
    CLUSTER_FILE_ROOTS = { 'all': './data/clusters' }
    TM_FILE_ROOTS = { 'all': './data/topic-models' }

    LABEL_WHITELIST = [
        'most_common_05',
        'most_common_10',
        'highest_pmi_05',
        'highest_pmi_10',
        'relevance_0_600',
        'summ_basic',
        'summ_luhn',
        'summ_lexrank',
        'summ_textrank',
        'total_words',
        'num_comments',
        'sentiment',
        'question_distrib',
        'question_hist',
    ]
    SUMM_LABEL_WHITELIST = [
        'num_comments',
        'sentiment',
        'most_common_05',
        'most_common_10',
        'summ_basic',
    ]
    # LDAP_LOGIN_VIEW = 'sign_in'
    LDAP_OPT_PROTOCOL_VERSION = 3
    LDAP_USER_OBJECT_FILTER = '(&(objectclass=Person)(sAMAccountName=%s))' # for active directory
    LDAP_HOST = 'cultureamp.net'
    LDAP_BASE_DN = "OU=Users,OU=cultureamp,DC=cultureamp,DC=net"


app = Flask(__name__)

app.config.from_object(DEFAULTS)
app.config.from_pyfile(path.join(path.split(__file__)[0], 'config', 'local.py'), silent=False)
app.secret_key = "NmeZszYsfoRNHtGkCM8YcCJM"

ldap = LDAP(app)


def label_whitelist():
    return app.config['LABEL_WHITELIST']


def summ_label_whitelist():
    return app.config['SUMM_LABEL_WHITELIST']


class Theme:
    label_whitelist = label_whitelist()
    summ_label_whitelist = summ_label_whitelist()

    def __init__(self, id, documents, labels):
        self.id = id
        self.documents = documents
        self._counts_by_q = None
        self._store_counts_by_q()
        labels['question_distrib'] = self._question_distribution()
        labels['question_hist'] = self._question_histogram()
        self.labels = [(name, labels[name]) for name in self.label_whitelist if name in labels]
        self.summ_labels = [(name, labels[name]) for name in self.summ_label_whitelist if name in labels]

    def _store_counts_by_q(self):
        self._counts_by_q = defaultdict(int)
        for doc in self.documents:
            self._counts_by_q[doc.question_id] += 1

    def _question_histogram(self):
        values = [(qid[-4:], ct) for qid, ct in self._counts_by_q.items() if ct > 1]
        singletons = sum(1 for ct in self._counts_by_q.values() if ct == 1)
        values.sort(key=lambda x: x[1], reverse=True)
        return '\n'.join(f"{k}:{'#' * v}" for k, v in values + [('SING', singletons)])

    def _question_distribution(self):
        counts_by_q = self._counts_by_q
        total = len(self.documents)
        if total == 0:
            return ""
        values = {qid[-4:]: f"{ct / total:.3f} [{ct}]" for qid, ct in counts_by_q.items() if ct > 1}
        singletons = sum(1 for ct in counts_by_q.values() if ct == 1)
        non_single = [f"{k}: {v}" for k, v in sorted(values.items(), key=lambda x: x[1], reverse=True)]
        return ", ".join(non_single + [f"SINGLETONS: {singletons / total :.3f} [{singletons}]"])


class ThemeDoc:
    def __init__(self, id, text, closeness, question_id):
        self.id = id
        self.text = text
        self.closeness = closeness
        self.question_id = question_id


class ThemeResult:
    def __init__(self, themes: List[Theme]):
        self.themes = themes
        self._decompose_themes()

    def _decompose_themes(self):
        self._question_labels = []
        self._theme_ids = []
        for theme in self.themes:
            self._question_labels.extend(td.question_id for td in theme.documents)
            self._theme_ids.extend([theme.id] * len(theme.documents))

    @property
    def purity(self):
        return purity(self._question_labels, self._theme_ids)

    @property
    def adj_nmi(self):
        return adjusted_mutual_info_score(self._question_labels, self._theme_ids)

    @property
    def total_comments(self):
        return sum(len(th.documents) for th in self.themes)

    @property
    def num_themes(self):
        return len(self.themes)

    @property
    def largest_theme_proportion(self):
        return max(len(th.documents) for th in self.themes) / self.total_comments

    @property
    def smallest_theme_proportion(self):
        return min(len(th.documents) for th in self.themes) / self.total_comments


def purity(labels_true, clusters_pred):
    assert len(labels_true) == len(clusters_pred)
    assigns = pd.DataFrame({'label': labels_true, 'cluster': clusters_pred})
    cluster_ids = assigns['cluster'].unique()
    total_in_max_in_cluster = 0
    for cid in cluster_ids:
        members = assigns.loc[assigns['cluster'] == cid]
        counts = members.groupby('label')['label'].value_counts()
        total_in_max_in_cluster += max(counts)
    return total_in_max_in_cluster / len(assigns)


def _read_json(json_path):
    with open(json_path) as f:
        return json.load(f)


@app.route('/themes/<category>/<configset>/<company>/<survey_id>')
@ldap.group_required(['Comments Prototype Access'])
def show_survey_theme(configset, category, company, survey_id):
    stores = []
    for store_by_cat in (CLUSTER_STORES, TOPIC_MODEL_STORES):
        try:
            stores.append(store_by_cat[category])
        except KeyError:
            pass
    def all_configsets():
        for st in stores:
            for cs in st._list_configsets(company, survey_id):
                yield cs
    theme_result = None
    for st in stores:
        try:
            theme_result = st.theme_result(configset, company, survey_id)
        except FileNotFoundError: # only one store will work
            continue
        break
    metric_keys = ('total_comments', 'num_themes', 'adj_nmi', 'purity',
        'largest_theme_proportion', 'smallest_theme_proportion')

    metrics = [(key, getattr(theme_result, key)) for key in metric_keys]
    return render_template("show-themes.html", theme_result=theme_result,
            all_configsets=list(all_configsets()), this_configset=configset,
            metrics=metrics)


def _show_survey_theme_for_store(store: 'ThemeStore', configset, company, survey_id):
    all_configsets = store._list_configsets(company, survey_id)
    theme_result = store.theme_result(configset, company, survey_id)
    return render_template("show-themes.html", theme_result=theme_result,
            all_configsets=all_configsets, this_configset=configset)




@app.route('/')
@ldap.login_required
def index():
    return render_template("index.html", comp_si_confsets=_list_all_survey_ids())


@app.before_request
def before_request():
    g.user = None
    if 'user_id' in session:
        # This is where you'd query your database to get the user info.
        g.user = {'ldapuser': session['user_id']}
        # Create a global with the LDAP groups the user is a member of.
        g.ldap_groups = ldap.get_user_groups(user=session['user_id'])


def _clean_up_user(input_user):
    extraneous = '@cultureamp.com'
    if input_user.endswith(extraneous):
        return input_user[:-len(extraneous)]
    else:
        return input_user


@app.route('/login', methods=['GET', 'POST'])
def login():
    if g.user:
        return redirect(url_for('index'))
    if request.method == 'POST':
        user = _clean_up_user(request.form['user'])
        passwd = request.form['passwd']
        test = ldap.bind_user(user, passwd)
        if test is None or passwd == '':
            return 'Invalid credentials'
        else:
            session['user_id'] = user
            return redirect(request.args.get('next', '/'))
    return render_template("login.html")


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))


class CompSurveyId(NamedTuple):
    company: str
    survey_id: str


class ThemeStore(metaclass=ABCMeta):
    def __init__(self, store_root):
        self.store_root = store_root

    def _list_configsets(self, company, survey_id):
        matching_json = glob.glob(path.join(self.store_root, '*', company,
                f"{survey_id}{self._main_theme_file_suffix()}"))
        configset_dirs = [_nth_parent(f, 2) for f in matching_json]
        return sorted([path.relpath(csd, self.store_root) for csd in configset_dirs])

    @abstractmethod
    def _main_theme_file_suffix(self):
        pass

    def all_survey_ids(self):
        def survey_id_and_configset(globmatch):
            relative = path.relpath(globmatch, self.store_root)
            confset_company, survey_json = path.split(relative)
            survey_id = survey_json[:-len(self._main_theme_file_suffix())]
            confset, company = path.split(confset_company)
            return CompSurveyId(company, survey_id), confset

        matching = glob.glob(path.join(self.store_root, '*', '*', f'*{self._main_theme_file_suffix()}'))
        survey_ids_to_config_sets = defaultdict(list)
        for m in matching:
            comp_survey_id, confset = survey_id_and_configset(m)
            survey_ids_to_config_sets[comp_survey_id].append(confset)
        return survey_ids_to_config_sets

    @abstractmethod
    def theme_result(self, configset, company, survey_id) -> ThemeResult:
        pass


class TMStore(ThemeStore):

    def _main_theme_file_suffix(self):
        return TOPIC_PICKLE_SUFF

    def theme_result(self, configset, company, survey_id) -> ThemeResult:
        parent = path.join(self.store_root, configset, company)
        tm_pickle = path.join(parent, f"{survey_id}{self._main_theme_file_suffix()}")
        label_json = path.join(parent, f"{survey_id}-cluster_labels.json")
        return TMStore._theme_result_from_stored(tm_pickle, label_json)

    @staticmethod
    def _theme_result_from_stored(tm_pickle_path, label_json_path):
        with open(tm_pickle_path, 'rb') as f:
            tmr = pickle.load(f)
        label_mapping = {int(v['topic_id']): v for k, v in _read_json(label_json_path).items()}
        def theme_docs(topic):
            for tmdoc in topic.documents(0.3):
                yield ThemeDoc(tmdoc.id, tmdoc.raw_text, tmdoc.topic_proportion, tmdoc.raw_data['question_id'])
        themes = [Theme(t.index, list(theme_docs(t)), label_mapping.get(t.index, {})) for t in tmr.all_topics()]
        return ThemeResult(themes)


class ClusterStore(ThemeStore):

    def _main_theme_file_suffix(self):
        return CLUSTER_JSON_SUFF

    def theme_result(self, configset, company, survey_id) -> ThemeResult:
        parent = path.join(self.store_root, configset, company)
        cluster_json = path.join(parent, f"{survey_id}-clusters.json")
        label_json = path.join(parent, f"{survey_id}-cluster_labels.json")
        return ClusterStore._theme_result_from_cluster_json(cluster_json, label_json)

    @staticmethod
    def _theme_result_from_cluster_json(clustered_doc_json_path, label_json_path):
        documents = _read_json(clustered_doc_json_path)
        label_mapping = {int(float(v['cluster_id'])): v for k, v in _read_json(label_json_path).items()}
        docs_by_theme_id = defaultdict(list)
        for doc in documents:
            if doc['cluster_id'] is None:
                continue
            cluster_dist = 0.0 if doc['cluster_dist'] is None else doc['cluster_dist']
            theme_doc = ThemeDoc(doc['id'], doc['text_val_original'], 1.0 - cluster_dist, doc['question_id'])
            docs_by_theme_id[int(float(doc['cluster_id']))].append(theme_doc)
        sorted_theme_ids = sorted(docs_by_theme_id)
        themes = [Theme(tid, docs_by_theme_id.get(tid, []), label_mapping.get(tid, "NA")) for tid in sorted_theme_ids]
        return ThemeResult(themes)


CLUSTER_STORES = {cat: ClusterStore(cfr) for cat, cfr in app.config['CLUSTER_FILE_ROOTS'].items()}

TOPIC_MODEL_STORES = {cat: TMStore(tfr) for cat, tfr in app.config['TM_FILE_ROOTS'].items()}


def _list_all_survey_ids():
    all_survs = defaultdict(list)
    for cat, st in list(CLUSTER_STORES.items()) + list(TOPIC_MODEL_STORES.items()):
        for key, vals in st.all_survey_ids().items():
            all_survs[cat, key].extend(sorted(vals))
    return all_survs.items()


def _nth_parent(target, level=1):
    npar = target
    while level >= 1:
        npar = path.dirname(npar)
        level -= 1
    return npar


# @app.route('/')
# def hello_world():
#     return 'Hello World!'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default='127.0.0.1')
    parser.add_argument("-p", "--port", default=5000, type=int)
    parser.add_argument("-d", "--debug", default=False, type=bool)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
