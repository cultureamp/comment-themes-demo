import glob
import json
from collections import defaultdict
from os import path
from typing import NamedTuple

from flask import Flask, render_template, request, g, session, redirect, url_for
from flask_simpleldap import LDAP

CLUSTER_JSON_SUFF = "-clusters.json"

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
    # LDAP_LOGIN_VIEW = 'sign_in'
    LDAP_OPT_PROTOCOL_VERSION = 3
    LDAP_USER_OBJECT_FILTER = '(&(objectclass=Person)(sAMAccountName=%s))' # for active directory


app = Flask(__name__)

app.config.from_object(DEFAULTS)
app.config.from_pyfile(path.join(path.split(__file__)[0], 'config', 'local.py'), silent=False)
app.secret_key = "NmeZszYsfoRNHtGkCM8YcCJM"

ldap = LDAP(app)


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
    label_mapping = {int(float(k)): v for k, v in _read_json(label_json_path).items()}
    docs_by_theme_id = defaultdict(list)
    for doc in documents:
        if doc['cluster_id'] is None:
            continue
        cluster_dist = 0.0 if doc['cluster_dist'] is None else doc['cluster_dist']
        theme_doc = ThemeDoc(doc['id'], doc['text_val_original'], 1.0 - cluster_dist)
        docs_by_theme_id[int(float(doc['cluster_id']))].append(theme_doc)
    sorted_theme_ids = sorted(docs_by_theme_id)
    themes = [Theme(tid, docs_by_theme_id[tid], label_mapping[tid]) for tid in sorted_theme_ids]
    return ThemeResult(themes)


@app.route('/themes/<configset>/<company>/<survey_id>')
@ldap.group_required(['Comments Prototype Access'])
def show_survey_theme(configset, company, survey_id):
    parent = path.join(cluster_file_root(), configset, company)
    cluster_json = path.join(parent, f"{survey_id}-clusters.json")
    label_json= path.join(parent, f"{survey_id}-cluster_labels.json")
    theme_result = theme_result_from_cluster_json(cluster_json, label_json)
    all_configsets = _list_cluster_configsets(company, survey_id)
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


def _list_cluster_configsets(company, survey_id):
    cluster_root = cluster_file_root()
    matching_json = glob.glob(path.join(cluster_root, '*', company, f"{survey_id}{CLUSTER_JSON_SUFF}"))
    configset_dirs = [_nth_parent(f, 2) for f in matching_json]
    return sorted([path.relpath(csd, cluster_root) for csd in configset_dirs])


def _list_all_survey_ids():
    cluster_root = cluster_file_root()

    def survey_id_and_configset(globmatch):
        relative = path.relpath(globmatch, cluster_root)
        confset_company, survey_json = path.split(relative)
        survey_id = survey_json[:-len(CLUSTER_JSON_SUFF)]
        confset, company = path.split(confset_company)
        return CompSurveyId(company, survey_id), confset

    matching = glob.glob(path.join(cluster_root, '*', '*', f'*{CLUSTER_JSON_SUFF}'))
    survey_ids_to_config_sets = defaultdict(list)
    for m in matching:
        comp_survey_id, confset = survey_id_and_configset(m)
        survey_ids_to_config_sets[comp_survey_id].append(confset)
    return list(survey_ids_to_config_sets.items())


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
    app.run()
