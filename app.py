"""Ember: an auditable, interactive Bayesian wildfire study."""
from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from wildfire.data import MONTHS, feature_frame, validate_inputs, load_data

st.set_page_config(page_title='Ember | Wildfire Intelligence', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp {background: #f7f8f4;}
.block-container {max-width: 1440px; padding-top: 2rem; padding-bottom: 2rem;}
[data-testid="stSidebar"] {background: #eaf0e9; border-right: 1px solid #dce4da;}
h1,h2,h3 {letter-spacing: -.035em; color: #193d32;}
[data-testid="stMetric"] {background: #fff; border: 1px solid #dfe6dc; border-radius: 16px; padding: 18px;}
[data-testid="stMetricLabel"] {color: #52695e;}
[data-testid="stMetricValue"] {color: #204c3c; letter-spacing: -.04em;}
[data-testid="stForm"] {background: #fff; border: 1px solid #dfe6dc; border-radius: 18px; padding: 22px;}
.stTabs [data-baseweb="tab-list"] {gap: 24px; border-bottom: 1px solid #dce4da;}
.stTabs [data-baseweb="tab"] {font-weight: 600; padding-bottom: 16px;}
.ember-hero {position: relative; overflow: hidden; background: #173e32; border-radius: 22px;
 padding: 34px 38px; margin: 12px 0 24px; color: #f9fbf5;}
.ember-hero h1 {font-size: clamp(30px, 3.7vw, 48px); line-height: 1.12; font-weight: 650;
 color: #f9fbf5; margin: 12px 0; max-width: 700px;}
.ember-hero p {color: #c8d9cb; max-width: 590px; font-size: 16px; margin-bottom: 6px;}
.eyebrow {font-size: 11px; letter-spacing: .17em; text-transform: uppercase; color: #adc8af; font-weight: 650;}
.ember-brand {font-size: 35px; font-weight: 750; color: #193e32; letter-spacing: -.06em;}
.brand-dot {color: #cf653b;}
.hero-art {position: absolute; right: 3%; bottom: -12px; opacity: .30; width: 235px;}
.section-tag {font-size: 11px; letter-spacing: .16em; font-weight: 650; color: #7a8b76; margin-top: 8px;}
.footer {border-top: 1px solid #dce4da; color: #748174; font-size: 12px; padding-top: 16px; margin-top: 30px;}
@media(max-width: 900px) {.hero-art {display:none;} .ember-hero {padding:24px;} .block-container {padding-top:1rem;}}
</style>''', unsafe_allow_html=True)


@st.cache_resource
def load_bundle(artifact_mtime):
    # Only trusted, locally trained artifacts. Never deserialize user uploads.
    return joblib.load(ROOT / 'artifacts/models.joblib')


@st.cache_data
def read_table(name, artifact_mtime):
    return pd.read_csv(ROOT / 'artifacts' / name)


def table(name):
    return read_table(name, (ROOT / 'artifacts' / name).stat().st_mtime_ns)


def range_messages(frame, bundle):
    messages = []
    for col, bounds in bundle['training_ranges'].items():
        outside = ~frame[col].between(bounds['min'], bounds['max'])
        if outside.any():
            messages.append(f"{col}: {outside.sum()} row(s) outside training range {bounds['min']:g}–{bounds['max']:g}")
    if not frame.month.isin(bundle['training_months']).all():
        messages.append('Some months were absent from the training split.')
    return messages


artifact = ROOT / 'artifacts/models.joblib'
if not artifact.exists():
    st.error('Models are missing. From the project folder run: python -m wildfire.train')
    st.stop()
bundle = load_bundle(artifact.stat().st_mtime_ns)
models = bundle['models']
results = table('model_comparison.csv')

with st.sidebar:
    st.markdown('<div class="ember-brand">ember<span class="brand-dot">.</span></div>', unsafe_allow_html=True)
    st.caption('A clearer view of model uncertainty.')
    st.divider()
    st.markdown('<div class="section-tag">RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
    selected = st.selectbox('Probability model', list(models), key='prediction_model')
    st.caption('Bayesian Model Averaging is the primary algorithm. All models use the same inputs and test records.')
    st.markdown(f"**Decision threshold** · {bundle['thresholds'][selected]:.2f}")
    st.caption('Selected on validation F1. Not a firefighting alert level.')
    st.divider()
    st.markdown('**Montesinho Natural Park**  \nPortugal · UCI Forest Fires')
    st.caption(f"{bundle['metadata']['rows']} records · 9 scenario inputs · 16 Bayesian candidates")
    st.link_button('View GitHub repository', 'https://github.com/RomitDeokar/wildfire_risk_ml', use_container_width=True)
    with st.expander('Reproducibility fingerprint'):
        st.code(bundle['metadata']['data_sha256'], wrap_lines=True)
        st.caption(f"Trained on Python {bundle['metadata']['python']}")

st.markdown('<div class="section-tag">UNIT 05 &nbsp; / &nbsp; PROJECT 03 &nbsp; / &nbsp; ENVIRONMENTAL MACHINE LEARNING</div>', unsafe_allow_html=True)
st.markdown('''<div class="ember-hero">
<div class="eyebrow">Bayesian modeling · transparent evidence</div>
<h1>Explore conditions.<br>Understand the uncertainty.</h1>
<p>A comparative wildfire study that brings weather, Bayesian model averaging, and honest evaluation into one workspace.</p>
<svg class="hero-art" viewBox="0 0 240 220" aria-hidden="true">
<circle cx="180" cy="48" r="25" fill="#ecad68"/>
<path d="M30 200L74 70L118 200Z" fill="#b7cfa0"/><path d="M86 210L144 25L202 210Z" fill="#d3dfb2"/>
<path d="M165 210L199 105L233 210Z" fill="#84aa87"/><path d="M0 211Q100 180 240 211V230H0Z" fill="#8bad85"/>
</svg></div>''', unsafe_allow_html=True)
st.info('Research scope: P(recorded burned area > 0 | recorded fire, conditions). '
        'Area = 0 is not a confirmed non-fire day. These are educational proxy probabilities, not daily ignition forecasts or emergency advice.')

predict_tab, batch_tab, compare_tab, weights_tab, data_tab, methods_tab = st.tabs(
    ['Scenario studio', 'Batch scoring', 'Model performance', 'Bayesian evidence', 'Dataset explorer', 'Methods & downloads'])

with predict_tab:
    st.subheader('One scenario. Five perspectives.')
    st.caption('Adjust environmental measurements, then estimate. The initial result uses the default scenario shown below.')
    inputs, output = st.columns([1.1, 1], gap='large')
    with inputs:
        with st.form('scenario_form'):
            st.markdown('**Environmental conditions**')
            a, b = st.columns(2)
            temp = a.number_input('Temperature (°C)', value=20.0, step=.5)
            rh = b.number_input('Relative humidity (%)', min_value=0.0, max_value=100.0, value=45.0)
            wind = a.number_input('Wind speed (km/h)', min_value=0.0, value=4.0, step=.1)
            rain = b.number_input('Rain (mm/m², UCI units)', min_value=0.0, value=0.0, step=.1)
            month = a.selectbox('Month', MONTHS, index=7)
            st.markdown('**Fire Weather Index components**')
            st.caption('Use externally computed indices; they are not derived from one day of weather.')
            ffmc = a.number_input('FFMC · fine fuel moisture', min_value=0.0, max_value=101.0, value=90.0)
            dmc = b.number_input('DMC · duff moisture', min_value=0.0, value=110.0)
            dc = a.number_input('DC · drought code', min_value=0.0, value=550.0)
            isi = b.number_input('ISI · initial spread index', min_value=0.0, value=8.0)
            submitted = st.form_submit_button('Estimate probability', type='primary', use_container_width=True)
    current = pd.DataFrame([dict(temp=temp, RH=rh, wind=wind, rain=rain, FFMC=ffmc,
                                 DMC=dmc, DC=dc, ISI=isi, month=month)])
    if submitted or 'scenario' not in st.session_state:
        st.session_state['scenario'] = current
    with output:
        try:
            raw = validate_inputs(st.session_state['scenario'])
        except (ValueError, TypeError) as exc:
            st.error(f'Check scenario inputs: {exc}')
            st.stop()
        X = feature_frame(raw)
        p = float(models[selected].predict_proba(X)[0, 1])
        st.markdown(f'**{selected}** · scenario estimate')
        st.metric('Positive recorded-area probability', f'{p:.1%}')
        st.progress(p)
        threshold = bundle['thresholds'][selected]
        st.caption(f"Predicted class: {'area > 0' if p >= threshold else 'area = 0'} · threshold {threshold:.2f}")
        bma_pipe = models['BMA']; bma = bma_pipe.named_steps['model']
        scaled = bma_pipe.named_steps['scale'].transform(X)
        low, high = bma.probability_interval(scaled)[0]
        candidate_p = bma.candidate_probabilities(scaled)[0]
        mean = float(candidate_p @ bma.weights_)
        spread = float(np.sqrt(np.sum(bma.weights_ * (candidate_p - mean)**2)))
        st.caption(f'BMA 90% approximate credible interval: {low:.1%}–{high:.1%} · candidate-mean SD {spread:.3f}')
        comparison = pd.DataFrame({'Model': list(models), 'Probability':
            [float(m.predict_proba(X)[0, 1]) for m in models.values()]})
        st.bar_chart(comparison.set_index('Model'), horizontal=True, color='#39765c', x_label='Probability', height=235)
        st.caption('Interval includes model-index and coefficient uncertainty under the Laplace approximation. '
                   'It does not include dataset bias, unseen conditions, or real-world ignition uncertainty.')
        for message in range_messages(raw, bundle):
            st.warning(message)
        snapshot = {'inputs': raw.iloc[0].to_dict(), 'probabilities': dict(zip(comparison.Model, comparison.Probability)),
                    'bma_probability_interval_90': [float(low), float(high)], 'selected_model': selected,
                    'threshold': threshold, 'scope': bundle['metadata']['target']}
        st.download_button('Save scenario results', json.dumps(snapshot, indent=2), 'ember_scenario.json', 'application/json')
    with st.expander('Explore sensitivity to a single input'):
        st.caption('Model response with other inputs fixed. Not a causal effect; combinations may be unobserved.')
        feature = st.selectbox('Vary input', ['temp', 'RH', 'wind', 'DMC', 'DC', 'FFMC', 'ISI'])
        bounds = bundle['training_ranges'][feature]
        grid = np.linspace(bounds['min'], bounds['max'], 40)
        scenarios = pd.concat([raw] * len(grid), ignore_index=True)
        scenarios[feature] = grid
        response = models[selected].predict_proba(feature_frame(scenarios))[:, 1]
        st.line_chart(pd.DataFrame({feature: grid, 'Probability': response}).set_index(feature), color='#39765c')

with batch_tab:
    st.subheader('From one scenario to a whole dataset')
    st.caption('Validated CSV scoring · up to 5,000 rows · no uploaded files are stored by the app')
    sample = pd.DataFrame([dict(temp=20, RH=45, wind=4, rain=0, FFMC=90, DMC=110, DC=550, ISI=8, month='aug')])
    st.download_button('Download CSV template', sample.to_csv(index=False), 'scenario_template.csv', 'text/csv')
    uploaded = st.file_uploader('Upload CSV with the nine template columns', type='csv')
    if uploaded is not None:
        try:
            if uploaded.size > 2_000_000:
                raise ValueError('CSV must be smaller than 2 MB.')
            batch = pd.read_csv(uploaded)
            if not 1 <= len(batch) <= 5000:
                raise ValueError('Upload between 1 and 5,000 rows.')
            batch = validate_inputs(batch); features = feature_frame(batch); scored = batch.copy()
            for name, model in models.items():
                scored[name + '_probability'] = model.predict_proba(features)[:, 1]
            scored['selected_model_class'] = (scored[selected + '_probability'] >= bundle['thresholds'][selected]).astype(int)
            for message in range_messages(batch, bundle):
                st.warning(message)
            st.success(f'{len(scored):,} scenarios scored with all {len(models)} algorithms.')
            st.dataframe(scored, hide_index=True, use_container_width=True)
            st.download_button('Download scored CSV', scored.to_csv(index=False), 'scored_scenarios.csv', 'text/csv')
        except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as exc:
            st.error(str(exc))

with compare_tab:
    st.subheader('Evidence, not assumed accuracy')
    st.caption('Identical spatial test records for every algorithm. Lower Brier is better; higher ROC-AUC and F1 are better.')
    row = results.set_index('algorithm').loc[selected]
    a, b, c, d = st.columns(4)
    a.metric('Test ROC-AUC', f"{row.test_roc_auc:.3f}")
    b.metric('Test F1', f"{row.test_f1:.3f}")
    c.metric('Test Brier', f"{row.test_brier:.3f}")
    d.metric('Test balanced accuracy', f"{row.test_balanced_accuracy:.3f}")
    st.warning('Predictive signal is limited. High F1 can come from predicting almost every record positive. '
               'Compare the prevalence baseline, balanced accuracy and F1 at 0.5. No model is guaranteed to outperform the baseline.')
    display = results[['algorithm', 'cv_brier', 'validation_brier', 'test_roc_auc', 'test_f1',
                       'test_brier', 'test_f1_at_0_5', 'test_precision', 'test_recall']]
    st.dataframe(display.style.format({col: '{:.3f}' for col in display.columns if col != 'algorithm'}),
                 hide_index=True, use_container_width=True)
    st.download_button('Download full comparison & timings', results.to_csv(index=False), 'model_comparison.csv', 'text/csv')
    st.image(str(ROOT / 'artifacts/roc_calibration.png'), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.markdown('**Sampling uncertainty**')
        intervals = table('test_metric_intervals.csv')
        st.dataframe(intervals[intervals.algorithm == selected], hide_index=True)
        st.caption('95% spatial-cell bootstrap intervals, 500 resamples. Conditional on fitted models; no refitting. '
                   'Only a few held-out cells: intervals can be wide and are not external-validation guarantees.')
    with right:
        st.markdown('**BMA classification breakdown**')
        st.image(str(ROOT / 'artifacts/bma_confusion.png'), use_container_width=True)
    with st.expander('Calibration bins, timing and the full audit table'):
        st.dataframe(table('calibration_bins.csv'), hide_index=True)
        st.dataframe(results, hide_index=True)
        st.caption('Validation F1 is optimistic after threshold selection. Brier is overall probability quality, '
                   'not pure calibration. Training timings include search; inference timings are warmed batch milliseconds per row.')
    st.caption(bundle['metadata']['test_reuse_note'])

with weights_tab:
    st.subheader('An average of possibilities, not a single guess')
    st.caption('Posterior support across all 16 subsets of four environmental feature blocks.')
    left, right = st.columns([1.4, 1])
    with left:
        st.image(str(ROOT / 'artifacts/model_weights.png'), use_container_width=True)
    with right:
        st.markdown('**Feature-block inclusion probability**')
        st.bar_chart(table('block_inclusion.csv').set_index('block'), color='#39765c')
        prior = bma.inclusion_prior
        st.caption(f'Selected block-inclusion prior: {prior:.2f}. Prior coefficient SD: {bma.prior_scale:g}. '
                   'Both selected by training-only grouped CV, not test performance.')
    st.dataframe(table('bma_weights.csv'), hide_index=True, use_container_width=True)
    st.latex(r'p(y=1\mid x,D) = \sum_M p(y=1\mid x,D,M)\,p(M\mid D)')
    st.caption('Each subset includes an intercept and normalized Gaussian coefficient priors. '
               'Laplace evidence plus a Bernoulli block prior yields approximate posterior weights. '
               'Predictive probabilities integrate coefficient uncertainty by Gauss-Hermite quadrature. '
               'Weights describe this candidate family, not causal importance.')

with data_tab:
    st.subheader('Know the data behind the estimate')
    frame = load_data()
    a, b, c = st.columns(3)
    a.metric('Recorded fires', f'{len(frame):,}')
    b.metric('Positive recorded area', f'{(frame.area > 0).mean():.1%}')
    c.metric('Spatial cells', str(frame.groupby(['X', 'Y']).ngroups))
    left, right = st.columns(2)
    with left:
        st.markdown('**Records by month**')
        st.bar_chart(frame.month.value_counts().reindex(MONTHS, fill_value=0), color='#39765c')
    with right:
        st.markdown('**Feature distributions**')
        variable = st.selectbox('Inspect measurement', ['temp', 'RH', 'wind', 'rain', 'FFMC', 'DMC', 'DC', 'ISI'])
        counts, edges = np.histogram(frame[variable], bins=18)
        st.bar_chart(pd.DataFrame({'Bin midpoint': (edges[:-1] + edges[1:]) / 2, 'Records': counts}).set_index('Bin midpoint'), color='#c57545')
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.caption('Cortez & Morais (2007), UCI Forest Fires · CC BY 4.0 · https://doi.org/10.24432/C5D88D')

with methods_tab:
    st.subheader('Built to explain. Built to reproduce.')
    st.markdown('''
**The target matters.** `1[area > 0]` measures positive recorded burned area conditional on a recorded fire.
UCI Forest Fires cannot establish daily ignition probability. Zero area does not mean no fire.

**Leakage controls.** Whole `(X, Y)` cells belong to only one split. Four-fold grouped CV inside training
selects hyperparameters by Brier score. Every scaler is fitted inside its training fold. Validation selects
F1 thresholds; the identical test rows are used for all algorithms. Month and weekday names are not full
timestamps, so we do not invent chronological ordering.

**Features.** Temperature, humidity, wind, log-transformed rain, FFMC, DMC, DC, ISI and cyclical month encoding.
Area is the label only; coordinates are grouping variables only; weekday is excluded.

**Limitations.** Only 517 records from one park; uncertain shared-weather/event dependence; few test cells;
no representative non-fire controls. Reusing a previously published test set is not fresh external validation.
Credible intervals are conditional on approximate posteriors; bootstrap intervals do not include retraining uncertainty.
''')
    summary = pd.DataFrame(bundle['metadata']['split_summary']).T
    st.dataframe(summary, use_container_width=True)
    st.markdown('**Submission pack**')
    for label, filename, mime in [
        ('Executed Jupyter notebook', 'notebooks/01_wildfire_risk.ipynb', 'application/x-ipynb+json'),
        ('Short project report (PDF)', 'artifacts/Ember_Project_Report.pdf', 'application/pdf'),
        ('Presentation (PowerPoint)', 'artifacts/Ember_Presentation.pptx', 'application/vnd.openxmlformats-officedocument.presentationml.presentation'),
        ('Spatial split assignments', 'artifacts/split_assignments.csv', 'text/csv'),
        ('Held-out predictions', 'artifacts/test_predictions.csv', 'text/csv'),
        ('Metric uncertainty intervals', 'artifacts/test_metric_intervals.csv', 'text/csv')]:
        path = ROOT / filename
        if path.exists():
            st.download_button(label, path.read_bytes(), path.name, mime)
    st.link_button('UCI dataset & attribution', 'https://archive.ics.uci.edu/dataset/162/forest%2Bfires')
    st.code('python -m wildfire.train\npython -m wildfire.deliverables\npython -m pytest -q\npython -m streamlit run app.py')
    with st.expander('Complete experiment metadata'):
        st.json(bundle['metadata'])

st.markdown('<div class="footer">EMBER / RESEARCH, NOT EMERGENCY ADVICE &nbsp; · &nbsp; Unit 5 comparative industry project &nbsp; · &nbsp; Measured results. Visible limitations.</div>', unsafe_allow_html=True)
