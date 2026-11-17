"""Generate the required short PDF report and editable presentation from measured artifacts.

Run after training: python -m wildfire.deliverables
"""
import json
from html import escape
import pandas as pd
from .data import ROOT


def build_deliverables():
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor

    out = ROOT / 'artifacts'
    scores = pd.read_csv(out / 'model_comparison.csv')
    metadata = json.loads((out / 'metadata.json').read_text())
    intervals = pd.read_csv(out / 'test_metric_intervals.csv')
    bma = scores.set_index('algorithm').loc['BMA']
    baseline = scores.set_index('algorithm').loc['Prevalence Baseline']
    weights = pd.read_csv(out / 'bma_weights.csv')
    scope = ('Target: 1[recorded burned area > 0], conditional on a recorded fire. '
             'Area = 0 is not a confirmed non-fire observation. This dataset cannot validate daily ignition risk.')
    protocol = ('Use whole (X,Y) spatial cells: train/validation/test seeds 42 and 43, unchanged from the '
                'original repository. Training-only four-fold GroupKFold minimizes Brier score; scalers fit '
                'inside each fold. Validation selects F1 thresholds; no test-fitted calibration. '
                'All algorithms share identical validation and test rows.')
    bayes = ('Enumerate all 16 subsets of weather, moisture, fire-index and season blocks. Each includes an '
             'intercept. Coefficients have normalized Gaussian priors; independent Bernoulli block priors '
             'control model complexity. Laplace marginal likelihood times model prior yields posterior '
             'weights. Twenty-point Gauss-Hermite quadrature integrates coefficient uncertainty in predictions.')
    grids = ('Expanded training-only grids: BMA prior SD {0.15, 0.35, 0.7, 1, 2.5}, block inclusion '
             '{0.2, 0.5, 0.8}; Logistic Regression C {0.001, 0.01, 0.1, 1, 10, 100}; '
             'Decision Tree depth {1,2,4,6}, leaf size {5,15,30}, pruning alpha {0,0.01}; '
             'Random Forest 300 trees, depth {3,unrestricted}, leaf size {5,15,30}, max_features {0.5,1}.')
    conclusion = (f'BMA test ROC-AUC is {bma.test_roc_auc:.4f}, F1 {bma.test_f1:.4f}, '
                  f'Brier {bma.test_brier:.4f}, balanced accuracy {bma.test_balanced_accuracy:.4f}. '
                  f'Baseline F1 is also {baseline.test_f1:.4f}. BMA predicts every test record positive '
                  'at the validation-selected threshold; this F1 is not evidence of useful discrimination. '
                  'Expanded tuning does not establish a meaningful improvement in test performance.')
    limits = ('Only 517 records from one Portuguese park, with no representative non-fire controls. '
              'Month/weekday are not complete timestamps. Spatial grouping reduces location leakage but '
              'does not remove all event/weather dependence. Test includes only eight cells. This revision '
              'reuses previously published test scores; it is not fresh external validation. '
              'Do not use this educational application for emergency decisions.')
    uncertainty = ('Pointwise 90% BMA latent-probability credible intervals sample the model index and '
                   'Laplace coefficient posterior (2,000 draws, seed 42). Conditional on the candidate '
                   'family; they do not account for dataset bias. Test metric 95% percentile intervals '
                   'resample spatial cells 500 times (seed 2026), skip one-class resamples, and do not refit models. '
                   'They describe conditional sampling variability, not external validity.')
    preprocessing = ('Nine scenario inputs become ten predictors: temperature, RH, wind, log1p(rain), '
                     'DMC, DC, FFMC, ISI and cyclical month sine/cosine. Coordinates only define groups; '
                     'area only defines the target; weekday is excluded. Invalid or nonfinite inputs fail validation.')
    requirements = [
        ['PDF requirement', 'Delivered evidence'],
        ['UCI data / careful target / weather', 'data/forestfires.csv; data.py; notebook sections 1-3'],
        ['Primary Bayesian Model Averaging', 'bma.py; all 16 model weights and block-inclusion plot'],
        ['LR / Tree / Random Forest comparisons', 'model_comparison.csv; *_cv.csv; shared test rows'],
        ['ROC-AUC / F1 / Brier / calibration', 'comparison table; roc_calibration.png; calibration_bins.csv'],
        ['Probability interface / weights', 'app.py: scenario studio, Bayesian evidence, batch scoring'],
        ['Notebook / repository / report / slides', 'Executed notebook; GitHub; this PDF; editable PPTX'],
        ['Timing / interpretability / audit plots', 'Full comparison CSV; split assignments; predictions; plots']]

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='EmberBody', fontSize=10, leading=15, spaceAfter=10))
    styles['Title'].textColor = colors.HexColor('#173e32')
    styles['Heading1'].textColor = colors.HexColor('#173e32')
    story = []

    def text(value, style='EmberBody'):
        story.append(Paragraph(escape(value), styles[style]))

    def heading(value):
        text(value, 'Heading1')

    def report_table(rows, widths):
        cells = [[Paragraph(escape(str(value)), styles['EmberBody']) for value in row] for row in rows]
        t = Table(cells, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eaf0e9')),
                               ('GRID', (0,0), (-1,-1), .4, colors.HexColor('#d8e1d5')),
                               ('VALIGN', (0,0), (-1,-1), 'TOP'), ('LEFTPADDING',(0,0),(-1,-1),8)]))
        story.extend([t, Spacer(1,12)])

    text('EMBER', 'Title')
    text('AI-Based Wildfire Occurrence Risk Assessment', 'Heading1')
    text('Unit 5 / Project 3 / Bayesian Model Averaging comparative study')
    heading('1. Objective and honest target')
    text(scope)
    text('Compare probabilistic environmental models and assess calibration and uncertainty. '
         'The implemented target is a documented proxy, not a claim of operational fire occurrence prediction.')
    heading('2. Data and experiment protocol')
    text(preprocessing); text(protocol)
    split_rows = [['Split','Records','Cells','Positive fraction']]
    for name, info in metadata['split_summary'].items():
        split_rows.append([name, info['records'], info['spatial_groups'], f"{info['positive_fraction']:.3f}"])
    report_table(split_rows, [130,90,90,170])
    heading('3. Primary algorithm and tuning')
    text(bayes); text(grids)
    story.append(PageBreak())
    heading('4. Measured comparative results')
    rows = [['Algorithm','Val. Brier','Test AUC','Test F1','Test Brier']]
    for row in scores.itertuples():
        rows.append([row.algorithm, f'{row.validation_brier:.4f}', f'{row.test_roc_auc:.4f}',
                     f'{row.test_f1:.4f}', f'{row.test_brier:.4f}'])
    report_table(rows, [150,82,82,82,84])
    text(conclusion)
    text('Higher ROC-AUC and F1 are better; lower Brier is better. Validation F1 is optimistic after '
         'threshold selection. Training times include the complete search; inference times average '
         '30 warmed batch predictions divided by row count, excluding I/O.')
    rows = [['Algorithm','Train/search (s)','Inference (ms/row)','Interpretability']]
    for row in scores.itertuples():
        rows.append([row.algorithm, f'{row.training_seconds_including_search:.3f}',
                     f'{row.inference_ms_per_row_batch:.4f}', row.interpretability])
    report_table(rows, [120,100,100,160])
    story.append(Image(str(out / 'roc_calibration.png'), width=6.5*inch, height=2.71*inch))
    story.append(PageBreak())
    heading('5. Bayesian evidence and uncertainty')
    text(f"Most supported subset: {weights.iloc[0]['candidate']} (weight {weights.iloc[0]['weight']:.2%}). "
         "Weights sum to one and are conditional on this family, preprocessing and selected priors.")
    text(uncertainty)
    story.append(Image(str(out / 'model_weights.png'), width=6.3*inch, height=4.41*inch))
    selected = intervals[intervals.algorithm == 'BMA']
    report_table([['BMA metric','Lower 95%','Upper 95%']] +
                 [[r.metric, f'{r.lower_95:.4f}', f'{r.upper_95:.4f}'] for r in selected.itertuples()], [200,140,140])
    story.append(PageBreak())
    heading('6. Application and submission checklist')
    text('The Streamlit interface offers an editable scenario, all model probabilities, Bayesian '
         'credible intervals, training-range warnings, JSON export, batch CSV scoring, single-input '
         'sensitivity, calibration and ROC plots, model weights, dataset exploration and downloads.')
    report_table(requirements, [190,290])
    heading('7. Limits and recommended next steps')
    text(limits)
    text('Obtain representative non-fire observations and exact dates, prespecify the observation window, '
         'then validate on genuinely new regions and time periods. Better data, not optimistic splitting, '
         'is needed for credible operational performance. Student name, roll number and institution '
         'should be supplied by the student; none have been invented in these deliverables.')
    heading('Reproduce and reference')
    text('python -m wildfire.train; python -m wildfire.deliverables; python -m pytest -q; '
         'python -m streamlit run app.py')
    text('Cortez, P. & Morais, A. (2007). Forest Fires. UCI Machine Learning Repository. '
         'https://doi.org/10.24432/C5D88D (CC BY 4.0). Code: https://github.com/RomitDeokar/wildfire_risk_ml')
    text(f"Python {metadata['python']}; SHA-256: {metadata['data_sha256']}")

    def footer(canvas, doc):
        canvas.setFont('Helvetica',8)
        canvas.drawString(40,25,'EMBER / Educational proxy study / Not emergency advice')
        canvas.drawRightString(555,25,str(doc.page))

    SimpleDocTemplate(str(out / 'Ember_Project_Report.pdf'), leftMargin=40, rightMargin=40,
                      topMargin=35, bottomMargin=40, title='Ember: Wildfire BMA Study').build(
                          story, onFirstPage=footer, onLaterPages=footer)

    deck = Presentation(); deck.slide_width=Inches(13.333); deck.slide_height=Inches(7.5)
    green=RGBColor.from_string('173E32'); cream=RGBColor.from_string('F7F8F4')
    gray=RGBColor.from_string('4B6257'); orange=RGBColor.from_string('C76C43')

    def box(slide, value, x, y, w, h, size=22, color=gray, bold=False):
        shape=slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf=shape.text_frame; tf.word_wrap=True
        for i, line in enumerate(value.split('\n')):
            p=tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text=line; p.font.name='Aptos'; p.font.size=Pt(size); p.font.bold=bold; p.font.color.rgb=color
            p.space_after=Pt(16)
        return shape

    def slide(title, body=None, image=None):
        s=deck.slides.add_slide(deck.slide_layouts[6]); s.background.fill.solid(); s.background.fill.fore_color.rgb=cream
        box(s,'EMBER / UNIT 5 / PROJECT 03',.65,.3,12,.4,12,orange,True)
        box(s,title,.65,.85,12,1,34,green,True)
        if body: box(s,body,.7,2.0,11.9,4.7,22)
        if image: s.shapes.add_picture(str(out/image),Inches(1.5),Inches(1.9),height=Inches(4.9))
        box(s,f'Positive recorded-area proxy, not ignition probability    |    {len(deck.slides):02d}',.7,7.05,12,.3,10)
        s.notes_slide.notes_text_frame.text=scope+'\n'+limits
        return s

    slide('Explore conditions. Understand uncertainty.',
          'AI-Based Wildfire Occurrence Risk Assessment\nBayesian Model Averaging with Logistic Regression, Decision Tree and Random Forest\nUCI Forest Fires / Montesinho, Portugal / 517 records\nStudent details: add your name, roll number and institution.')
    slide('First, define what we can actually predict.',scope+'\nResearch question: do environmental measurements discriminate this proxy on unseen spatial cells?')
    slide('Weather in. Leakage controls throughout.',preprocessing+'\n'+protocol)
    slide('16 possibilities. One Bayesian average.',bayes+'\nEmpirical prior tuning is not full integration over hyperparameters.')
    slide('Tune inside training. Freeze before evaluation.',
          'Four-fold grouped CV: minimize Brier score.\nBMA: five prior scales and three block-prior settings.\nExpanded LR/tree/forest grids; identical folds and test records.\nValidation selects F1 thresholds; test scores never select a configuration.')
    s=slide('Measured results on the shared test set.')
    t=s.shapes.add_table(6,5,Inches(.7),Inches(2.1),Inches(11.9),Inches(2.8)).table
    t.columns[0].width=Inches(3.7)
    values=[['Algorithm','Val. Brier','Test AUC','Test F1','Test Brier']]+[
        [r.algorithm,f'{r.validation_brier:.3f}',f'{r.test_roc_auc:.3f}',f'{r.test_f1:.3f}',f'{r.test_brier:.3f}']
        for r in scores.itertuples()]
    for i,row in enumerate(values):
        for j,value in enumerate(row):
            cell=t.cell(i,j); cell.text=value; cell.fill.solid(); cell.fill.fore_color.rgb=green if i==0 else cream
            for p in cell.text_frame.paragraphs:
                p.font.size=Pt(17); p.font.color.rgb=cream if i==0 else green
    box(s,'BMA F1 matches the prevalence baseline. Balanced accuracy: 0.500.\nThere is no established meaningful predictive gain from expanded tuning.',.7,5.25,12,1.3,20)
    slide('Check calibration, not just classification.',image='roc_calibration.png')
    slide('Posterior weights tell a transparent story.',image='model_weights.png')
    slide('Uncertainty is part of the result.',uncertainty)
    slide('A working, auditable research demonstration.',
          'Scenario studio with all model probabilities and credible intervals.\nBatch CSV scoring, JSON export and input-range checks.\nModel performance, Bayesian evidence and dataset explorer.\nExecuted notebook, short report, editable presentation and audit CSVs.')
    slide('Better data is the next real improvement.',limits+'\nNext: representative controls, exact observation windows, independent regions and chronological validation.')
    slide('Reproduce the evidence.',
          'python -m wildfire.train\npython -m wildfire.deliverables\npython -m pytest -q\npython -m streamlit run app.py\nUCI: doi.org/10.24432/C5D88D / Cortez & Morais (2007), CC BY 4.0')
    deck.save(out / 'Ember_Presentation.pptx')
    print('Generated short report and 12-slide editable presentation in artifacts/.')


if __name__ == '__main__':
    build_deliverables()
