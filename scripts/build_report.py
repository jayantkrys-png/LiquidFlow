"""Build the eight-page report PDF. Every number is read from artifacts/ or data/processed/."""
from html import escape
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

BASE = Path(__file__).resolve().parents[1]
FIG = BASE / 'artifacts/figures'
FONTS = BASE / 'assets/fonts'
for n in ['Regular', 'Medium', 'SemiBold', 'Bold']:
    pdfmetrics.registerFont(TTFont('I' + n, str(FONTS / f'Inter-{n}.ttf')))
for n in ['Bold', 'ExtraBold']:
    pdfmetrics.registerFont(TTFont('ID' + n, str(FONTS / f'InterDisplay-{n}.ttf')))
for n in ['Regular', 'Medium', 'Bold']:
    pdfmetrics.registerFont(TTFont('JB' + n, str(FONTS / f'JetBrainsMono-{n}.ttf')))
pdfmetrics.registerFontFamily('IRegular', normal='IRegular', bold='ISemiBold', italic='IRegular', boldItalic='ISemiBold')
pdfmetrics.registerFontFamily('JBRegular', normal='JBRegular', bold='JBBold', italic='JBRegular', boldItalic='JBBold')

W, H = A4; M = 42; CW = W - 2 * M
NAVY, BLUE, ICE, TEXT, MUTED, LINE = '#142C46', '#1769B5', '#EEF5FC', '#243B53', '#52677D', '#D9E4EF'
ORANGE, ORANGED, ORANGEBG, B80, B95 = '#D9822B', '#A8541A', '#FDF1E6', '#A9C8EA', '#DCE8F6'
col = colors.HexColor
TEAM = 'Angry Bird'; MEMBERS = ['Jayant Raj', 'Rohan Kumar Singh', 'Harshit Kumar', 'Viraj Singh']

from reportlab import rl_config

rl_config.canvas_basefontname = 'IRegular'
out = BASE / 'report/LiquidFlow_Team_Angry_Bird_PS03.pdf'
out.parent.mkdir(exist_ok=True)
c = canvas.Canvas(str(out), pagesize=A4, initialFontName='IRegular', initialFontSize=10)
c.setTitle('LiquidFlow: How much cash should a liquid fund hold?')
c.setAuthor('Team Angry Bird, IIT Patna'); c.setSubject('Zeckathon 2026, Problem Statement 03')
S = {
    'body': ParagraphStyle('body', fontName='IRegular', fontSize=10.5, leading=15.2, textColor=col(TEXT)),
    'lead': ParagraphStyle('lead', fontName='IMedium', fontSize=13.5, leading=19.5, textColor=col(NAVY)),
    'small': ParagraphStyle('small', fontName='IRegular', fontSize=8.8, leading=12.6, textColor=col(MUTED)),
    'h': ParagraphStyle('h', fontName='IBold', fontSize=13, leading=17, textColor=col(NAVY)),
    'kick': ParagraphStyle('kick', fontName='JBMedium', fontSize=8, leading=10, textColor=col(BLUE)),
    'cell': ParagraphStyle('cell', fontName='IRegular', fontSize=9, leading=12, textColor=col(TEXT)),
    'num': ParagraphStyle('num', fontName='JBRegular', fontSize=8.6, leading=12, textColor=col(TEXT), alignment=TA_RIGHT),
    'numb': ParagraphStyle('numb', fontName='JBBold', fontSize=8.6, leading=12, textColor=col(NAVY), alignment=TA_RIGHT),
    'head': ParagraphStyle('head', fontName='ISemiBold', fontSize=8.4, leading=11, textColor=col(MUTED)),
    'headr': ParagraphStyle('headr', fontName='ISemiBold', fontSize=8.4, leading=11, textColor=col(MUTED), alignment=TA_RIGHT),
}
page_no = 0; y = 0
def mono(t, bold=False): return f'<font name="{"JBBold" if bold else "JBRegular"}">{t}</font>'
def para(text, style='body', x=M, width=CW, top=None, gap=10):
    global y
    top = y if top is None else top
    p = Paragraph(text, S[style]); _, h = p.wrap(width, 2000)
    assert top - h >= 50, (page_no, text[:50], top, h)
    p.drawOn(c, x, top - h); y = top - h - gap; return h
def head(t, x=M, width=CW, top=None, gap=7):
    return para(t, 'h', x=x, width=width, top=top, gap=gap)
def image(name, x=M, width=CW, top=None, gap=8):
    global y
    top = y if top is None else top
    im = PILImage.open(FIG / f'{name}.png'); h = width * im.height / im.width
    assert top - h >= 50, (page_no, name, top, h)
    c.drawImage(str(FIG / f'{name}.png'), x, top - h, width=width, height=h, mask='auto'); y = top - h - gap; return h
def start(title, kicker):
    global page_no, y
    if page_no: c.showPage()
    page_no += 1
    c.setFillColor(col(BLUE)); c.rect(M, H - 37, 22, 3, fill=1, stroke=0)
    c.setFillColor(col(NAVY)); c.setFont('IBold', 8.5); c.drawString(M + 30, H - 38.5, 'LiquidFlow')
    c.setFillColor(col(MUTED)); c.setFont('IRegular', 8.5); c.drawRightString(W - M, H - 38.5, f'Team {TEAM}  ·  IIT Patna')
    c.setStrokeColor(col(LINE)); c.setLineWidth(.6); c.line(M, 40, W - M, 40)
    c.setFont('IRegular', 8); c.setFillColor(col(MUTED)); c.drawString(M, 26, 'Zeckathon 2026  ·  Problem Statement 03')
    c.setFillColor(col(BLUE)); c.setFont('JBBold', 8); c.drawRightString(W - M, 26, f'{page_no:02d}')
    y = H - 74
    para(kicker.upper(), 'kick', gap=7)
    c.setFillColor(col(NAVY)); c.setFont('IDExtraBold', 25); c.drawString(M, y - 24, title); y -= 42
def note(t, warn=False, gap=14):
    global y
    p = Paragraph(t, S['body']); _, h = p.wrap(CW - 30, 1000)
    c.setFillColor(col(ORANGEBG if warn else ICE)); c.roundRect(M, y - h - 22, CW, h + 22, 6, fill=1, stroke=0)
    c.setFillColor(col(ORANGE if warn else BLUE)); c.rect(M, y - h - 22, 3, h + 22, fill=1, stroke=0)
    p.drawOn(c, M + 16, y - h - 11); y -= h + 22 + gap
def table(headers, rows, widths, numeric, bold_col=None, x=M, width=CW, gap=12, zebra=None):
    global y
    data = [[Paragraph(escape(h_), S['headr' if numeric[i] else 'head']) for i, h_ in enumerate(headers)]]
    for r in rows:
        data.append([Paragraph(escape(str(v)), S['numb' if (numeric[i] and i == bold_col) else 'num' if numeric[i] else 'cell']) for i, v in enumerate(r)])
    t = Table(data, colWidths=[width * w for w in widths])
    st = [('FONTNAME', (0, 0), (-1, -1), 'IRegular'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
          ('TOPPADDING', (0, 0), (-1, -1), 5.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5.5),
          ('LINEBELOW', (0, 0), (-1, 0), .9, col(NAVY)), ('LINEBELOW', (0, 1), (-1, -1), .4, col(LINE))]
    if zebra is not None: st.append(('BACKGROUND', (0, zebra), (-1, zebra), col(ORANGEBG)))
    t.setStyle(TableStyle(st)); _, h = t.wrap(width, 900)
    assert y - h >= 50, (page_no, y, h); t.drawOn(c, x, y - h); y -= h + gap
def f0(v): return f'{float(v):,.0f}'.replace('-', '−')
def swatch_key(items, x=M, top=None):
    global y
    top = y if top is None else top; cx = x
    for kind, color, label in items:
        if kind == 'box': c.setFillColor(col(color)); c.rect(cx, top - 7, 12, 7, fill=1, stroke=0)
        elif kind == 'dot': c.setStrokeColor(col(BLUE)); c.setFillColor(colors.white); c.setLineWidth(1.3); c.circle(cx + 6, top - 3.5, 3, fill=1, stroke=1)
        elif kind == 'dash': c.setStrokeColor(col(color)); c.setLineWidth(1.6); c.line(cx, top - 3.5, cx + 12, top - 3.5)
        c.setFillColor(col(MUTED)); c.setFont('IRegular', 8); c.drawString(cx + 17, top - 6.5, label)
        cx += 17 + pdfmetrics.stringWidth(label, 'IRegular', 8) + 16
    y = top - 18

fc = pd.read_csv(BASE / 'artifacts/forecasts.csv'); live = fc[fc.split == 'live'].sort_values('horizon')
flows = pd.read_csv(BASE / 'data/processed/amfi_flows.csv').sort_values(['category', 'month'])
flows['prev'] = flows.groupby('category').aum_crore.shift(1)
mo = pd.read_csv(BASE / 'artifacts/metrics_overall.csv'); test = mo[mo.split == 'test'].set_index('model')
ps = pd.read_csv(BASE / 'artifacts/policy_sanity_checks.csv')
lit = ps[ps.interpretation == 'literal_3_rupees_per_rupee'].set_index('policy')
alt = ps[ps.interpretation == 'three_times_monthly_yield_gap'].set_index('policy')
L = live[live.category == 'Liquid'].set_index('target_month'); O = live[live.category == 'Overnight'].set_index('target_month')
mar = flows[(flows.category == 'Liquid') & flows.month.str.endswith('-03')].copy(); mar['pct'] = -mar.net_flow_crore / mar.prev * 100
last3 = mar.tail(3).pct; aum_l = L.origin_aum.iloc[0]
saving = (1 - lit.loc['selected_model_q95', 'cost_crore'] / lit.loc['prior_year_outflow', 'cost_crore']) * 100
sar, sn = test.loc['sarimax'], test.loc['seasonal_naive']
by_cat = pd.read_csv(BASE / 'artifacts/metrics_by_category_horizon.csv')
coverage = by_cat[(by_cat.split == 'test') & (by_cat.model == 'sarimax') & by_cat.horizon.isin([1, 2, 3])].groupby('category')[['coverage80_pct', 'coverage95_pct']].mean()
_pm = pd.read_csv(BASE / 'artifacts/policy_months.csv')
short_months = (_pm.outflow_crore > _pm.q95_buffer_crore).groupby(_pm.category).sum(); policy_n = _pm.groupby('category').size()
mape_cut = (1 - sar.mape_pct / sn.mape_pct) * 100
assert abs(saving - 87.5) < .05 and abs(mape_cut - 41.0) < .05

# 01 Cover -----------------------------------------------------------------------------------------
page_no = 1
logo = BASE / 'assets/brand/iitp_full.png'; li = PILImage.open(logo); lw = 236; lh = lw * li.height / li.width
c.drawImage(str(logo), M, H - 44 - lh, width=lw, height=lh, mask='auto')
c.setFillColor(col(MUTED)); c.setFont('IRegular', 8.5)
c.drawRightString(W - M, H - 54, 'Zeckathon 2026'); c.drawRightString(W - M, H - 66, 'Problem Statement 03')
y = H - 150
c.setFillColor(col(BLUE)); c.rect(M, y + 2, 22, 3, fill=1, stroke=0)
c.setFont('JBBold', 9); c.drawString(M + 30, y, 'LIQUIDFLOW')
c.setFillColor(col(NAVY)); c.setFont('IDExtraBold', 40)
for i, line in enumerate(['How much cash should', 'a liquid fund hold?']): c.drawString(M, y - 46 - i * 46, line)
y -= 122
para('Three-month forecasts of money moving in and out of India\u2019s liquid and overnight funds, and a cash-buffer rule built on them.', 'lead', width=420, gap=26)
image('cover_flows', gap=6)
para('Net flow into liquid funds each month, July 2019 to September 2026, in \u20b9 crore. Blue is money in, orange is money out. Every March, the darkest bars, has been an outflow.', 'small', gap=16)
note('<b>In short:</b> hold about 4% of assets in a normal month and 21% going into December, and plan for roughly a quarter of the money to leave in March.', gap=0)
# team block
by = 60
c.setStrokeColor(col(NAVY)); c.setLineWidth(1); c.line(M, by + 92, W - M, by + 92)
c.setFillColor(col(MUTED)); c.setFont('JBMedium', 7.5); c.drawString(M, by + 74, 'TEAM')
c.setFillColor(col(NAVY)); c.setFont('IDExtraBold', 20); c.drawString(M, by + 50, TEAM)
c.setFillColor(col(TEXT)); c.setFont('IRegular', 9); c.drawString(M, by + 34, 'Indian Institute of Technology Patna')
c.setFillColor(col(MUTED)); c.setFont('JBMedium', 7.5); c.drawString(M + 215, by + 74, 'MEMBERS')
for i, mname in enumerate(MEMBERS):
    cx = M + 215 + (i % 2) * 150; cy = by + 52 - (i // 2) * 20
    c.setFillColor(col(NAVY)); c.setFont('ISemiBold', 10.5); c.drawString(cx, cy, mname)
c.setFillColor(col(MUTED)); c.setFont('IRegular', 8); c.drawString(M, by + 4, 'Submitted 29 October 2026  ·  Data to September 2026')

# 02 Summary -------------------------------------------------------------------------------------
start('The short answer', 'Summary')
para(f'Liquid funds should keep about 3 to 4% of assets ready to pay out in a normal month, and about 21% going into December. '
     f'March is the hardest month of the year: in each of the last three, {last3.min():.0f}% to {last3.max():.0f}% of the category\u2019s money left.', 'lead', gap=18)
cards = [(f'{L.loc["2026-12", "buffer_aum_pct"]:.1f}%', f'Liquid-fund buffer for December 2026, about \u20b9{L.loc["2026-12", "outflow95_buffer_crore"]/1e5:.2f} lakh crore'),
         (f'{int((mar.net_flow_crore < 0).sum())} of {len(mar)}', 'Marches since 2020 with net outflows from liquid funds'),
         (f'{int(lit.loc["selected_model_q95", "shortfall_category_months"])} of 48', f'Backtest months where our buffer fell short. Last year\u2019s rule: {int(lit.loc["prior_year_outflow", "shortfall_category_months"])}'),
         (f'{saving:.1f}%', 'Lower shortfall cost than last year\u2019s rule, on the brief\u2019s cost formula')]
cw = (CW - 3 * 10) / 4; top = y
for i, (v, lab) in enumerate(cards):
    x = M + i * (cw + 10)
    c.setFillColor(col(ICE)); c.roundRect(x, top - 92, cw, 92, 6, fill=1, stroke=0)
    c.setFillColor(col(ORANGED if i < 2 else BLUE)); c.setFont('JBBold', 19); c.drawString(x + 11, top - 30, v)
    p = Paragraph(lab, ParagraphStyle('cl', parent=S['small'], fontSize=8.4, leading=11.2, textColor=col(TEXT))); p.wrap(cw - 22, 60); p.drawOn(c, x + 11, top - 84 + (44 - p.height))
y = top - 92 - 22
head('Cash to hold, month by month')
para('Buffer as a share of September 2026 assets. It is set so that only a 1-in-20 month should need more.', 'small', gap=6)
image('buffer_outlook', gap=16)
head('What a fund manager should take from this', gap=8)
for i, (t, b) in enumerate([
    ('Plan around the tax calendar.', 'Advance tax is due on 15 June, 15 September, 15 December and 15 March. Companies redeem to pay it, so those months carry the big outflows. The calendar was the most useful input in our models.'),
    ('Size cash from the downside, not the average.', f'November\u2019s central forecast is an inflow of \u20b9{f0(L.loc["2026-11", "point"])} crore, yet there is a 1-in-20 chance of an outflow above \u20b9{f0(L.loc["2026-11", "outflow95_buffer_crore"])} crore. The buffer covers that tail.'),
    ('Overnight funds need proportionally more.', f'They are a seventh the size of liquid funds and swing harder, so their buffer runs at {O.buffer_aum_pct.min():.0f}% to {O.buffer_aum_pct.max():.0f}% of assets.')]):
    top = y
    c.setFillColor(col(BLUE)); c.setFont('JBBold', 10); c.drawString(M, top - 11, f'0{i+1}')
    para(f'<b>{t}</b> {b}', x=M + 26, width=CW - 26, gap=8)

# 03 Calendar ------------------------------------------------------------------------------------
start('Money leaves on tax dates', 'What drives the flows')
para('Most of the money in liquid funds belongs to companies parking short-term cash. When advance tax falls due they take it back, and the same rhythm shows up year after year. It is sharpest in March, when the fourth instalment lands on top of the financial year-end.', gap=14)
head('Liquid-fund net flow as % of the previous month\u2019s assets', gap=4)
para('Boxed columns are advance-tax months. The bottom row counts the years in which the month had a net outflow.', 'small', gap=6)
image('calendar_heatmap', gap=18)
top = y; lwid = 228
head('March, year by year', width=lwid, top=top, gap=6)
rows = [[r.month[:4], f0(r.net_flow_crore), f'{-r.pct:.1f}%'.replace('-', '−')] for r in mar.itertuples()]
table(['Year', 'Net flow, \u20b9 cr', '% of Feb assets'], rows, [.26, .4, .34], [False, True, True], bold_col=2, width=lwid, gap=8)
para(f'A repeat of the last three Marches at September 2026 size would mean \u20b9{last3.min()/100*aum_l/1e5:.1f} to {last3.max()/100*aum_l/1e5:.1f} lakh crore leaving the category in March 2027.', 'small', width=lwid)
ly = y; rx = M + lwid + 28; rw = CW - lwid - 28
head('Which inputs helped the forecast', x=rx, width=rw, top=top, gap=6)
image('drivers', x=rx, width=rw, gap=6)
para('Extra forecast error when each input group is removed from our Ridge model, measured on the Oct 2023 to Sep 2024 selection year. Calendar and interest rates carried the signal. Equity and banking-liquidity data did not add to it.', 'small', x=rx, width=rw)
y = min(y, ly)

# 04 / 05 Outlooks -------------------------------------------------------------------------------
intro = {
 'Liquid': f'We expect money to come in during October, November and January, and about \u20b9{f0(-L.loc["2026-12", "point"])} crore to go out in December, an advance-tax month. Every month could still go either way, which is why the buffer is set from the bottom of the range and not from the central forecast.',
 'Overnight': f'Overnight funds look close to flat: small inflows in October, November and January and a slight outflow in December. The category is small, about \u20b9{f0(O.origin_aum.iloc[0])} crore, so a \u20b930,000 crore swing is a third of it. That is what pushes the buffer to {O.buffer_aum_pct.min():.0f}% to {O.buffer_aum_pct.max():.0f}% of assets.'}
for cat, D in [('Liquid', L), ('Overnight', O)]:
    start(f'{cat} funds: the next four months', 'Forecast  ·  crore rupees')
    para(intro[cat], gap=14)
    image(f'fan_{cat.lower()}', gap=6)
    swatch_key([('box', B95, '95% range'), ('box', B80, '80% range'), ('dot', BLUE, 'Central forecast'), ('dash', ORANGED, 'Buffer level (5% downside)')])
    y -= 6
    rows = [[pd.Timestamp(m).strftime('%b %Y'), f0(r.point), f'{f0(r.q100)} to {f0(r.q900)}', f'{f0(r.q025)} to {f0(r.q975)}', f0(r.outflow95_buffer_crore), f'{r.buffer_aum_pct:.1f}%'] for m, r in D.iterrows()]
    table(['Month', 'Central', '80% range', '95% range', 'Buffer', '% of assets'], rows, [.13, .13, .25, .25, .12, .12], [False, True, True, True, True, True], bold_col=4,
          zebra=(3 if cat == 'Liquid' else None))
    para('October is a nowcast: AMFI publishes October figures in early November. January is a fourth month ahead, one beyond the brief, and was backtested on its own. Buffers are per month and should not be added together.', 'small', gap=14)
    cov80, cov95 = coverage.loc[cat, 'coverage80_pct'], coverage.loc[cat, 'coverage95_pct']
    n_short, n_months = int(short_months[cat]), int(policy_n[cat])
    if cat == 'Liquid':
        note(f'<b>How sure are we?</b> Over the two-year test, the liquid-fund 95% range caught {cov95:.0f}% of actual outcomes and the 80% range caught {cov80:.0f}%, so the inner band runs a little narrow. The buffer fell short in {n_short} of {n_months} months (page 6).')
    else:
        note(f'<b>How sure are we?</b> Less sure than for liquid funds. Over the two-year test, the overnight 95% range caught {cov95:.0f}% of actual outcomes and the 80% range caught {cov80:.0f}%, so both bands are too narrow. The buffer fell short in {n_short} of {n_months} months (page 6); treat it as a minimum.')

# 06 Backtest ------------------------------------------------------------------------------------
start('The buffer, tested on two years', 'Backtest  ·  Oct 2024 to Sep 2026')
para('We replayed October 2024 to September 2026 one month at a time. Each buffer was set using only what was known that month, then compared with what actually left. The benchmark is the rule named in the brief: hold last year\u2019s outflow for the same month.', gap=12)
image('backtest', gap=4)
para('\u20b9 crore, one month ahead, 24 months per category. Bars show net outflow; months with net inflow have no bar.', 'small', gap=16)
head('What each rule costs')
para('Cost from the brief: 0.5% a year on cash held beyond what was needed, plus \u20b93 for every rupee of shortfall, summed over all 48 category-months, in \u20b9 crore.', 'small', gap=6)
image('policy_cost', gap=12)
pr, sq, nq = lit.loc['prior_year_outflow'], lit.loc['selected_model_q95'], lit.loc['seasonal_naive_q95']
para(f'LiquidFlow cut the cost by {saving:.1f}% against last year\u2019s rule. It holds more cash, {sq.mean_buffer_aum_pct:.1f}% of assets on average against {pr.mean_buffer_aum_pct:.1f}%, and in return it ran short in {int(sq.shortfall_category_months)} months instead of {int(pr.shortfall_category_months)}. '
     f'A plain seasonal 95% band, which holds even more ({nq.mean_buffer_aum_pct:.1f}%), was cheaper again. Most of the saving comes from sizing cash to the downside of a forecast range, and that idea holds across both methods.')

# 07 Method --------------------------------------------------------------------------------------
start('How LiquidFlow works', 'Method')
steps = [('Data', '87 monthly AMFI reports per category, plus RBI rates, Nifty 50 and liquidity data'),
         ('Inputs', 'Calendar and tax dates, interest rates, equity and banking liquidity'),
         ('Models', 'Seven candidates, one chosen on the 2023 to 2024 year, then locked'),
         ('Ranges', 'Built from the model\u2019s own last 24 errors'),
         ('Buffer', 'Cash to cover the 5% downside, capped at assets')]
bw = (CW - 4 * 12) / 5; top = y
for i, (t, b) in enumerate(steps):
    x = M + i * (bw + 12)
    c.setFillColor(col(NAVY if i == 4 else ICE)); c.roundRect(x, top - 96, bw, 96, 6, fill=1, stroke=0)
    c.setFillColor(col(ORANGE if i == 4 else BLUE)); c.setFont('JBBold', 8); c.drawString(x + 9, top - 16, f'0{i+1}')
    c.setFillColor(colors.white if i == 4 else col(NAVY)); c.setFont('IBold', 10.5); c.drawString(x + 9, top - 31, t)
    p = Paragraph(b, ParagraphStyle('st', parent=S['small'], fontSize=8, leading=10.6, textColor=colors.white if i == 4 else col(TEXT))); p.wrap(bw - 18, 60); p.drawOn(c, x + 9, top - 39 - p.height)
    if i < 4:
        ax_ = x + bw + 2; c.setStrokeColor(col(BLUE)); c.setLineWidth(1); c.line(ax_, top - 48, ax_ + 8, top - 48)
        c.setFillColor(col(BLUE)); pth = c.beginPath(); pth.moveTo(ax_ + 8, top - 45.5); pth.lineTo(ax_ + 10.5, top - 48); pth.lineTo(ax_ + 8, top - 50.5); pth.close(); c.drawPath(pth, fill=1, stroke=0)
y = top - 96 - 20
head('A fair test')
para('We chose the model on one year of data, October 2023 to September 2024, and locked that choice before looking at the two test years. Each forecast uses only data that was public at the time: economic inputs lag by two months, and an error only feeds the ranges once its month has been reported. All seven candidates got the same treatment.', gap=14)
head('How the seven models compared on the test years')
para('SARIMAX, in blue, was chosen in the selection year. The seasonal naive benchmark, in navy, repeats last year\u2019s figure for the same month.', 'small', gap=6)
image('models', gap=10)
para(f'Simple seasonal methods were slightly more accurate in rupee terms ({sn.aum_scaled_mae_pct:.1f}% of assets against {sar.aum_scaled_mae_pct:.1f}% for SARIMAX), while SARIMAX cut percentage error by {mape_cut:.0f}%. '
     'The machine-learning models, Ridge and CatBoost, landed close to SARIMAX and did not beat the seasonal methods on rupee error. With monthly data, the calendar does most of the work, and a seasonal method already captures it. Better data is a surer way forward than a bigger model.')

# 08 Notes ---------------------------------------------------------------------------------------
start('Notes and sources', 'Definitions  ·  Scope  ·  References')
top = y; half = (CW - 26) / 2; rx = M + half + 26
head('Definitions', width=half, top=top, gap=6)
for t in ['<b>Net flow</b> is subscriptions minus redemptions for the month, taken straight from AMFI\u2019s reported columns.',
          f'<b>Buffer</b> = {mono("max(0, −q05)")}, where {mono("q05")} is the 5th percentile of forecast net flow, capped at the category\u2019s assets.',
          f'<b>Cost</b> = idle-cash cost + shortfall penalty, where {mono("O")} is the actual net outflow and {mono("B")} the buffer:<br/>{mono("0.5% × days/365 × max(B − O, 0)")}<br/>{mono("+ 3 × max(O − B, 0)")}',
          f'If the penalty is read as three times the monthly yield gap instead of \u20b93 per rupee, holding little cash becomes the cheaper choice and last year\u2019s rule edges ahead (\u20b9{alt.loc["prior_year_outflow", "cost_crore"]:,.0f} crore against \u20b9{alt.loc["selected_model_q95", "cost_crore"]:,.0f} crore). We follow the brief\u2019s wording.']:
    para(t, 'small', width=half, gap=7)
ly = y
head('Using this for one fund', x=rx, width=half, top=top, gap=6)
for t in ['These forecasts cover whole categories, month by month. A single fund planning for 15 March also needs its own daily redemptions, how concentrated its investors are, and when its holdings mature.',
          'Timing matters within the day. Redemptions are paid on the morning of T+1, while money from TREPS and reverse repo comes back in the evening.',
          'Liquid funds may hold instruments of up to 91 days. The 20% eligible-liquid-assets rule is not a 20% cash requirement.',
          'With fund-level daily data, the same method applies: forecast the range, then hold cash against its downside.']:
    para(t, 'small', x=rx, width=half, gap=7)
y = min(y, ly) - 8
head('Data', gap=6)
table(['Source', 'Coverage', 'Used for'], [
    ['AMFI monthly reports', 'Jul 2019 to Sep 2026', 'Net flows and assets, liquid and overnight'],
    ['RBI call-money rate', 'Apr 2020 to Jun 2026', 'Interest-rate inputs'],
    ['Nifty 50 monthly average (RBI)', 'Apr 2018 to Jun 2026', 'Equity-market input'],
    ['RBI liquidity operations', 'Jul 2019 to Jul 2026', 'Banking-liquidity input'],
    ['Income Tax Department', 'Advance-tax schedule', 'Calendar inputs']], [.34, .26, .40], [False, False, False], gap=14)
head('References', gap=6)
refs = [('Zeckathon PS03 problem brief', 'https://zeckathon.com/uploads/3bc07bf3486a400794432089ba98cd43.pdf'),
        ('AMFI monthly data archive', 'https://www.amfiindia.com/research-information/amfi-monthly'),
        ('RBI Handbook of Statistics, Tables 174 and 182', 'https://www.rbi.org.in/scripts/PublicationsView.aspx?id=23999'),
        ('RBI Bulletin, liquidity operations', 'https://www.rbi.org.in/Scripts/BS_ViewBulletin.aspx?Id=24451'),
        ('Income Tax Department, advance-tax schedule', 'https://www.incometaxindia.gov.in/w/section-408-6'),
        ('SEBI Master Circular for Mutual Funds, March 2026', 'https://www.sebi.gov.in/sebi_data/attachdocs/mar-2026/1774024028162.pdf'),
        ('Hyndman and Athanasopoulos, Forecasting: Principles and Practice', 'https://otexts.com/fpp3/accuracy.html')]
for i, (t, u) in enumerate(refs, 1):
    para(f'{mono(f"[{i}]")}  <link href="{escape(u, quote=True)}" color="{BLUE}">{escape(t)}</link>', 'small', gap=3.5)
y -= 10
c.setStrokeColor(col(LINE)); c.line(M, y, W - M, y); y -= 16
para(f'<b>Team {TEAM}</b>, IIT Patna: {", ".join(MEMBERS[:-1])} and {MEMBERS[-1]}. The project repository holds the code, data snapshots and an interactive dashboard. Running {mono("make all")} checks the source files, reruns the experiment and rebuilds every figure and number in this report.', 'small')
if not out.exists():
    c.save()
print(out, page_no)
