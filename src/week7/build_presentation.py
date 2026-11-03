from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
NAVY=RGBColor(0x1F,0x38,0x64); TEAL=RGBColor(0x02,0x80,0x90); RED=RGBColor(0xC0,0x39,0x2B); GREY=RGBColor(0x59,0x59,0x59); WHITE=RGBColor(255,255,255); LIGHT=RGBColor(0xEE,0xF2,0xF7)
p=Presentation(); p.slide_width=Inches(13.333); p.slide_height=Inches(7.5)
B=p.slide_layouts[6]
def bg(s,c):
    f=s.background.fill; f.solid(); f.fore_color.rgb=c
def tx(s,x,y,w,h,t,sz=16,col=GREY,b=False,it=False):
    tb=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=tb.text_frame; tf.word_wrap=True
    lines=t if isinstance(t,list) else [t]
    for i,l in enumerate(lines):
        para=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        r=para.add_run(); r.text=l; r.font.size=Pt(sz); r.font.color.rgb=col; r.font.bold=b; r.font.italic=it; r.font.name="Calibri"
        para.space_after=Pt(8)
    return tb
def title(s,t,sub=None):
    tx(s,0.6,0.35,12.2,0.9,t,32,NAVY,True)
    if sub: tx(s,0.6,1.1,12.2,0.5,sub,16,GREY,it=True)
def card(s,x,y,w,h,big,lab,col=NAVY):
    sh=s.shapes.add_shape(1,Inches(x),Inches(y),Inches(w),Inches(h)); sh.fill.solid(); sh.fill.fore_color.rgb=LIGHT; sh.line.fill.background()
    tx(s,x+0.2,y+0.15,w-0.4,1,big,40,col,True); tx(s,x+0.2,y+1.15,w-0.4,h-1.2,lab,14,GREY)
def bar(s,x,y,w,h,cats,series,ttl,fmt='0.00'):
    cd=CategoryChartData(); cd.categories=cats
    for n,v in series: cd.add_series(n,v)
    ch=s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,Inches(x),Inches(y),Inches(w),Inches(h),cd).chart
    ch.has_title=True; ch.chart_title.text_frame.text=ttl; ch.chart_title.text_frame.paragraphs[0].runs[0].font.size=Pt(14)
    pl=ch.plots[0]; pl.has_data_labels=True; pl.data_labels.number_format=fmt; pl.data_labels.number_format_is_linked=False; pl.data_labels.font.size=Pt(11)
    ch.has_legend=len(series)>1
    if ch.has_legend: ch.legend.include_in_layout=False; ch.legend.font.size=Pt(11)
    ch.value_axis.has_major_gridlines=False; ch.value_axis.tick_labels.font.size=Pt(10); ch.category_axis.tick_labels.font.size=Pt(11)
    cols=[NAVY,TEAL,RED]
    for i,se in enumerate(pl.series): se.format.fill.solid(); se.format.fill.fore_color.rgb=cols[i%3]

# 1 title
s=p.slides.add_slide(B); bg(s,NAVY)
tx(s,0.8,2.2,11.5,1.2,"AI Exam Answer Grader",48,WHITE,True)
tx(s,0.8,3.4,11.5,0.8,"From handwritten script to recommended mark, tested on real teacher grades",22,RGBColor(0xCA,0xDC,0xFC))
tx(s,0.8,5.6,11.5,0.8,"Group 2 · NLP Course Project 2026 · AI University Platform · Final presentation (Weeks 7–8)",16,RGBColor(0xCA,0xDC,0xFC))

# 2 pipeline
s=p.slides.add_slide(B); title(s,"What we built","A triage aid: it suggests a mark, explains it, and flags answers a human must check")
steps=[("Group 1 OCR","Scanned scripts → text per question, with OCR confidence"),("Adapter (Week 7)","Merges pages, reads Group 1's real CSV, grades MCQs by letter"),("NLP grader","Rubric points checked with lexical, embedding and NLI signals"),("Review flags","Low OCR confidence, missing text, contradictions → human"),("Teacher","Keeps final authority over every mark")]
for i,(a,b) in enumerate(steps):
    x=0.6+i*2.5
    sh=s.shapes.add_shape(5,Inches(x),Inches(2.3),Inches(2.2),Inches(2.6)); sh.fill.solid(); sh.fill.fore_color.rgb=NAVY if i!=4 else TEAL; sh.line.fill.background()
    tx(s,x+0.15,2.45,1.9,0.7,a,17,WHITE,True); tx(s,x+0.15,3.25,1.9,1.6,b,13,WHITE)
tx(s,0.6,5.4,12,1.2,["API: POST /evaluate-answer · POST /grade-digitized-answers (Group 1 → Group 2) · FastAPI, 99 automated tests passing"],15,GREY)

# 3 audit
s=p.slides.add_slide(B); title(s,"Our own benchmark had a shortcut","On the original 70 answers, counting words graded as well as our baseline")
card(s,0.6,1.9,3.8,2.2,"0.82","Spearman correlation between answer length and score in our synthetic data")
card(s,4.7,1.9,3.8,2.2,"1.25","MAE of a word-count-only model (TF-IDF baseline: 1.27)",RED)
card(s,8.8,1.9,3.8,2.2,"−0.09","length–score correlation in our new 85-answer challenge set",TEAL)
tx(s,0.6,4.5,12,2.5,["Fix: a challenge set where length says nothing about quality (terse correct, verbose irrelevant, swapped facts, prompt injection, Hinglish, typos).","Statistics now clustered by question; 'better' only when the 95% CI excludes zero."],16)

# 4 synthetic results
s=p.slides.add_slide(B); title(s,"Transformers help on unseen questions","Leave-one-question-out on the original 70 (A) and train-70 → challenge-85 (B); MAE in marks, lower is better")
bar(s,0.6,1.7,7.2,5.2,["Length only","TF-IDF (Wk 2)","spaCy hybrid (Wk 6)","Lexical rubric","NLI (DeBERTa)"],[("A: unseen questions",[1.25,1.271,1.093,1.129,0.90])],"Protocol A MAE")
bar(s,8.1,1.7,4.8,5.2,["Length","TF-IDF","spaCy hybrid","Lexical"],[("B: challenge set",[3.47,2.43,2.95,2.01])],"Protocol B MAE (challenge)")

# 5 week 7 data
s=p.slides.add_slide(B); title(s,"Week 7: real handwritten scripts from Group 1","5 students · 35 questions (20 MCQ + 15 short, 2 marks each) · teacher's red-pen marks read off the scans")
card(s,0.6,1.9,3.8,2.2,"65","short answers attempted, each with a real teacher mark (first human grades in the project)")
card(s,4.7,1.9,3.8,2.2,"5 / 5","students whose per-question marks add up exactly to the circled total",TEAL)
card(s,8.8,1.9,3.8,2.2,"41 / 65","answers the teacher gave full marks: a lenient grader",RED)
tx(s,0.6,4.5,12,2.5,["MCQ: OCR letters match the scans 99/100. The faculty answer key says Q1 = B, but the teacher's ticks accept D on every script — a key error we report, not silently fix.","Two exam keys built from the answer key: auto-split reference vs a two-point authored rubric."],16)

# 6 OCR quality
s=p.slides.add_slide(B); title(s,"How good is Group 1's OCR?","Group 1 text compared with our transcription of the 65 attempted answers")
card(s,0.6,1.9,2.9,2.2,"5.6%","median character error rate on found answers")
card(s,3.7,1.9,2.9,2.2,"3","answers missed entirely (teacher gave them 2, 2 and 1)",RED)
card(s,6.8,1.9,2.9,2.2,"2","answers cut in half: the continuation page was lost",RED)
card(s,9.9,1.9,2.9,2.2,"11 / 15","damaged answers caught by 'OCR confidence < 0.8' (precision 0.65)",TEAL)
tx(s,0.6,4.5,12,2.5,["Group 1's own needs_review flag caught only 4 of the 15 damaged answers.","Recommendation: route any answer with OCR confidence < 0.8 or status not_found to a human."],16)

# 7 real grading
s=p.slides.add_slide(B); title(s,"Grading real answers: honest result","Teacher mark 0–2; graders trained only on synthetic data; lexical-rubric grader shown")
bar(s,0.6,1.7,7.4,5.2,["Predict avg","Length","TF-IDF","Lexical (auto key)","Lexical (authored)"],[("Clean transcription",[0.769,0.869,1.054,1.246,0.785]),("Group 1 OCR text",[0.815,0.9,1.123,1.308,0.862])],"MAE vs teacher (marks out of 2)",'0.00')
tx(s,8.3,1.9,4.6,5,["Zero-shot graders are much stricter than this teacher.","The authored rubric beats the mechanical one (exact agreement 43% vs 23%).","OCR errors add ~0.08 marks of error per answer.","After calibrating to the teacher, nothing beats 'predict the teacher's average' (MAE ≈ 0.66). Use it as a ranking + feedback aid, not a mark."],15)

# 8 noise + external
s=p.slides.add_slide(B); title(s,"Robustness and outside validation","Synthetic OCR noise on the challenge set · public human-graded Mohler dataset (2,273 answers)")
bar(s,0.6,1.7,6.2,5.2,["0%","5%","10%","20%"],[("TF-IDF",[0.442,0.338,0.279,0.143]),("Lexical",[0.519,0.388,0.206,0.003])],"QWK as character error rate rises",'0.00')
bar(s,7.0,1.7,5.9,5.2,["Constant","Length","TF-IDF","Lexical"],[("Mohler QWK",[0.0,0.021,0.365,0.367])],"Mohler (real graders): QWK",'0.00')
tx(s,7.1,6.85,5.8,0.5,"Human vs human QWK on Mohler: 0.44 · length gives no signal on real data",12,GREY,it=True)

# 9 responsible AI
s=p.slides.add_slide(B); title(s,"Responsible AI","What stops this tool from hurting a student")
items=[("Human final say","Every score is a recommendation; the API never writes a mark."),("Two independent flags","OCR confidence and grader confidence measure different things, so both route to a human."),("Prompt-injection safe","Answer is data; LLM judge must quote evidence or gets no credit."),("Known bias","Stricter than a lenient teacher; tested on 5 students and one teacher only.")]
for i,(a,b) in enumerate(items):
    x=0.6+(i%2)*6.2; y=1.8+(i//2)*2.5
    sh=s.shapes.add_shape(1,Inches(x),Inches(y),Inches(5.9),Inches(2.2)); sh.fill.solid(); sh.fill.fore_color.rgb=LIGHT; sh.line.fill.background()
    tx(s,x+0.25,y+0.2,5.4,0.6,a,20,NAVY,True); tx(s,x+0.25,y+0.9,5.4,1.2,b,15,GREY)

# 10 close
s=p.slides.add_slide(B); bg(s,NAVY)
tx(s,0.8,0.8,11.5,0.9,"Takeaways and next steps",36,WHITE,True)
tx(s,0.8,2.0,11.5,4.8,["End-to-end Group 1 → Group 2 pipeline works on real scripts, with MCQ and short answers.","Weak OCR confidence is the single best review trigger; missed and truncated pages are the main OCR failure.","NLI beats our Week 6 model on unseen synthetic questions (MAE 0.90 vs 1.09), but real-teacher agreement is not there yet.","Next: more real scripts and teachers, faculty-written rubrics, calibrate per course, run the LLM judge."],20,WHITE)
p.save("Group2_Final_Presentation.pptx")
