from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from datetime import datetime
import random

styles = getSampleStyleSheet()

title_style = styles["Heading1"]
title_style.alignment = TA_CENTER
title_style.textColor = colors.darkblue

heading_style = styles["Heading2"]
heading_style.textColor = colors.darkblue

normal = styles["BodyText"]

center = styles["BodyText"]
center.alignment = TA_CENTER

small = styles["BodyText"]
small.fontSize = 9

def get_severity(hb):

    if hb < 7:
        return "Severe"

    elif hb < 10:
        return "Moderate"

    elif hb < 12:
        return "Mild"

    return "Normal"


def get_recommendations(hb):

    if hb < 7:

        return {
            "foods":[
                "Spinach",
                "Liver",
                "Dates",
                "Beans",
                "Eggs",
                "Pomegranate"
            ],

            "avoid":[
                "Tea",
                "Coffee",
                "Junk Food"
            ],

            "follow":"Consult physician immediately."
        }

    elif hb < 10:

        return {

            "foods":[
                "Spinach",
                "Dates",
                "Eggs",
                "Chicken",
                "Orange"
            ],

            "avoid":[
                "Tea",
                "Coffee"
            ],

            "follow":"Repeat CBC after two weeks."

        }

    else:

        return {

            "foods":[
                "Balanced Diet",
                "Fruits",
                "Vegetables"
            ],

            "avoid":[
                "Fast Food"
            ],

            "follow":"Routine annual health check-up."

        }
    
def generate_pdf(
    filename,
    patient,
    age,
    gender,
    state,
    district,
    hb,
    mch,
    mchc,
    mcv,
    prediction,
    confidence
):

    doc = SimpleDocTemplate(
        filename,
        topMargin=25,
        bottomMargin=25,
        leftMargin=35,
        rightMargin=35
    )

    story = []

    report_id = f"HV-{datetime.now().year}-{random.randint(100000,999999)}"

    report_date = datetime.now().strftime("%d %B %Y")
    report_time = datetime.now().strftime("%I:%M %p")

    severity = get_severity(hb)

    rec = get_recommendations(hb)

    # ===================================
    # HEADER
    # ===================================

    story.append(
        Paragraph(
            "<b>🏥 HEMOVISION AI</b>",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-Based Multimodal Anemia Detection Report",
            center
        )
    )

    story.append(Spacer(1,8))

    report_table = Table([

        ["Report ID", report_id],

        ["Date", report_date],

        ["Time", report_time]

    ], colWidths=[120,260])

    report_table.setStyle(TableStyle([

        ("GRID",(0,0),(-1,-1),0.5,colors.grey),

        ("BACKGROUND",(0,0),(0,-1),colors.darkblue),

        ("TEXTCOLOR",(0,0),(0,-1),colors.white),

        ("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),

        ("BOTTOMPADDING",(0,0),(-1,-1),6)

    ]))

    story.append(report_table)

    story.append(Spacer(1,10))

    story.append(
        Paragraph(
            "Patient Summary",
            heading_style
        )
    )

    summary = Table([

        ["Patient",patient,"Prediction",prediction],

        ["Age",str(age),"Confidence",f"{confidence:.2f}%"],

        ["Gender",gender,"Severity",severity],

        ["State",state,"Status","Completed"],

        ["District",district,"",""]

    ], colWidths=[80,120,80,120])

    summary.setStyle(TableStyle([

        ("GRID",(0,0),(-1,-1),0.5,colors.grey),

        ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#0d6efd")),

        ("BACKGROUND",(2,0),(2,-1),colors.darkred),

        ("TEXTCOLOR",(0,0),(0,-1),colors.white),

        ("TEXTCOLOR",(2,0),(2,-1),colors.white),

        ("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),

        ("BOTTOMPADDING",(0,0),(-1,-1),6)

    ]))

    story.append(summary)

    story.append(Spacer(1,10))
    # ==========================================
    # CLINICAL PARAMETERS
    # ==========================================

    story.append(
        Paragraph(
            "Clinical Parameters",
            heading_style
        )
    )

    clinical = Table([

        ["Parameter","Patient","Normal"],

        ["Hemoglobin (Hb)",f"{hb} g/dL","12-16 g/dL"],

        ["MCH",f"{mch} pg","27-33 pg"],

        ["MCHC",f"{mchc} g/dL","32-36 g/dL"],

        ["MCV",f"{mcv} fL","80-100 fL"]

    ], colWidths=[170,100,120])

    clinical.setStyle(TableStyle([

        ("GRID",(0,0),(-1,-1),0.5,colors.grey),

        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0d6efd")),

        ("TEXTCOLOR",(0,0),(-1,0),colors.white),

        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),

        ("BACKGROUND",(0,1),(-1,-1),colors.beige),

        ("BOTTOMPADDING",(0,0),(-1,-1),6)

    ]))

    story.append(clinical)

    story.append(Spacer(1,10))

    # ==========================================
    # AI RESULT
    # ==========================================

    story.append(
        Paragraph(
            "AI Screening Result",
            heading_style
        )
    )

    result_color = colors.green
    result_text = "NORMAL"

    if prediction.lower() == "anemia":
        result_color = colors.red
        result_text = "ANEMIA DETECTED"

    result = Table([
        [result_text],
        [f"Confidence : {confidence:.2f}%"],
        [f"Severity : {severity}"]
    ], colWidths=[390])

    result.setStyle(TableStyle([

        ("GRID",(0,0),(-1,-1),1,result_color),

        ("BACKGROUND",(0,0),(-1,0),result_color),

        ("TEXTCOLOR",(0,0),(-1,0),colors.white),

        ("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),

        ("ALIGN",(0,0),(-1,-1),"CENTER"),

        ("BOTTOMPADDING",(0,0),(-1,-1),8)

    ]))

    story.append(result)

    story.append(Spacer(1,10))


    # ==========================================
    # AI CLINICAL SUMMARY
    # ==========================================

    story.append(
        Paragraph(
            "AI Clinical Summary",
            heading_style
        )
    )

    if prediction.lower() == "anemia":

        summary = (
            "The FusionNet multimodal AI model predicts that "
            "the patient may have anemia based on the uploaded "
            "clinical parameters. Laboratory confirmation with "
            "CBC and physician evaluation is recommended."
        )

    else:

        summary = (
            "The AI model predicts no significant indication of "
            "anemia. Continue maintaining a balanced diet and "
            "routine medical check-ups."
        )

    story.append(
        Paragraph(
            summary,
            normal
        )
    )

    story.append(Spacer(1,10))


    doc.build(story)