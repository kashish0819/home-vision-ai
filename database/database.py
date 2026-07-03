import sqlite3
import pandas as pd
from datetime import datetime


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_NAME = PROJECT_ROOT / "database" / "prediction.db"


def connect_db():
    return sqlite3.connect(str(DB_NAME), check_same_thread=False)


def create_table():

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    patient_id TEXT,

    age INTEGER,

    gender TEXT,

    state TEXT,

    district TEXT,

    image_path TEXT,

    hemoglobin REAL,
    mch REAL,
    mchc REAL,
    mcv REAL,

    prediction TEXT,
    severity TEXT,

    confidence REAL,

    date TEXT,
    time TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_prediction(
    patient_id,
    age,
    gender,
    state,
    district,
    image_path,
    hb,
    mch,
    mchc,
    mcv,
    prediction,
    severity,
    confidence
):
    
    conn = connect_db()
    cursor = conn.cursor()

    now = datetime.now()

    cursor.execute("""
    INSERT INTO predictions(

    patient_id,
    age,
    gender,
    state,
    district,
    image_path,
    hemoglobin,
    mch,
    mchc,
    mcv,
    prediction,
    severity,
    confidence,
    date,
    time

)
    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)

    """,(

        (
    patient_id,
    age,
    gender,
    state,
    district,
    image_path,
    hb,
    mch,
    mchc,
    mcv,
    prediction,
    severity,
    confidence,
    now.strftime("%Y-%m-%d"),
    now.strftime("%H:%M:%S")
)

    ))

    report_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return report_id

def load_data():

    conn = connect_db()

    df = pd.read_sql("SELECT * FROM predictions", conn)

    conn.close()

    return df


create_table()

# ---------------------------------------
# Get All Reports
# ---------------------------------------

def get_all_predictions():

    conn = connect_db()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            patient_id,
            age,
            gender,
            state,
            district,
            prediction,
            confidence,
            date,
            time
        FROM predictions
        ORDER BY id DESC
    """)

    data = cursor.fetchall()

    conn.close()

    return data


# ---------------------------------------
# Get One Report
# ---------------------------------------

def get_prediction(report_id):

    conn = connect_db()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM predictions
        WHERE id=?
    """, (report_id,))

    data = cursor.fetchone()

    conn.close()

    return data