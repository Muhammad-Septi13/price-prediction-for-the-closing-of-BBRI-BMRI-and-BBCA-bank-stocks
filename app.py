# ============================================================
# STREAMLIT APP - REVISI FINAL & SEMPURNA VISUALISASI
# Features: 2 Model Global (LSTM & BiLSTM tuning), Data Bawaan,
#           Upload Excel/CSV, Radar Metric Chart, Premium Plotly.
# ============================================================

import os
# Paksa TensorFlow menggunakan CPU untuk mencegah freeze/crash saat inferensi di beberapa perangkat
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

from sklearn.metrics import mean_squared_error, mean_absolute_error

# ===============================
# KONFIGURASI & STRUKTUR FOLDER
# ===============================
WINDOW_SIZE = 5
FEATURES    = ['Open', 'High', 'Low', 'Close']
N_FEATURES  = 4
SAHAM_LIST  = ["BBCA", "BBRI", "BMRI"]

# Otomatis buat folder jika belum ada agar tidak memicu FileNotFoundError
for folder in ["models", "scalers", "data"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

MODEL_FILES = {

    "LSTM":
    "models/best_LSTM_global.h5",

    "BiLSTM":
    "models/best_BiLSTM_global.h5"

}


SCALER_FILES = {

    "LSTM":
    "scalers/scaler_LSTM_global.pkl",

    "BiLSTM":
    "scalers/scaler_BiLSTM_global.pkl"

}

BUILTIN_DATA = {s: f"data/{s}_2024_2024.csv" for s in SAHAM_LIST}

# ---------------------------------------------------------------
# GANTI SESUAI HASIL HYPERPARAMETER TUNING KAMU
# ---------------------------------------------------------------
BEST_LSTM_KEY = "LSTM"

BEST_BILSTM_KEY = "BiLSTM"

st.set_page_config(page_title="Prediksi Harga Saham", page_icon="📈", layout="wide")

# ===============================
# PREMIUM CSS STYLING
# ===============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2.5rem; border-radius: 20px; margin-bottom: 2rem;
        text-align: center; box-shadow: 0 10px 40px rgba(59,130,246,0.25);
    }
    .main-header h1 { color:white; font-size:2.2rem; font-weight:700; margin-bottom:0.5rem; }
    .main-header p  { color:rgba(255,255,255,0.9); font-size:1rem; margin:0; }

    /* Model info cards */
    .model-card {
        padding: 1.2rem 1.5rem; border-radius: 14px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08); margin-bottom: 0.5rem;
    }
    .card-lstm   { background:linear-gradient(135deg,#eff6ff,#dbeafe); border-left:5px solid #3b82f6; }
    .card-bilstm { background:linear-gradient(135deg,#f0fdf4,#dcfce7); border-left:5px solid #22c55e; }
    .card-title  { font-weight:700; font-size:1rem; margin-bottom:0.25rem; }
    .card-badge  {
        display:inline-block; background:#f59e0b; color:white;
        padding:0.15rem 0.6rem; border-radius:999px;
        font-size:0.75rem; font-weight:700; margin-left:0.5rem;
    }
    .card-meta { font-size:0.82rem; color:#4b5563; }

    /* Section header */
    .section-header {
        background:linear-gradient(90deg,#f8fafc,#ffffff);
        padding:1rem 1.5rem; border-radius:12px;
        margin:2rem 0 1.5rem 0; border-left:4px solid #3b82f6;
    }
    .section-header h2 { color:#1f2937; font-size:1.25rem; font-weight:600; margin:0; }

    /* Metric comparison cards */
    .metric-header {
        text-align:center; padding:0.75rem;
        border-radius:10px; margin-bottom:1rem; font-weight:700; font-size:1rem;
    }
    .metric-header-lstm   { background:linear-gradient(135deg,#3b82f6,#1d4ed8); color:white; }
    .metric-header-bilstm { background:linear-gradient(135deg,#22c55e,#15803d); color:white; }

    .mcard {
        border-radius: 12px; padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        position: relative;
    }
    .mcard-lstm   { background:white; border-left:4px solid #3b82f6; }
    .mcard-bilstm { background:white; border-left:4px solid #22c55e; }
    .mcard-win-lstm   { background:#eff6ff; border-left:4px solid #3b82f6; box-shadow: 0 2px 16px rgba(59,130,246,0.18); }
    .mcard-win-bilstm { background:#f0fdf4; border-left:4px solid #22c55e; box-shadow: 0 2px 16px rgba(34,197,94,0.18); }

    .mcard-label { font-size:0.75rem; color:#6b7280; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.2rem; }
    .mcard-value { font-size:1.5rem; font-weight:700; }
    .mcard-value-lstm   { color:#3b82f6; }
    .mcard-value-bilstm { color:#22c55e; }
    .mcard-win-badge {
        position:absolute; top:0.7rem; right:0.8rem;
        background:#f59e0b; color:white;
        padding:0.1rem 0.5rem; border-radius:999px;
        font-size:0.7rem; font-weight:700;
    }

    /* Winner banner */
    .winner-banner {
        background:linear-gradient(135deg,#fef3c7,#fde68a);
        border:1px solid #f59e0b; border-left:5px solid #f59e0b;
        padding:0.9rem 1.2rem; border-radius:12px; margin:1rem 0;
    }
    .winner-banner p { margin:0; color:#92400e; font-size:0.9rem; }

    .forecast-container {
        background:white; padding:1.5rem;
        border-radius:16px; box-shadow:0 4px 20px rgba(0,0,0,0.08);
    }

    [data-testid="stSidebar"] {
        background:linear-gradient(180deg,#1e293b 0%,#0f172a 100%);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stRadio label { color:white !important; }

    .stButton > button {
        background:linear-gradient(135deg,#3b82f6,#1d4ed8);
        color:white; border:none; padding:0.75rem 2rem;
        font-size:1rem; font-weight:600; border-radius:12px;
        width:100%; transition:all 0.3s ease;
    }
    .stButton > button:hover {
        transform:translateY(-2px);
        box-shadow:0 8px 25px rgba(59,130,246,0.4);
    }
    #MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ===============================
# HELPER FUNCTIONS
# ===============================
def split_data(data):
    """
    Split data menjadi train validation test
    70 : 15 : 15
    """

    train_size = int(len(data) * 0.70)
    val_size = int(len(data) * 0.15)

    train = data[:train_size]

    val = data[
        train_size:
        train_size + val_size
    ]

    test = data[
        train_size + val_size:
    ]

    return train, val, test



def evaluate_dataset(model, scaler, data):

    scaled = scaler.transform(data)


    X, y_scaled = create_window(
        scaled,
        WINDOW_SIZE
    )


    if len(X) == 0:
        return None


    pred_scaled = model.predict(
        X,
        verbose=0
    )


    y_actual = inverse_close(
        scaler,
        y_scaled
    )


    y_pred = inverse_close(
        scaler,
        pred_scaled
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_actual,
            y_pred
        )
    )


    mae = mean_absolute_error(
        y_actual,
        y_pred
    )


    mask = y_actual != 0

    mape = np.mean(
        np.abs(
            (y_actual[mask]-y_pred[mask])
            /
            y_actual[mask]
        )
    ) * 100


    return {
        "RMSE":rmse,
        "MAE":mae,
        "MAPE":mape,
        "actual":y_actual,
        "pred":y_pred
    }

def create_window(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i, 3])
    return np.array(X), np.array(y)

def inverse_close(scaler, s):
    dummy = np.zeros((len(s), N_FEATURES))
    dummy[:, 3] = s.reshape(-1)
    return scaler.inverse_transform(dummy)[:, 3]

def evaluate_model(model, scaler, test_data):
    test_scaled = scaler.transform(test_data)
    X_test, y_test_scaled = create_window(test_scaled, WINDOW_SIZE)
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_actual = inverse_close(scaler, y_test_scaled)
    y_pred   = inverse_close(scaler, y_pred_scaled)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae  = mean_absolute_error(y_actual, y_pred)
    
    # Penanganan pembagian dengan nol pada MAPE
    mask = y_actual != 0
    mape = np.mean(np.abs((y_actual[mask] - y_pred[mask]) / y_actual[mask])) * 100
    return y_actual, y_pred, rmse, mae, mape, X_test

def direction_accuracy(y_actual, y_pred):
    actual_dir = np.where(y_actual[1:] > y_actual[:-1], 1, 0)
    pred_dir   = np.where(y_pred[1:]   > y_actual[:-1], 1, 0)
    return np.mean(actual_dir == pred_dir) * 100

def recursive_forecast(model, scaler, last_window, n_days):
    preds = []
    w = last_window.copy()
    for _ in range(n_days):
        ps = model.predict(w.reshape(1, WINDOW_SIZE, N_FEATURES), verbose=0)
        dummy = np.zeros((1, N_FEATURES))
        dummy[:, 3] = ps.reshape(-1)
        p = scaler.inverse_transform(dummy)[0, 3]
        preds.append(p)
        new_row = w[-1].copy()
        new_row[3] = ps[0, 0] if hasattr(ps, "shape") and len(ps.shape) > 1 else ps[0]
        w = np.vstack([w[1:], new_row])
    return preds

def generate_excel_template():
    df_t = pd.DataFrame({
        "Date":  ["2024-01-02","2024-01-03","2024-01-04","2024-01-05","2024-01-08"],
        "Open":  [9000,9050,9100,9075,9125],
        "High":  [9100,9150,9200,9150,9200],
        "Low":   [8950,9000,9050,9025,9075],
        "Close": [9050,9100,9150,9100,9150],
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_t.to_excel(writer, index=False, sheet_name="Data Saham")
    return buf.getvalue()

# ===============================
# EKSEKUSI PREDIKSI & VISUALISASI
# ===============================
def show_prediction(all_data, all_dates, asset_name, n_days, container, show_forecast=True):
    if all_data is None:

        container.warning(
            "Silakan masukkan data terlebih dahulu."
        )

        return

    start = time.time()

    with container:
        train_data, val_data, test_data = split_data(
            all_data
        )


        if len(test_data) <= WINDOW_SIZE:
            container.error(f"❌ Data test terlalu sedikit ({len(test_data)} baris).")
            return

        train_size = len(train_data)
        val_size = len(val_data)

        test_dates = all_dates[
            train_size + val_size :
        ]

        # ===== EVALUASI KEDUA MODEL =====
        with st.spinner("Mengevaluasi inferensi model deep learning..."):
            t0 = time.time()
            y_act_lstm, y_pred_lstm, rmse_lstm, mae_lstm, mape_lstm, X_test = evaluate_model(
                model_lstm, scaler_lstm, test_data)
            time_lstm = time.time() - t0

            t0 = time.time()
            y_act_bi, y_pred_bi, rmse_bi, mae_bi, mape_bi, X_test_bilstm  = evaluate_model(
                model_bilstm, scaler_bilstm, test_data)
            time_bi = time.time() - t0

        dir_lstm = direction_accuracy(y_act_lstm, y_pred_lstm)
        dir_bi   = direction_accuracy(y_act_bi,   y_pred_bi)

        # ===================================================
        # METRIKS EVALUASI PERBANDINGAN BERWARNA
        # ===================================================
        container.markdown('<div class="section-header"><h2>📊 Evaluasi Model pada Data Test</h2></div>', unsafe_allow_html=True)

        col_spacer, col_h1, col_h2 = st.columns([1.2, 1, 1])
        with col_h1:
            st.markdown(f"""<div class="metric-header metric-header-lstm">🔵 {lstm_label}<br><span style="font-size:0.75rem; font-weight:400;"></span></div>""", unsafe_allow_html=True)
        with col_h2:
            st.markdown(f"""<div class="metric-header metric-header-bilstm">🟢 {bilstm_label}<br><span style="font-size:0.75rem; font-weight:400;"></span></div>""", unsafe_allow_html=True)

        metrics = [
            ("RMSE",          f"Rp {rmse_lstm:,.2f}",   f"Rp {rmse_bi:,.2f}",   rmse_lstm < rmse_bi,   rmse_lstm > rmse_bi),
            ("MAE",           f"Rp {mae_lstm:,.2f}",    f"Rp {mae_bi:,.2f}",    mae_lstm  < mae_bi,    mae_lstm  > mae_bi),
            ("MAPE",          f"{mape_lstm:.2f}%",       f"{mape_bi:.2f}%",      mape_lstm < mape_bi,   mape_lstm > mape_bi),
            # ("Akurasi Arah",  f"{dir_lstm:.1f}%",        f"{dir_bi:.1f}%",       dir_lstm  > dir_bi,    dir_lstm  < dir_bi),
            ("Waktu Proses",  f"{time_lstm:.3f} s",     f"{time_bi:.3f} s",     time_lstm < time_bi,   time_lstm > time_bi),
        ]

        for label, val_lstm, val_bi, lstm_wins, bi_wins in metrics:
            col_label, col_lstm, col_bi = st.columns([1.2, 1, 1])
            with col_label:
                st.markdown(f"""<div style="padding:0.9rem 0; font-weight:600; font-size:0.9rem; color:#374151; border-bottom:1px solid #f1f5f9;">📌 {label}</div>""", unsafe_allow_html=True)
            with col_lstm:
                win_class = "mcard-win-lstm" if lstm_wins else "mcard-lstm"
                win_badge = '<span class="mcard-win-badge">✓ UNGGUL</span>' if lstm_wins else ""
                st.markdown(f"""<div class="mcard {win_class}">{win_badge}<div class="mcard-label">{lstm_label}</div><div class="mcard-value mcard-value-lstm">{val_lstm}</div></div>""", unsafe_allow_html=True)
            with col_bi:
                win_class = "mcard-win-bilstm" if bi_wins else "mcard-bilstm"
                win_badge = '<span class="mcard-win-badge">✓ UNGGUL</span>' if bi_wins else ""
                st.markdown(f"""<div class="mcard {win_class}">{win_badge}<div class="mcard-label">{bilstm_label}</div><div class="mcard-value mcard-value-bilstm">{val_bi}</div></div>""", unsafe_allow_html=True)

        # --- RADAR CHART (VISUALISASI 1) ---
        # st.markdown("<br>", unsafe_allow_html=True)
        # max_rmse, max_mae, max_mape = max(rmse_lstm, rmse_bi) or 1, max(mae_lstm, mae_bi) or 1, max(mape_lstm, mape_bi) or 1
        # max_time = max(time_lstm, time_bi) or 1

        # radar_lstm = [round((1 - rmse_lstm/max_rmse)*100, 1), round((1 - mae_lstm/max_mae)*100, 1), round((1 - mape_lstm/max_mape)*100, 1), round(dir_lstm, 1), round((1 - time_lstm/max_time)*100, 1)]
        # radar_bi   = [round((1 - rmse_bi/max_rmse)*100, 1), round((1 - mae_bi/max_mae)*100, 1), round((1 - mape_bi/max_mape)*100, 1), round(dir_bi, 1), round((1 - time_bi/max_time)*100, 1)]
        # radar_labels = ["RMSE (Inverted)", "MAE (Inverted)", "MAPE (Inverted)", "Akurasi Arah", "Efisiensi Waktu"]

        # fig_radar = go.Figure()
        # fig_radar.add_trace(go.Scatterpolar(r=radar_lstm + [radar_lstm[0]], theta=radar_labels + [radar_labels[0]], fill='toself', name=lstm_label, line=dict(color='#3b82f6', width=2), fillcolor='rgba(59,130,246,0.15)'))
        # fig_radar.add_trace(go.Scatterpolar(r=radar_bi + [radar_bi[0]], theta=radar_labels + [radar_labels[0]], fill='toself', name=bilstm_label, line=dict(color='#22c55e', width=2), fillcolor='rgba(34,197,94,0.15)'))
        # fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor='#e2e8f0')), height=380, margin=dict(l=40, r=40, t=30, b=30), title=dict(text="Radar Performa (Skor Makin Lebar = Makin Baik)", font=dict(size=14), x=0.5))
        # st.plotly_chart(fig_radar, use_container_width=True)

        # --- RINGKASAN SKOR ---
        # lstm_wins_count = sum([rmse_lstm < rmse_bi, mae_lstm < mae_bi, mape_lstm < mape_bi, dir_lstm > dir_bi, time_lstm < time_bi])
        # bi_wins_count = 5 - lstm_wins_count

        # col_s1, col_s2 = st.columns(2)
        # with col_s1:
        #     st.markdown(f"""<div style="background:linear-gradient(135deg,#eff6ff,#dbeafe); border-radius:14px; padding:1rem; text-align:center; border:1px solid #bfdbfe;"><div style="font-size:0.85rem; color:#1d4ed8; font-weight:600;">🔵 {lstm_label}</div><div style="font-size:2.3rem; font-weight:700; color:#3b82f6;">{lstm_wins_count}/5</div><div style="font-size:0.8rem; color:#4b5563;">Metrik Unggul</div></div>""", unsafe_allow_html=True)
        # with col_s2:
        #     st.markdown(f"""<div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7); border-radius:14px; padding:1rem; text-align:center; border:1px solid #bbf7d0;"><div style="font-size:0.85rem; color:#15803d; font-weight:600;">🟢 {bilstm_label}</div><div style="font-size:2.3rem; font-weight:700; color:#22c55e;">{bi_wins_count}/5</div><div style="font-size:0.8rem; color:#4b5563;">Metrik Unggul</div></div>""", unsafe_allow_html=True)

        # st.markdown(f"""<div class="winner-banner"><p>⭐ <b>Kesimpulan:</b> <b>{lstm_label if lstm_wins_count >= bi_wins_count else bilstm_label}</b> disarankan untuk analisis teknikal ini karena mendominasi skor evaluasi biner data pengujian.</p></div>""", unsafe_allow_html=True)

        # --- LINE CHART INTERAKTIF (VISUALISASI 2) ---
        st.markdown('<div class="section-header"><h2>📈 Grafik Perbandingan Harga Aktual vs Prediksi</h2></div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-tip">💡 <b>Tips:</b> Gunakan slider di bawah grafik untuk zoom periode tertentu. Klik legenda untuk sembunyikan/tampilkan garis. Hover untuk lihat nilai detail.</div>', unsafe_allow_html=True)
        
        pred_dates = test_dates[WINDOW_SIZE:]

        actual_close = y_act_lstm
    
        # Error band (selisih absolut prediksi dari aktual)
        error_lstm = np.abs(y_pred_lstm - actual_close)
        error_bi   = np.abs(y_pred_bi   - actual_close)
    
        # Subplot: atas = harga, bawah = error absolut
        fig_line = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.72, 0.28],
            vertical_spacing=0.06,
            subplot_titles=("Harga Aktual vs Prediksi", "Error Absolut per Hari")
        )
    
        # ── Panel atas: harga ──
        # Area aktual
        fig_line.add_trace(
            go.Scatter(
                x=pred_dates,
                y=actual_close,
                name="Harga Aktual",
                mode="lines",
                line=dict(
                    width=3,
                    dash="solid"
                ),
                hovertemplate=
                "Aktual : Rp %{y:,.0f}<extra></extra>"
            )
        ) 
        # Prediksi LSTM dengan area kepercayaan ±error_lstm
        fig_line.add_trace(go.Scatter(
            x=np.concatenate([pred_dates, pred_dates[::-1]]),
            y=np.concatenate([y_pred_lstm + error_lstm*0.5, (y_pred_lstm - error_lstm*0.5)[::-1]]),
            fill='toself', fillcolor='rgba(59,130,246,0.08)',
            line=dict(width=0), showlegend=False, hoverinfo='skip', name="band_lstm"
        ), row=1, col=1)
        fig_line.add_trace(
            go.Scatter(
                x=pred_dates,
                y=y_pred_lstm,
                name="LSTM ",
                mode="lines",
                line=dict(
                    width=3,
                    shape="spline"
                ),
                hovertemplate=
                "LSTM : Rp %{y:,.0f}<extra></extra>"
            )
        )
    
        # Prediksi BiLSTM dengan area kepercayaan
        fig_line.add_trace(go.Scatter(
            x=np.concatenate([pred_dates, pred_dates[::-1]]),
            y=np.concatenate([y_pred_bi + error_bi*0.5, (y_pred_bi - error_bi*0.5)[::-1]]),
            fill='toself', fillcolor='rgba(34,197,94,0.08)',
            line=dict(width=0), showlegend=False, hoverinfo='skip', name="band_bilstm"
        ), row=1, col=1)
        fig_line.add_trace(
            go.Scatter(
                x=pred_dates,
                y=y_pred_bi,
                name="BiLSTM ",
                mode="lines",
                line=dict(
                    width=3,
                    dash="dot",
                    shape="spline"
                ),
                hovertemplate=
                "BiLSTM : Rp %{y:,.0f}<extra></extra>"
            )
        )
    
        # Marker harga terakhir
        fig_line.add_annotation(
            x=pred_dates[-1], y=actual_close[-1],
            text=f"Rp {actual_close[-1]:,.0f}",
            showarrow=True, arrowhead=2, arrowcolor="#1e293b",
            bgcolor="#1e293b", font=dict(color="white", size=11),
            ax=55, ay=-30, row=1, col=1
        )
    
        # ── Panel bawah: error absolut ──
        fig_line.add_trace(go.Bar(
            x=pred_dates, y=error_lstm,
            name=f"Error {lstm_label}",
            marker_color="rgba(59,130,246,0.5)",
            hovertemplate="Error LSTM: Rp %{y:,.0f}<extra></extra>"
        ), row=2, col=1)
        fig_line.add_trace(go.Bar(
            x=pred_dates, y=error_bi,
            name=f"Error {bilstm_label}",
            marker_color="rgba(34,197,94,0.5)",
            hovertemplate="Error BiLSTM: Rp %{y:,.0f}<extra></extra>"
        ), row=2, col=1)
    
        fig_line.update_layout(
            height=600,
            template="plotly_white",
            hovermode="x unified",
            title=dict(
                text="Perbandingan Harga Aktual vs Prediksi LSTM dan BiLSTM",
                x=0.5,
                y=0.97,
                font=dict(size=18)
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.15,
                xanchor="center",
                x=0.5,
                font=dict(size=11),
                bgcolor="rgba(0,0,0,0.85)",
                bordercolor="#FFF7F7",
                borderwidth=1
            ),
                margin=dict(
                l=20,
                r=20,
                t=120,
                b=20
            ),
            barmode="group"
        )
    
        # Annotasi rata-rata error
        fig_line.add_hline(y=error_lstm.mean(), line_dash="dash", line_color="#3b82f6",
                        annotation_text=f"Rata-rata error LSTM: Rp {error_lstm.mean():,.0f}",
                        annotation_position="top left", row=2, col=1)
        fig_line.add_hline(y=error_bi.mean(),   line_dash="dash", line_color="#22c55e",
                        annotation_text=f"Rata-rata error BiLSTM: Rp {error_bi.mean():,.0f}",
                        annotation_position="top right", row=2, col=1)
    
        container.plotly_chart(fig_line, use_container_width=True)
        if show_forecast:
            # --- FORECAST BAR CHART FUTURE (VISUALISASI 3) ---
            st.markdown(f'<div class="section-header"><h2>🔮 Proyeksi Rekursif {n_days} Hari ke Depan — {asset_name}</h2></div>', unsafe_allow_html=True)
            # st.markdown(f'<div class="section-header"><h2>🔮 Proyeksi Rekursif {n_days} Hari ke Depan — {asset_name}</h2></div>')

            if len(X_test) == 0:
                container.error("Data terlalu sedikit untuk melakukan prediksi.")
                return

            lstm_fc   = recursive_forecast(model_lstm,   scaler_lstm,   X_test[-1], n_days)
            bilstm_fc = recursive_forecast(
            model_bilstm,
            scaler_bilstm,
            X_test_bilstm[-1],
            n_days
            )

            def change_label(arr, i):
                if i == 0: return "—"
                diff = arr[i] - arr[i-1]
                return f"▲ Rp {diff:,.0f}" if diff >= 0 else f"▼ Rp {abs(diff):,.0f}"

            future_df = pd.DataFrame({
                "Hari ke-":                  range(1, n_days+1),
                f"🔵 {lstm_label} (IDR)":    [f"Rp {x:,.0f}" for x in lstm_fc],
                # "Tren Perubahan":            [change_label(lstm_fc, i) for i in range(n_days)],
                f"🟢 {bilstm_label} (IDR)":  [f"Rp {x:,.0f}" for x in bilstm_fc],
                "Selisih Model":             [f"Rp {abs(lstm_fc[i]-bilstm_fc[i]):,.0f}" for i in range(n_days)],
            })

            col_f1, col_f2 = st.columns([1, 1])
            with col_f1:
                st.markdown('<div class="forecast-container">', unsafe_allow_html=True)
                st.dataframe(future_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_f2:
                hari_labels = [f"Hari {i}" for i in range(1, n_days+1)]
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(x=hari_labels, y=lstm_fc, name=lstm_label, marker_color="#3b82f6", text=[f"Rp {x:,.0f}" for x in lstm_fc], textposition="outside"))
                fig_bar.add_trace(go.Bar(x=hari_labels, y=bilstm_fc, name=bilstm_label, marker_color="#22c55e", text=[f"Rp {x:,.0f}" for x in bilstm_fc], textposition="outside"))
                
                # Penyesuaian rentang Y Axis agar label angka di atas batang tidak terpotong
                min_val = min(min(lstm_fc), min(bilstm_fc)) * 0.95
                max_val = max(max(lstm_fc), max(bilstm_fc)) * 1.05
                
                fig_bar.update_layout(barmode="group", height=380, template="plotly_white", yaxis=dict(range=[min_val, max_val], tickformat=",.0f"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_bar, use_container_width=True)

            # --- PREMIUM FOOTER LOG ---
            total_time = time.time() - start
            st.markdown(f"""
            <style>.footer-blue {{ background:linear-gradient(90deg,#1e293b,#3b82f6); padding:1.2rem; border-radius:15px; text-align:center; margin-top:2rem; color:white; box-shadow:0 4px 15px rgba(59,130,246,0.25); }}</style>
            <div class="footer-blue">⏱️ <b>Total Execution Time:</b> {total_time:.3f}s &nbsp;|&nbsp; 📊 📊 <b>Asset Code:</b> {asset_name} &nbsp;|&nbsp; 📅 <b>Forecast Window:</b> {n_days} Days &nbsp;|&nbsp; 🤖 <b>Primary System:</b> {lstm_label}</div>
            """, unsafe_allow_html=True)

# ===============================
# LOAD MODEL (cached)
# ===============================
@st.cache_resource
def load_best_models():
    # Menggunakan checker ketersediaan file biner lokal
    for key, path in MODEL_FILES.items():
        if not os.path.exists(path): raise FileNotFoundError(f"File model hilang: {path}")
    for key, path in SCALER_FILES.items():
        if not os.path.exists(path): raise FileNotFoundError(f"File scaler hilang: {path}")

    m_lstm   = tf.keras.models.load_model(MODEL_FILES[BEST_LSTM_KEY],   compile=False)
    m_bilstm = tf.keras.models.load_model(MODEL_FILES[BEST_BILSTM_KEY], compile=False)
    s_lstm   = joblib.load(SCALER_FILES[BEST_LSTM_KEY])
    s_bilstm = joblib.load(SCALER_FILES[BEST_BILSTM_KEY])
    return m_lstm, m_bilstm, s_lstm, s_bilstm

try:
    model_lstm, model_bilstm, scaler_lstm, scaler_bilstm = load_best_models()
    lstm_label = "LSTM Model"

    bilstm_label = "BiLSTM Model"
except Exception as e:
    st.error("⚠️ **Berkas Model (*.h5/*.pkl) tidak ditemukan di folder lokal.**")
    st.info("""
    Silakan letakkan file hasil training Anda dengan penamaan terstruktur berikut pada project folder:
    * `models/final_LSTM_BBCA_OHLC.h5`
    * `models/final_BiLSTM_BBCA_OHLC.h5`
    * `scalers/scaler_LSTM_BBCA_OHLC.pkl`
    * `scalers/scaler_BiLSTM_BBCA_OHLC.pkl`
    """)
    st.stop()

# ===============================
# HEADER
# ===============================
st.markdown(f"""
<div class="main-header">
    <h1>📈 Prediksi Harga Penutupan Saham</h1>
    <p>Perbandingan <b>{lstm_label}</b> ⭐ vs <b>{bilstm_label}</b> — hasil <i>hyperparameter tuning</i></p>
    <p style="font-size:0.82rem; opacity:0.75; margin-top:0.4rem;">
        Untuk investor & trader yang ingin estimasi arah harga saham
    </p>
</div>
""", unsafe_allow_html=True)

# ===============================
# INFO MODEL
# ===============================
st.markdown('<div class="section-header"><h2>🤖 Model yang Digunakan</h2></div>', unsafe_allow_html=True)
col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown(f"""
    <div class="card-title" style="color:#2563eb;">
</div>
        <div class="card-title">🔵 {lstm_label} <span class="card-badge"></span></div>
        <div class="card-meta"></i><br><b>→ Rekomendasi utama</b></div>
    </div>""", unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""
    <div class="card-title" style="color:#16a34a;">
</div>
        <div class="card-title">🟢 {bilstm_label}</div>
        <div class="card-meta"><br>Ditampilkan untuk perbandingan ilmiah.</div>
    </div>""", unsafe_allow_html=True)

tab_builtin, tab_upload, tab_manual = st.tabs([
    "📊 Saham Bawaan",
    "📁 Upload CSV / Excel",
    "✍️ Input Manual"
])

builtin_data = None
builtin_dates = None

with tab_builtin:

    st.subheader("📊 Saham Bawaan")

    saham_label = st.selectbox(
        "Pilih Saham",
        SAHAM_LIST
    )

    csv_path = BUILTIN_DATA[saham_label]

    df_builtin = pd.read_csv(csv_path)

    df_builtin["Date"] = pd.to_datetime(
        df_builtin["Date"],
        dayfirst=True,
        format="mixed"
    )

    df_builtin = df_builtin.sort_values("Date")

    builtin_data = (
        df_builtin[FEATURES]
        .astype(float)
        .values
    )
    builtin_dates = df_builtin["Date"].values

    st.info(
        f"""
    📅 Periode data : {df_builtin['Date'].min().strftime('%d-%m-%Y')}
    s.d.
    {df_builtin['Date'].max().strftime('%d-%m-%Y')}

    📊 Total data : {len(df_builtin)}
    """
    )

    train_data, val_data, test_data = split_data(builtin_data)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🟦 Train",
            f"{len(train_data)} data",
            "70%"
        )

    with col2:
        st.metric(
            "🟨 Validation",
            f"{len(val_data)} data",
            "15%"
        )

    with col3:
        st.metric(
            "🟥 Test",
            f"{len(test_data)} data",
            "15%"
        )

    st.markdown("---")

    n_days_builtin = st.number_input(
        "📅 Prediksi Hari ke Depan",
        min_value=1,
        max_value=30,
        value=5,
        key="n_builtin"
    )

    run_builtin = st.button(
        "🔮 Jalankan Prediksi",
        key="run_builtin"
    )

    result_builtin = st.container()

    if run_builtin:

        show_prediction(
            builtin_data,
            builtin_dates,
            saham_label,
            n_days_builtin,
            result_builtin
        )

upload_data = None
upload_dates = None

with tab_upload:

    st.subheader("📁 Upload CSV / Excel")

    uploaded = st.file_uploader(
        "Upload CSV atau Excel",
        type=["csv","xlsx","xls"]
    )

    if uploaded:

        if uploaded.name.endswith(".csv"):
            df_upload = pd.read_csv(uploaded)
        else:
            df_upload = pd.read_excel(uploaded)

        df_upload.columns = [
            c.strip().title()
            for c in df_upload.columns
        ]

        required = ["Open","High","Low","Close"]

        if all(col in df_upload.columns for col in required):

            if "Date" in df_upload.columns:

                df_upload["Date"] = pd.to_datetime(
                    df_upload["Date"],
                    errors="coerce"
                )

                df_upload = df_upload.sort_values("Date")

                upload_dates = df_upload["Date"].values

            else:

                upload_dates = np.arange(len(df_upload))

            upload_data = (
                df_upload[FEATURES]
                .astype(float)
                .values
            )

            if len(upload_data) <= WINDOW_SIZE + 5:
                st.error("Minimal data 10 baris.")
                upload_data = None

            st.success("✅ Data berhasil dimuat")

            train_data, val_data, test_data = split_data(upload_data)

            train_size = len(train_data)
            val_size = len(val_data)

            df_train = df_upload.iloc[:train_size].reset_index(drop=True)

            df_val = df_upload.iloc[
                train_size:train_size+val_size
            ].reset_index(drop=True)

            df_test = df_upload.iloc[
                train_size+val_size:
            ].reset_index(drop=True)

            st.info(
            f"""
            📊 Total data : {len(df_upload)}

            🟦 Train : {len(train_data)} data
            🟨 Validation : {len(val_data)} data
            🟥 Test : {len(test_data)} data
            """
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("🟦 Train", len(train_data), "70%")

            with col2:
                st.metric("🟨 Validation", len(val_data), "15%")

            with col3:
                st.metric("🟥 Test", len(test_data), "15%")

            tab_train, tab_val, tab_test = st.tabs(
                [
                    "🟦 Train",
                    "🟨 Validation",
                    "🟥 Test"
                ]
            )

            st.markdown("---")

            with tab_train:

                st.dataframe(
                    df_train,
                    use_container_width=True
                )

                process_train = st.button(
                    "⚙️ Proses Data Train",
                    key="process_train"
                )

                if process_train:

                    result_lstm = evaluate_dataset(
                        model_lstm,
                        scaler_lstm,
                        train_data
                    )

                    result_bilstm = evaluate_dataset(
                        model_bilstm,
                        scaler_bilstm,
                        train_data
                    )

                    df_train_eval = pd.DataFrame([
                        {
                            "Model": lstm_label,
                            "RMSE": result_lstm["RMSE"],
                            "MAE": result_lstm["MAE"],
                            "MAPE": result_lstm["MAPE"]
                        },
                        {
                            "Model": bilstm_label,
                            "RMSE": result_bilstm["RMSE"],
                            "MAE": result_bilstm["MAE"],
                            "MAPE": result_bilstm["MAPE"]
                        }
                    ])

                    st.subheader("📊 Evaluasi Data Train")

                    st.dataframe(
                        df_train_eval.style.format({
                            "RMSE":"Rp {:,.2f}",
                            "MAE":"Rp {:,.2f}",
                            "MAPE":"{:.2f}%"
                        }),
                        use_container_width=True
                    )

            with tab_val:

                st.dataframe(
                    df_val,
                    use_container_width=True
                )

                process_val = st.button(
                    "⚙️ Proses Data Validation",
                    key="process_val"
                )

                if process_val:

                    result_lstm = evaluate_dataset(
                        model_lstm,
                        scaler_lstm,
                        val_data
                    )

                    result_bilstm = evaluate_dataset(
                        model_bilstm,
                        scaler_bilstm,
                        val_data
                    )

                    df_val_eval = pd.DataFrame([
                        {
                            "Model": lstm_label,
                            "RMSE": result_lstm["RMSE"],
                            "MAE": result_lstm["MAE"],
                            "MAPE": result_lstm["MAPE"]
                        },
                        {
                            "Model": bilstm_label,
                            "RMSE": result_bilstm["RMSE"],
                            "MAE": result_bilstm["MAE"],
                            "MAPE": result_bilstm["MAPE"]
                        }
                    ])

                    st.subheader("📊 Evaluasi Data Validation")

                    st.dataframe(
                        df_val_eval.style.format({
                            "RMSE":"Rp {:,.2f}",
                            "MAE":"Rp {:,.2f}",
                            "MAPE":"{:.2f}%"
                        }),
                        use_container_width=True
                    )

            with tab_test:

                st.dataframe(
                    df_test,
                    use_container_width=True
                )

                process_test = st.button(
                    "⚙️ Proses Data Test",
                    key="process_test"
                )

                if process_test:

                    result_lstm = evaluate_dataset(
                        model_lstm,
                        scaler_lstm,
                        test_data
                    )

                    result_bilstm = evaluate_dataset(
                        model_bilstm,
                        scaler_bilstm,
                        test_data
                    )

                    df_test_eval = pd.DataFrame([
                        {
                            "Model": lstm_label,
                            "RMSE": result_lstm["RMSE"],
                            "MAE": result_lstm["MAE"],
                            "MAPE": result_lstm["MAPE"]
                        },
                        {
                            "Model": bilstm_label,
                            "RMSE": result_bilstm["RMSE"],
                            "MAE": result_bilstm["MAE"],
                            "MAPE": result_bilstm["MAPE"]
                        }
                    ])

                    st.subheader("📊 Evaluasi Data Test")

                    st.dataframe(
                        df_test_eval.style.format({
                            "RMSE":"Rp {:,.2f}",
                            "MAE":"Rp {:,.2f}",
                            "MAPE":"{:.2f}%"
                        }),
                        use_container_width=True
                    )

manual_data = None
manual_dates = None

with tab_manual:

    st.subheader("✍️ Input Manual")

    # ===============================
    # DATA DASAR INPUT MANUAL
    # ===============================
    if upload_data is not None:

        # gunakan 4 data terakhir hasil upload
        source_df = df_upload.copy()

    else:

        # gunakan saham bawaan yang sedang dipilih
        source_df = df_builtin.copy()


    # ambil 4 hari terakhir
    last4_df = (
        source_df[["Date"] + FEATURES]
        .tail(4)
        .reset_index(drop=True)
    )

    st.subheader("📊 Data 4 Hari Sebelumnya")

    st.dataframe(
        last4_df,
        use_container_width=True
    )

    st.subheader("✍️ Input Data Hari Berikutnya")

    col1, col2 = st.columns(2)

    with col1:

        open_input = st.number_input(
            "Open",
            min_value=0.0,
            value=0.0
        )

        high_input = st.number_input(
            "High",
            min_value=0.0,
            value=0.0
        )

    with col2:

        low_input = st.number_input(
            "Low",
            min_value=0.0,
            value=0.0
        )

        close_input = st.number_input(
            "Close",
            min_value=0.0,
            value=0.0
        )

    if (
        open_input > 0 and
        high_input > 0 and
        low_input > 0 and
        close_input > 0
    ):

        new_row = np.array([
            open_input,
            high_input,
            low_input,
            close_input
        ])

        manual_data = np.vstack([
            last4_df[FEATURES].values,
            new_row
        ])

        manual_dates = np.arange(
            len(manual_data)
        )

        st.success("✅ Data siap digunakan")

    st.markdown("---")

    st.subheader("🔮 Pengaturan Prediksi")

    n_days_manual = st.slider(
        "Berapa hari yang ingin diprediksi?",
        min_value=1,
        max_value=7,
        value=5,
        key="n_manual"
    )

    result_manual = st.container()

    run_manual = st.button(
        "🔮 Jalankan Prediksi",
        key="run_manual"
    )

    if run_manual:

        if manual_data is None:
            result_manual.warning(
                "Silakan isi nilai Open, High, Low, dan Close terlebih dahulu."
            )
        else:

            scaled_lstm = scaler_lstm.transform(manual_data)
            scaled_bilstm = scaler_bilstm.transform(manual_data)

            lstm_fc = recursive_forecast(
                model_lstm,
                scaler_lstm,
                scaled_lstm,
                n_days_manual
            )

            bilstm_fc = recursive_forecast(
                model_bilstm,
                scaler_bilstm,
                scaled_bilstm,
                n_days_manual
            )

            future_df = pd.DataFrame({
                "Hari ke-": range(1, n_days_manual+1),
                f"🔵 {lstm_label}": [f"Rp {x:,.0f}" for x in lstm_fc],
                f"🟢 {bilstm_label}": [f"Rp {x:,.0f}" for x in bilstm_fc]
            })

            result_manual.subheader(
                f"🔮 Prediksi {n_days_manual} Hari ke Depan"
            )

            result_manual.dataframe(
                future_df,
                use_container_width=True,
                hide_index=True
            )

            hari_labels = [
                f"Hari {i}"
                for i in range(1, n_days_manual+1)
            ]

            fig_line = go.Figure()

            fig_line.add_trace(
                go.Scatter(
                    x=hari_labels,
                    y=lstm_fc,
                    mode="lines+markers",
                    name=lstm_label,
                    line=dict(
                        color="#3b82f6",
                        width=3
                    )
                )
            )

            fig_line.add_trace(
                go.Scatter(
                    x=hari_labels,
                    y=bilstm_fc,
                    mode="lines+markers",
                    name=bilstm_label,
                    line=dict(
                        color="#22c55e",
                        width=3,
                        dash="dot"
                    )
                )
            )

            fig_line.update_layout(
                title="📈 Prediksi Harga Masa Depan",
                template="plotly_white",
                height=450,
                hovermode="x unified",
                xaxis_title="Hari Prediksi",
                yaxis_title="Harga Penutupan",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )

            result_manual.plotly_chart(
                fig_line,
                use_container_width=True
            )