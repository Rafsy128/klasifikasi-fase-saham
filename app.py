from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Klasifikasi Fase Saham', page_icon='📈', layout='wide')

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / 'outputs'
PHASE_ORDER = ['Akumulasi', 'Markup', 'Distribusi', 'Markdown']
PHASE_COLORS = {
    'Akumulasi': '#2C71D6',
    'Markup': '#2EAD62',
    'Distribusi': '#F2A93B',
    'Markdown': '#D84A4A',
}

@st.cache_data
def load_csv(path_string: str, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame | None:
    path = Path(path_string)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, parse_dates=list(parse_dates))
    except Exception as exc:
        st.error(f'Gagal membaca {path.name}: {exc}')
        return None

@st.cache_data
def load_json(path_string: str) -> dict | None:
    path = Path(path_string)
    if not path.exists():
        return None
    try:
        with path.open('r', encoding='utf-8') as handle:
            return json.load(handle)
    except Exception as exc:
        st.error(f'Gagal membaca {path.name}: {exc}')
        return None

def missing(filename: str, output_dir: Path) -> None:
    st.warning(f'File `{filename}` belum tersedia. Lokasi: `{output_dir / filename}`')

def pct(value) -> str:
    return '-' if value is None else f'{float(value) * 100:.2f}%'

def price_volume_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Harga'
    ))
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', yaxis='y2', opacity=0.25))
    fig.update_layout(
        title='Pergerakan Harga dan Volume', height=600, xaxis_rangeslider_visible=False,
        yaxis=dict(title='Harga'), yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
        legend=dict(orientation='h')
    )
    return fig

def phase_chart(df: pd.DataFrame, phase_col: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close', line=dict(color='#333333')))
    for phase in PHASE_ORDER:
        subset = df[df[phase_col] == phase]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset['Date'], y=subset['Close'], mode='markers', name=phase,
            marker=dict(size=7, color=PHASE_COLORS[phase])
        ))
    fig.update_layout(title=title, height=550, xaxis_title='Tanggal', yaxis_title='Harga Penutupan')
    return fig

def confusion_chart(df: pd.DataFrame, title: str) -> go.Figure:
    return px.imshow(df, text_auto=True, aspect='auto', title=title, color_continuous_scale='Blues')

def radar_chart(df: pd.DataFrame) -> go.Figure:
    metrics = ['Accuracy', 'Precision', 'Recall', 'Macro F1']
    fig = go.Figure()
    for _, row in df.iterrows():
        values = [float(row[m]) for m in metrics]
        fig.add_trace(go.Scatterpolar(r=values + values[:1], theta=metrics + metrics[:1], fill='toself', name=row['Model']))
    fig.update_layout(title='Radar Perbandingan Model', polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    return fig

# Validasi folder saham
if not OUTPUT_ROOT.exists():
    st.error(f'Folder outputs tidak ditemukan: {OUTPUT_ROOT}')
    st.stop()

available_stocks = sorted(folder.name for folder in OUTPUT_ROOT.iterdir() if folder.is_dir())
if not available_stocks:
    st.error('Tidak ditemukan folder saham. Gunakan struktur outputs/AAPL dan outputs/BBCA_JK.')
    st.stop()

st.sidebar.title('📊 Navigasi')
selected_stock = st.sidebar.selectbox('Pilih saham', available_stocks)
OUTPUT_DIR = OUTPUT_ROOT / selected_stock

if st.sidebar.button('🔄 Muat Ulang Data', use_container_width=True):
    st.cache_data.clear()
    st.rerun()

menu = st.sidebar.radio('Pilih halaman', [
    'Ringkasan', 'Dataset Awal', 'EDA', 'Feature Engineering', 'Rule-Based Labeling',
    'Train/Test Split', 'Random Forest', 'Support Vector Machine', 'Perbandingan Model', 'Hasil Prediksi'
])

# Load file per saham aktif
files = {
    'dataset_awal': ('dataset_awal.csv', ('Date',)),
    'dataset_feature': ('dataset_feature_engineered.csv', ('Date',)),
    'dataset_labeled': ('dataset_labeled.csv', ('Date',)),
    'dataset_training': ('dataset_training.csv', ('Date',)),
    'dataset_testing': ('dataset_testing.csv', ('Date',)),
    'label_summary': ('label_summary.csv', ()),
    'rf_predictions': ('rf_test_predictions.csv', ('Date',)),
    'svm_predictions': ('svm_test_predictions.csv', ('Date',)),
    'rf_report': ('rf_classification_report.csv', ()),
    'svm_report': ('svm_classification_report.csv', ()),
    'rf_confusion': ('rf_confusion_matrix.csv', ()),
    'svm_confusion': ('svm_confusion_matrix.csv', ()),
    'rf_importance': ('rf_feature_importance.csv', ()),
    'model_comparison': ('model_comparison.csv', ()),
}
data = {
    key: load_csv(str(OUTPUT_DIR / filename), parse_dates)
    for key, (filename, parse_dates) in files.items()
}
metadata = load_json(str(OUTPUT_DIR / 'metadata.json'))
metrics = load_json(str(OUTPUT_DIR / 'metrics.json'))

active_ticker = metadata.get('ticker', selected_stock) if metadata else selected_stock
st.sidebar.divider()
st.sidebar.write(f'Folder aktif: **{selected_stock}**')
st.sidebar.write(f'Ticker aktif: **{active_ticker}**')

st.title('📈 Klasifikasi Empat Fase Utama Saham')
st.caption(f'Ticker aktif: **{active_ticker}** — Random Forest dan Support Vector Machine')

if menu == 'Ringkasan':
    st.header('Ringkasan Penelitian')
    if metadata:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Dataset Awal', metadata.get('rows_raw', 0))
        c2.metric('Dataset Berlabel', metadata.get('rows_labeled', 0))
        c3.metric('Data Training', metadata.get('rows_training', 0))
        c4.metric('Data Testing', metadata.get('rows_testing', 0))
    comparison = data['model_comparison']
    if comparison is None:
        missing('model_comparison.csv', OUTPUT_DIR)
    else:
        st.dataframe(comparison, use_container_width=True)
        best = comparison.loc[comparison['Macro F1'].idxmax()]
        st.success(f"Model terbaik: **{best['Model']}** dengan Macro F1 **{best['Macro F1']:.4f}**")
        st.plotly_chart(radar_chart(comparison), use_container_width=True)

elif menu == 'Dataset Awal':
    df = data['dataset_awal']
    if df is None:
        missing('dataset_awal.csv', OUTPUT_DIR)
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric('Jumlah Data', len(df))
        c2.metric('Tanggal Awal', str(df['Date'].min().date()))
        c3.metric('Tanggal Akhir', str(df['Date'].max().date()))
        st.dataframe(df, use_container_width=True, height=500)

elif menu == 'EDA':
    df = data['dataset_awal']
    if df is None:
        missing('dataset_awal.csv', OUTPUT_DIR)
    else:
        tabs = st.tabs(['Harga & Volume', 'Statistik', 'Missing', 'Duplikat', 'Outlier', 'Korelasi'])
        with tabs[0]:
            st.plotly_chart(price_volume_chart(df), use_container_width=True)
        with tabs[1]:
            st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe().T, use_container_width=True)
        with tabs[2]:
            missing_df = pd.DataFrame({'Jumlah Missing': df.isna().sum(), 'Persentase (%)': df.isna().mean() * 100})
            st.dataframe(missing_df, use_container_width=True)
        with tabs[3]:
            count = int(df.duplicated().sum())
            st.metric('Jumlah Duplikat', count)
            if count:
                st.dataframe(df[df.duplicated(keep=False)], use_container_width=True)
        with tabs[4]:
            numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
            rows = []
            for col in numeric:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                count = int(((df[col] < lower) | (df[col] > upper)).sum())
                rows.append({'Variabel': col, 'Batas Bawah': lower, 'Batas Atas': upper, 'Jumlah Outlier': count})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.plotly_chart(px.box(df, y=numeric, title='Boxplot OHLCV'), use_container_width=True)
        with tabs[5]:
            corr = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
            st.plotly_chart(px.imshow(corr, text_auto='.2f', zmin=-1, zmax=1, color_continuous_scale='RdBu_r'), use_container_width=True)

elif menu == 'Feature Engineering':
    df = data['dataset_feature']
    if df is None:
        missing('dataset_feature_engineered.csv', OUTPUT_DIR)
    else:
        st.dataframe(df, use_container_width=True, height=500)
        features = [c for c in ['Return', 'RVOL', 'TrendDiff', 'PricePosition', 'Breakout', 'Breakdown'] if c in df.columns]
        if features:
            st.plotly_chart(px.imshow(df[features].corr(), text_auto='.2f', zmin=-1, zmax=1, color_continuous_scale='RdBu_r'), use_container_width=True)

elif menu == 'Rule-Based Labeling':
    df = data['dataset_labeled']
    if df is None:
        missing('dataset_labeled.csv', OUTPUT_DIR)
    else:
        st.dataframe(df, use_container_width=True, height=450)
        summary = data['label_summary']
        if summary is not None:
            st.plotly_chart(px.bar(summary, x='Label', y='Jumlah', text='Jumlah', category_orders={'Label': PHASE_ORDER}), use_container_width=True)
        if 'Label' in df.columns:
            st.plotly_chart(phase_chart(df, 'Label', 'Harga Berdasarkan Label Wyckoff'), use_container_width=True)

elif menu == 'Train/Test Split':
    train, test = data['dataset_training'], data['dataset_testing']
    if train is None or test is None:
        st.warning('Dataset training atau testing belum tersedia.')
    else:
        c1, c2 = st.columns(2)
        c1.metric('Data Training', len(train))
        c2.metric('Data Testing', len(test))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train['Date'], y=train['Close'], mode='lines', name='Training'))
        fig.add_trace(go.Scatter(x=test['Date'], y=test['Close'], mode='lines', name='Testing'))
        fig.add_vline(x=test['Date'].min(), line_dash='dash', annotation_text='Batas Train/Test')
        st.plotly_chart(fig, use_container_width=True)
        tab1, tab2 = st.tabs(['Training', 'Testing'])
        with tab1:
            st.dataframe(train, use_container_width=True)
        with tab2:
            st.dataframe(test, use_container_width=True)

elif menu in ['Random Forest', 'Support Vector Machine']:
    is_rf = menu == 'Random Forest'
    model_key = 'Random Forest' if is_rf else 'Support Vector Machine'
    report = data['rf_report'] if is_rf else data['svm_report']
    confusion = data['rf_confusion'] if is_rf else data['svm_confusion']
    predictions = data['rf_predictions'] if is_rf else data['svm_predictions']
    st.header(model_key)
    if metrics:
        model_metrics = metrics.get(model_key, {})
        values = model_metrics.get('metrics', {})
        cols = st.columns(4)
        cols[0].metric('Accuracy', pct(values.get('accuracy')))
        cols[1].metric('Precision Macro', pct(values.get('precision_macro')))
        cols[2].metric('Recall Macro', pct(values.get('recall_macro')))
        cols[3].metric('Macro F1', pct(values.get('f1_macro')))
        st.json(model_metrics.get('best_params', {}))
    if report is not None:
        st.dataframe(report, use_container_width=True)
    if confusion is not None:
        matrix = confusion.set_index(confusion.columns[0])
        st.plotly_chart(confusion_chart(matrix, f'Confusion Matrix {model_key}'), use_container_width=True)
    if is_rf and data['rf_importance'] is not None:
        imp = data['rf_importance'].sort_values('Importance', ascending=True)
        st.plotly_chart(px.bar(imp, x='Importance', y='Feature', orientation='h'), use_container_width=True)
    if predictions is not None:
        st.dataframe(predictions, use_container_width=True, height=450)
        pred_col = 'Prediction' if 'Prediction' in predictions.columns else 'PredictedLabel' if 'PredictedLabel' in predictions.columns else None
        if pred_col:
            st.plotly_chart(phase_chart(predictions, pred_col, f'Prediksi Fase {model_key}'), use_container_width=True)

elif menu == 'Perbandingan Model':
    comparison = data['model_comparison']
    if comparison is None:
        missing('model_comparison.csv', OUTPUT_DIR)
    else:
        st.dataframe(comparison, use_container_width=True)
        best = comparison.loc[comparison['Macro F1'].idxmax()]
        st.success(f"Model terbaik: **{best['Model']}** dengan Macro F1 **{best['Macro F1']:.4f}**")
        melted = comparison.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'Macro F1'], var_name='Metrik', value_name='Nilai')
        st.plotly_chart(px.bar(melted, x='Metrik', y='Nilai', color='Model', barmode='group', range_y=[0, 1]), use_container_width=True)
        st.plotly_chart(radar_chart(comparison), use_container_width=True)

elif menu == 'Hasil Prediksi':
    comparison = data['model_comparison']
    if comparison is None:
        missing('model_comparison.csv', OUTPUT_DIR)
    else:
        best_name = comparison.loc[comparison['Macro F1'].idxmax(), 'Model']
        predictions = data['rf_predictions'] if best_name == 'Random Forest' else data['svm_predictions']
        st.info(f'Model yang ditampilkan: **{best_name}**')
        if predictions is None:
            st.warning('Dataset hasil prediksi belum tersedia.')
        else:
            st.dataframe(predictions, use_container_width=True, height=500)
            pred_col = 'Prediction' if 'Prediction' in predictions.columns else 'PredictedLabel' if 'PredictedLabel' in predictions.columns else None
            status_col = 'PredictionStatus' if 'PredictionStatus' in predictions.columns else None
            if pred_col:
                st.plotly_chart(phase_chart(predictions, pred_col, f'Harga dan Fase Hasil Prediksi {best_name}'), use_container_width=True)
            if status_col:
                summary = predictions[status_col].value_counts().rename_axis('Status').reset_index(name='Jumlah')
                st.plotly_chart(px.bar(summary, x='Status', y='Jumlah', text='Jumlah'), use_container_width=True)
