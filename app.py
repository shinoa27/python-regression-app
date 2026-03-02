import streamlit as st
import pandas as pd
import plotly.express as px

# --- 1. 核心演算法 ---
def calculate_statistics(x_list, y_list):
    n = len(x_list)
    if n == 0: return 0.0, 0.0, 0.0
    mean_x = sum(x_list) / n
    mean_y = sum(y_list) / n
    numerator = sum_sq_x = sum_sq_y = 0.0
    for i in range(n):
        dx = x_list[i] - mean_x
        dy = y_list[i] - mean_y
        numerator += dx * dy
        sum_sq_x += dx ** 2
        sum_sq_y += dy ** 2
    denominator = (sum_sq_x * sum_sq_y) ** 0.5
    r_value = 0.0 if denominator == 0 else numerator / denominator
    m = 0.0 if sum_sq_x == 0 else numerator / sum_sq_x
    c = mean_y - m * mean_x
    return r_value, m, c

# --- 初始化系統記憶體 (State Management) ---
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False
if 'history' not in st.session_state:
    st.session_state.history = [] 
if 'default_data' not in st.session_state:
    st.session_state.default_data = pd.DataFrame({"X 軸數據": [1.0, 2.0, 3.0, 4.0, 5.0], "Y 軸數據": [2.0, 4.0, 5.0, 4.0, 5.0]})

# 【重要修正】預先為所有計算變數建立空位，徹底防範 KeyError
for key in ['m', 'c', 'r_value']:
    if key not in st.session_state:
        st.session_state[key] = 0.0
for key in ['x_data', 'y_data']:
    if key not in st.session_state:
        st.session_state[key] = []

# --- 2. 網頁介面設計 ---
st.title("多維數據統計與預測實驗室")
st.write("本系統完全以 Python 基礎迴圈實作皮爾森相關係數與最小平方法運算，並具備機器學習預測與歷史紀錄功能。")

st.subheader("✍️ 動態數據輸入區")
edited_df = st.data_editor(st.session_state.default_data, num_rows="dynamic", use_container_width=True)

# --- 3. 訓練模型按鈕 ---
if st.button("開始訓練模型與計算"):
    try:
        clean_df = edited_df.dropna()
        x_data = clean_df["X 軸數據"].tolist()
        y_data = clean_df["Y 軸數據"].tolist()

        if len(x_data) < 2:
            st.warning("⚠️ 數據量不足，請至少輸入兩筆以上的成對數據。")
        else:
            r_value, m, c = calculate_statistics(x_data, y_data)
            
            st.session_state['m'] = m
            st.session_state['c'] = c
            st.session_state['r_value'] = r_value
            st.session_state['x_data'] = x_data
            st.session_state['y_data'] = y_data
            st.session_state.model_ready = True
            st.session_state.history = [] 
            
    except Exception as e:
        st.error(f"系統發生錯誤：({e})")

# --- 4. 顯示結果、圖表與預測區塊 ---
if st.session_state.model_ready:
    st.success("模型訓練成功！")
    m = st.session_state['m']
    c = st.session_state['c']
    r_value = st.session_state['r_value']
    x_data = st.session_state['x_data']
    y_data = st.session_state['y_data']
    
    col_res1, col_res2 = st.columns(2)
    col_res1.metric(label="皮爾森相關係數 (r)", value=f"{r_value:.4f}")
    col_res2.metric(label="回歸直線方程式", value=f"y = {m:.2f}x + {c:.2f}")
    
    st.subheader("📈 數據分佈與最小平方法回歸線")
    fig = px.scatter(x=x_data, y=y_data, labels={'x': 'X 軸數據', 'y': 'Y 軸數據'})
    fig.update_traces(marker=dict(size=12, color='#1f77b4', line=dict(width=2, color='DarkSlateGrey')), name="真實數據點")
    x_min, x_max = min(x_data), max(x_data)
    fig.add_scatter(x=[x_min, x_max], y=[m*x_min+c, m*x_max+c], mode='lines', name='迴歸直線', line=dict(color='red', width=3, dash='dash'))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- 5. 機器學習預測引擎與歷史紀錄 ---
    st.subheader("🔮 機器學習預測引擎")
    st.write(f"目前套用模型：**y = {m:.2f}x + {c:.2f}**")
    
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        predict_x = st.number_input("請輸入未知的 X 數值：", value=0.0, step=1.0)
    
    predicted_y = m * predict_x + c
    st.info(f"📌 當 X = {predict_x} 時，模型預測的 Y 值為：**{predicted_y:.2f}**")
    
    with col_btn:
        st.write("") 
        st.write("")
        if st.button("💾 儲存預測結果"):
            st.session_state.history.append({"輸入的未知 X": predict_x, "模型預測的 Y": round(predicted_y, 2)})
    
    if st.session_state.history:
        st.write("📋 **預測歷史紀錄**")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

else:
    st.warning("請先在上方的資料表輸入數據，並點擊「開始訓練模型與計算」按鈕。")

# --- 6. 網頁瀏覽計數器 ---
st.divider()
st.subheader("👁️ 系統造訪統計")

# 改用更穩定的 visitorbadge 服務，並使用 Streamlit 原生的 st.image 渲染
# path 參數後面只需要放您專屬的獨特字串即可
st.image("https://api.visitorbadge.io/api/visitors?path=python-regression-app-bwrqbrfnjvarjdk9juncjl&label=VISITS&countColor=%2379C83D")
