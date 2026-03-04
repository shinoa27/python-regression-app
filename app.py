import streamlit as st
import pandas as pd
import plotly.express as px

# 核心演算法
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

# 初始化系統記憶體 (State Management)
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False
if 'history' not in st.session_state:
    st.session_state.history = [] 
if 'models' not in st.session_state:
    st.session_state.models = {} 
if 'clean_data' not in st.session_state:
    st.session_state.clean_data = pd.DataFrame()
if 'user_notes' not in st.session_state:
    st.session_state.user_notes = "" 

# 支援多組獨立資料表的記憶體結構
if 'group_data' not in st.session_state:
    st.session_state.group_data = {
        "A組": pd.DataFrame({"X 軸數據": [1.0, 2.0, 3.0], "Y 軸數據": [2.0, 4.0, 5.0]}),
        "B組": pd.DataFrame({"X 軸數據": [1.5, 2.5, 3.5], "Y 軸數據": [1.0, 3.0, 4.5]})
    }

for key in ['m', 'c', 'r_value']:
    if key not in st.session_state:
        st.session_state[key] = 0.0
for key in ['x_data', 'y_data']:
    if key not in st.session_state:
        st.session_state[key] = []

st.title("多維數據對比與預測實驗室")

# 【新增】頂部開發者微型宣告
st.caption("開發者備註：本系統目前仍處於 Beta 測試階段，作為學術展示用途，演算法與介面將持續迭代優化。")

st.write("本系統支援多組數據建檔、客製化圖表渲染，並可針對獨立數據組進行迴歸預測。")

# 區塊 1：數據分組輸入與管理
st.subheader("1. 數據分組輸入區")

col_new_name, col_new_btn = st.columns([3, 1])
with col_new_name:
    new_group_name = st.text_input("新增組別", label_visibility="collapsed", placeholder="請輸入新組別名稱 (例如：實驗C組)")
with col_new_btn:
    if st.button("➕ 新增組別", use_container_width=True):
        if not new_group_name.strip():
            st.warning("組別名稱不能為空白。")
        elif new_group_name in st.session_state.group_data:
            st.warning("此組別名稱已存在，請更換名稱。")
        else:
            st.session_state.group_data[new_group_name] = pd.DataFrame({"X 軸數據": [0.0], "Y 軸數據": [0.0]})
            st.rerun()

for group_name in list(st.session_state.group_data.keys()):
    with st.expander(f"📁 {group_name} 數據管理", expanded=True):
        col_rename, col_del = st.columns([4, 1])
        with col_rename:
            new_name_input = st.text_input("📝 修改組別名稱：", value=group_name, key=f"rename_{group_name}")
            if new_name_input != group_name and new_name_input.strip():
                if st.button("💾 確認修改名稱", key=f"btn_rename_{group_name}"):
                    if new_name_input in st.session_state.group_data:
                        st.error("名稱已存在，請選擇其他名稱！")
                    else:
                        st.session_state.group_data[new_name_input] = st.session_state.group_data.pop(group_name)
                        st.session_state.model_ready = False 
                        st.rerun()
        with col_del:
            st.write("") 
            st.write("")
            if st.button("🗑️ 刪除此組別", key=f"del_{group_name}"):
                del st.session_state.group_data[group_name]
                st.session_state.model_ready = False 
                st.rerun()
                
        st.session_state.group_data[group_name] = st.data_editor(
            st.session_state.group_data[group_name], 
            num_rows="dynamic", 
            use_container_width=True,
            key=f"editor_{group_name}"
        )

# 訓練模型按鈕與資料預處理
if st.button("開始訓練分組模型", type="primary"):
    try:
        if not st.session_state.group_data:
            st.error("目前無任何數據組，請先新增組別。")
        else:
            combined_data_list = []
            st.session_state.models = {}
            
            for group_name, df in st.session_state.group_data.items():
                clean_df = df.dropna().copy()
                x_data = clean_df["X 軸數據"].tolist()
                y_data = clean_df["Y 軸數據"].tolist()
                
                if len(x_data) >= 2:
                    r_value, m, c = calculate_statistics(x_data, y_data)
                    st.session_state.models[group_name] = {
                        'm': m, 'c': c, 'r_value': r_value,
                        'x_min': min(x_data), 'x_max': max(x_data)
                    }
                    clean_df["數據組名稱"] = group_name
                    combined_data_list.append(clean_df)
            
            if combined_data_list:
                st.session_state.clean_data = pd.concat(combined_data_list, ignore_index=True)
                st.session_state.model_ready = True
                st.session_state.history = [] 
                st.success(f"成功訓練 {len(st.session_state.models)} 組模型！")
            else:
                st.error("沒有任何一組數據包含兩筆以上的有效數值，無法建立模型。")
                
    except Exception as e:
        st.error(f"系統發生錯誤：({e})")

st.divider()

# 區塊 2：動態圖表渲染區
if st.session_state.model_ready:
    st.subheader("2. 視覺化分析圖表")
    
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])
    with col_ctrl1:
        custom_title = st.text_input("自訂圖表標題", value="多組數據分析比較圖")
    with col_ctrl2:
        available_groups = list(st.session_state.models.keys())
        selected_groups = st.multiselect("選擇欲顯示的數據組", options=available_groups, default=available_groups)
    with col_ctrl3:
        chart_type = st.radio("圖表類型", options=["散佈圖+迴歸線", "柱狀圖"])

    if not selected_groups:
        st.warning("請至少選擇一組數據以顯示圖表。")
    else:
        plot_df = st.session_state.clean_data[st.session_state.clean_data["數據組名稱"].isin(selected_groups)]
        color_sequence = px.colors.qualitative.Plotly
        color_map = {group: color_sequence[i % len(color_sequence)] for i, group in enumerate(selected_groups)}

        if chart_type == "散佈圖+迴歸線":
            fig = px.scatter(
                plot_df, x="X 軸數據", y="Y 軸數據", color="數據組名稱", 
                title=custom_title, color_discrete_map=color_map
            )
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
            
            for group in selected_groups:
                model = st.session_state.models[group]
                m, c = model['m'], model['c']
                x_min, x_max = model['x_min'], model['x_max']
                
                fig.add_scatter(
                    x=[x_min, x_max], y=[m * x_min + c, m * x_max + c],
                    mode='lines', name=f'{group} 迴歸線',
                    line=dict(color=color_map[group], width=3, dash='dash')
                )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("各組數據統計參數：")
            stat_data = []
            for group in selected_groups:
                m = st.session_state.models[group]
                stat_data.append({"數據組": group, "相關係數 (r)": f"{m['r_value']:.4f}", "方程式": f"y = {m['m']:.2f}x + {m['c']:.2f}"})
            st.table(pd.DataFrame(stat_data))

        elif chart_type == "柱狀圖":
            fig = px.bar(
                plot_df, x="X 軸數據", y="Y 軸數據", color="數據組名稱",
                barmode="group", title=custom_title, color_discrete_map=color_map
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 區塊 3：多模型機器學習預測引擎
    st.subheader("3. 獨立模型預測引擎")
    
    col_pred1, col_pred2 = st.columns([1, 2])
    with col_pred1:
        target_model_name = st.selectbox("請選擇預測基準模型：", options=available_groups)
    
    if target_model_name:
        target_model = st.session_state.models[target_model_name]
        m, c = target_model['m'], target_model['c']
        
        st.write(f"已載入 **{target_model_name}** 模型：`y = {m:.2f}x + {c:.2f}`")
        
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            predict_x = st.number_input("請輸入未知的 X 數值：", value=0.0, step=1.0)
        
        predicted_y = m * predict_x + c
        st.info(f"📌 當 X = {predict_x} 時，**{target_model_name}** 模型預測的 Y 值為：**{predicted_y:.2f}**")
        
        with col_btn:
            st.write("") 
            st.write("")
            if st.button("儲存預測結果"):
                st.session_state.history.append({
                    "使用模型": target_model_name,
                    "輸入的未知 X": predict_x, 
                    "模型預測的 Y": round(predicted_y, 2)
                })
        
        if st.session_state.history:
            st.write("**預測歷史紀錄**")
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

else:
    st.warning("請先在上方的資料表輸入數據，並點擊「開始訓練分組模型」按鈕。")

st.divider()

# 區塊 4：個人備註與系統計數
st.markdown("###### 📝 個人實驗觀察與備註")
st.session_state.user_notes = st.text_area(
    "備註區域", 
    value=st.session_state.user_notes, 
    label_visibility="collapsed", 
    placeholder="您可以在此輸入觀察結論、實驗記錄或其他備註，系統會自動保留文字..."
)

st.write("") 
col_badge1, col_badge2 = st.columns([1, 4])
with col_badge1:
    st.image("https://api.visitorbadge.io/api/visitors?path=python-regression-app-bwrqbrfnjvarjdk9juncjl&label=VISITS&countColor=%2379C83D")

# 【新增】網頁最底部的開發者註腳 (Footer)
st.markdown(
    """
    <hr style='margin-top: 2em; margin-bottom: 1em;'>
    <div style='text-align: center; color: #888888; font-size: 13px;'>
        <i>系統版本 v1.0 (Beta) — 專為學術展示與數據預測設計。</i><br>
        <i>開發者保有隨時更新系統功能與演算法之權利。</i>
    </div>
    """, 
    unsafe_allow_html=True
)
