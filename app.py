import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================
# 🌤 WEATHER MVT ANALYZER - Ứng dụng phân tích dữ liệu thời tiết
# ================================

st.set_page_config(page_title="Weather MVT Analyzer", layout="wide")

st.title("🌤 ỨNG DỤNG PHÂN TÍCH DỮ LIỆU THỜI TIẾT THEO ĐỊNH LÝ GIÁ TRỊ TRUNG BÌNH (MVT)")
st.markdown("---")

# ============================================================
# 1️⃣ NHẬP DỮ LIỆU
# ============================================================
st.sidebar.header("🗂 NHẬP DỮ LIỆU")

option = st.sidebar.radio("Chọn nguồn dữ liệu:", ["Dữ liệu mẫu", "Tải file CSV của bạn"])

if option == "Dữ liệu mẫu":
    df = pd.read_csv("data/sample_weather.csv")
else:
    uploaded = st.sidebar.file_uploader("Tải file CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.warning("⛔ Vui lòng tải file CSV hoặc chọn dữ liệu mẫu.")
        st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

st.subheader("📊 Dữ liệu đầu vào")
st.dataframe(df.head())

# ============================================================
# 2️⃣ BIỂU ĐỒ MINH HỌA
# ============================================================

st.subheader("📈 Biểu đồ minh họa dữ liệu thời tiết")
col = st.selectbox("Chọn yếu tố để phân tích:", ["temperature", "humidity", "precip"], index=0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["timestamp"], df[col], marker="o", label=col)
ax.set_xlabel("Thời gian")
ax.set_ylabel(col)
ax.legend()
st.pyplot(fig)

# ============================================================
# 3️⃣ TÍNH TOÁN CƠ BẢN
# ============================================================

st.subheader("🧮 Tính toán tốc độ thay đổi")

# Tốc độ thay đổi trung bình giữa các ngày (sai phân)
df["slope"] = df[col].diff()

# Đạo hàm xấp xỉ tại từng điểm (sai phân trung tâm)
df["derivative"] = (df[col].shift(-1) - df[col].shift(1)) / 2

st.write("**Tốc độ thay đổi trung bình (giữa 2 thời điểm liên tiếp):**")
st.dataframe(df[["timestamp", col, "slope", "derivative"]].head(10))

avg_slope = np.mean(df["slope"].dropna())
st.success(f"📈 Tốc độ thay đổi trung bình của {col} ≈ {avg_slope:.3f}")

# ============================================================
# 4️⃣ PHÂN TÍCH THEO ĐỊNH LÝ GIÁ TRỊ TRUNG BÌNH (MVT)
# ============================================================

st.subheader("📍 Phân tích theo Định lý Giá trị Trung bình (MVT)")

results = []
for i in range(1, len(df) - 1):
    x1, x2 = df["timestamp"].iloc[i - 1], df["timestamp"].iloc[i]
    f1, f2 = df[col].iloc[i - 1], df[col].iloc[i]
    avg_rate = (f2 - f1)
    # Tìm điểm MVT ước lượng bằng nội suy tuyến tính
    c = (f2 - f1) / 2 + f1
    results.append({"Khoảng": f"[{x1.strftime('%H:%M')}, {x2.strftime('%H:%M')}]",
                    "Δf": round(f2 - f1, 3),
                    "Tốc độ TB": round(avg_rate, 3),
                    "Điểm MVT (xấp xỉ)": round(c, 3)})

mvt_df = pd.DataFrame(results)
st.dataframe(mvt_df.head(10))

# ============================================================
# 5️⃣ GIẢI THÍCH CHI TIẾT THEO TỪNG BƯỚC
# ============================================================

st.subheader("🧠 Giải thích chi tiết")

st.markdown("""
### 🧩 Ý nghĩa các bước:
1. **Sai phân**: tính sự thay đổi của đại lượng giữa hai thời điểm gần nhau.
2. **Đạo hàm xấp xỉ**: cho biết tốc độ thay đổi tức thời tại từng thời điểm.
3. **MVT (Mean Value Theorem)**: tồn tại ít nhất một điểm trong khoảng mà tại đó đạo hàm bằng với tốc độ thay đổi trung bình.
4. **Nội suy tuyến tính** được dùng để ước lượng điểm đó trong dữ liệu rời rạc.

### 🧾 Cách đọc kết quả:
- Cột `Δf`: chênh lệch giá trị của yếu tố thời tiết giữa hai thời điểm.
- Cột `Tốc độ TB`: trung bình mức thay đổi trong khoảng đó.
- Cột `Điểm MVT`: giá trị tại vị trí mà tốc độ thay đổi tức thời ≈ tốc độ TB.

### 💡 Ghi chú:
- Nếu dữ liệu mượt và chi tiết, điểm MVT sẽ nằm rất gần giá trị thật.
- Nếu dữ liệu bị làm tròn hoặc lấy mẫu thưa, sai số MVT sẽ tăng rõ rệt.
""")

# ============================================================
# 6️⃣ PHÂN TÍCH ẢNH HƯỞNG CỦA VIỆC LÀM TRÒN DỮ LIỆU
# ============================================================

st.subheader("⚙️ Ảnh hưởng của việc làm tròn dữ liệu")

round_level = st.slider("Chọn mức làm tròn dữ liệu:", 0, 2, 1)
df_rounded = df.copy()
df_rounded[col] = df_rounded[col].round(round_level)

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df["timestamp"], df[col], label="Dữ liệu gốc", alpha=0.7)
ax2.plot(df_rounded["timestamp"], df_rounded[col], "--", label=f"Làm tròn {round_level} chữ số")
ax2.set_title("Ảnh hưởng của việc làm tròn dữ liệu đến biến thiên")
ax2.legend()
st.pyplot(fig2)

st.markdown(f"""
### 🧭 Nhận xét:
- Làm tròn ở mức {round_level} chữ số khiến dữ liệu bớt biến thiên nhỏ.
- Điều này có thể **làm sai lệch tốc độ thay đổi tức thời**, ảnh hưởng đến việc tìm điểm MVT.
- Trong dự báo thời tiết, việc làm tròn thô có thể khiến mô hình đánh giá **không chính xác các xu hướng ngắn hạn**.
""")

st.success("✅ Phân tích hoàn tất! Bạn có thể thay đổi yếu tố hoặc mức làm tròn để xem kết quả khác nhau.")
