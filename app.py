import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# ================================
# 🌤 WEATHER MVT ANALYZER - Phân tích dữ liệu thời tiết theo MVT
# ================================

st.set_page_config(page_title="Weather MVT Analyzer", layout="wide")

st.title("🌤 ỨNG DỤNG PHÂN TÍCH DỮ LIỆU THỜI TIẾT THEO ĐỊNH LÝ GIÁ TRỊ TRUNG BÌNH (MVT)")
st.markdown("---")

# ============================================================
# 1️⃣ HÀM TIỆN ÍCH
# ============================================================

def get_weather_data(city="Hanoi", api_key=None):
    """Lấy dữ liệu thời tiết 5 ngày (mỗi 3 giờ) từ OpenWeatherMap"""
    if not api_key:
        st.error("⛔ Thiếu API key. Hãy thêm vào secrets hoặc nhập trực tiếp.")
        return None

    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&units=metric&appid={api_key}&lang=vi"
    resp = requests.get(url)
    

    data = resp.json()
    records = []
    for item in data["list"]:
        records.append({
            "timestamp": item["dt_txt"],
            "temperature": item["main"]["temp"],
            "humidity": item["main"]["humidity"],
            "precip": item.get("rain", {}).get("3h", 0)
        })
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ============================================================
# 2️⃣ NHẬP DỮ LIỆU
# ============================================================

st.sidebar.header("🗂 NHẬP DỮ LIỆU")

option = st.sidebar.radio(
    "Chọn nguồn dữ liệu:",
    ["Dữ liệu mẫu", "Tải file CSV", "Lấy trực tiếp từ API"],
    index=0
)

if option == "Dữ liệu mẫu":
    df = pd.read_csv("data/sample_weather.csv")

elif option == "Tải file CSV":
    uploaded = st.sidebar.file_uploader("Tải file CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.warning("⛔ Vui lòng tải file CSV hoặc chọn dữ liệu khác.")
        st.stop()

else:
    city = st.sidebar.text_input("Nhập tên thành phố:", "Hanoi")
    api_key = st.sidebar.text_input("🔑 Nhập API key OpenWeatherMap:", type="password")
    if st.sidebar.button("📡 Lấy dữ liệu"):
        df = get_weather_data(city, api_key)
        if df is not None:
            st.success(f"✅ Đã tải dữ liệu thời tiết của **{city}** thành công!")
        else:
            st.stop()
    else:
        st.info("Nhập thành phố và API key rồi bấm **Lấy dữ liệu**.")
        st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"])
st.subheader("📊 Dữ liệu đầu vào")
st.dataframe(df.head())

# ============================================================
# 3️⃣ BIỂU ĐỒ MINH HỌA
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
# 4️⃣ TÍNH TOÁN CƠ BẢN
# ============================================================

st.subheader("🧮 Tính toán tốc độ thay đổi")

df["slope"] = df[col].diff()
df["derivative"] = (df[col].shift(-1) - df[col].shift(1)) / 2

avg_slope = np.mean(df["slope"].dropna())
st.success(f"📈 Tốc độ thay đổi trung bình của {col} ≈ {avg_slope:.3f}")

# ============================================================
# 5️⃣ PHÂN TÍCH THEO ĐỊNH LÝ GIÁ TRỊ TRUNG BÌNH (MVT)
# ============================================================

st.subheader("📍 Phân tích theo Định lý Giá trị Trung bình (MVT)")

results = []
for i in range(1, len(df) - 1):
    x1, x2 = df["timestamp"].iloc[i - 1], df["timestamp"].iloc[i]
    f1, f2 = df[col].iloc[i - 1], df[col].iloc[i]
    avg_rate = (f2 - f1)
    c = (f2 - f1) / 2 + f1
    results.append({
        "Khoảng": f"[{x1.strftime('%H:%M')}, {x2.strftime('%H:%M')}]",
        "Δf": round(f2 - f1, 3),
        "Tốc độ TB": round(avg_rate, 3),
        "Điểm MVT (xấp xỉ)": round(c, 3)
    })

mvt_df = pd.DataFrame(results)
st.dataframe(mvt_df.head(10))

# ============================================================
# 6️⃣ GIẢI THÍCH & ẢNH HƯỞNG CỦA LÀM TRÒN DỮ LIỆU
# ============================================================

st.subheader("🧠 Giải thích & Phân tích làm tròn dữ liệu")

round_level = st.slider("Chọn mức làm tròn dữ liệu:", 0, 2, 1)
df_rounded = df.copy()
df_rounded[col] = df_rounded[col].round(round_level)

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df["timestamp"], df[col], label="Dữ liệu gốc", alpha=0.7)
ax2.plot(df_rounded["timestamp"], df_rounded[col], "--", label=f"Làm tròn {round_level} chữ số")
ax2.legend()
ax2.set_title("Ảnh hưởng của việc làm tròn dữ liệu")
st.pyplot(fig2)

st.markdown(f"""
### 💡 Nhận xét:
- Làm tròn ở mức {round_level} chữ số khiến dữ liệu bớt biến thiên nhỏ.
- Điều này có thể làm sai lệch tốc độ thay đổi tức thời (đạo hàm xấp xỉ).
- Do đó, việc làm tròn quá mức có thể ảnh hưởng đến dự báo hoặc phân tích MVT.
""")

st.success("✅ Phân tích hoàn tất! Hãy thử thay đổi yếu tố, mức làm tròn hoặc nguồn dữ liệu.")
