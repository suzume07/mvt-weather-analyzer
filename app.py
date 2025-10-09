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
    if resp.status_code != 200:
        st.error(f"Lỗi truy xuất API: {resp.status_code} — {resp.text}")
        return None
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
# 6️⃣ GIẢI THÍCH & PHÂN TÍCH LÀM TRÒN DỮ LIỆU (MỚI)
# ============================================================
st.subheader("🧠 Giải thích & Phân tích làm tròn dữ liệu (mức 0,1,2,3 chữ số)")

# Hàm tính đạo hàm xấp xỉ chính xác theo đơn vị thời gian (value / day)
def compute_derivative_series(df_local, col_name):
    t0 = df_local['timestamp'].iloc[0]
    t_days = (df_local['timestamp'] - t0).dt.total_seconds() / 86400.0
    y = df_local[col_name].to_numpy(dtype=float)
    n = len(y)
    d = np.full(n, np.nan, dtype=float)
    if n < 2:
        return pd.Series(d, index=df_local.index)
    # interior: central differences
    for i in range(1, n-1):
        dt = t_days[i+1] - t_days[i-1]
        if dt == 0:
            d[i] = np.nan
        else:
            d[i] = (y[i+1] - y[i-1]) / dt
    # endpoints: forward/backward
    dt0 = t_days[1] - t_days[0]
    if dt0 != 0:
        d[0] = (y[1] - y[0]) / dt0
    dtN = t_days[-1] - t_days[-2] if n >= 2 else np.nan
    if not np.isnan(dtN) and dtN != 0:
        d[-1] = (y[-1] - y[-2]) / dtN
    return pd.Series(d, index=df_local.index)

# Hàm đếm số khoảng có MVT ước lượng (nơi đạo hàm hai đầu 'bracket' secant)
def count_mvt_intervals(df_local, col_name, deriv_series):
    t0 = df_local['timestamp'].iloc[0]
    t_days = (df_local['timestamp'] - t0).dt.total_seconds() / 86400.0
    y = df_local[col_name].to_numpy(dtype=float)
    n = len(y)
    count = 0
    for i in range(n - 1):
        if (t_days[i+1] - t_days[i]) == 0:
            continue
        S = (y[i+1] - y[i]) / (t_days[i+1] - t_days[i])
        d_i = deriv_series.iloc[i]
        d_ip1 = deriv_series.iloc[i+1]
        if np.isnan(S) or np.isnan(d_i) or np.isnan(d_ip1):
            continue
        if (d_i - S) * (d_ip1 - S) < 0 or (d_i - S) == 0 or (d_ip1 - S) == 0:
            count += 1
    return count

# Các mức làm tròn cần so sánh
rounding_levels = [0, 1, 2, 3]

# Tính đạo hàm gốc
deriv_orig = compute_derivative_series(df, col)

# Tạo bảng tóm tắt so sánh
summary_rows = []
for k in rounding_levels:
    df_r = df.copy()
    df_r[col] = df_r[col].round(k)
    deriv_r = compute_derivative_series(df_r, col)

    # Mask các giá trị hợp lệ
    mask = (~deriv_orig.isna()) & (~deriv_r.isna())
    if mask.sum() > 0:
        mae = float(np.nanmean(np.abs(deriv_orig[mask] - deriv_r[mask])))
        max_err = float(np.nanmax(np.abs(deriv_orig[mask] - deriv_r[mask])))
        sign_changes = int(np.sum((np.sign(deriv_orig[mask]) * np.sign(deriv_r[mask])) < 0))
        sign_change_pct = float(sign_changes / mask.sum() * 100.0)
    else:
        mae = np.nan
        max_err = np.nan
        sign_changes = 0
        sign_change_pct = np.nan

    # Số khoảng có MVT ước lượng
    mvt_count_orig = count_mvt_intervals(df, col, deriv_orig)
    mvt_count_round = count_mvt_intervals(df_r, col, deriv_r)

    summary_rows.append({
        "Làm tròn (chữ số)": k,
        "MAE đạo hàm (value/day)": mae,
        "Max err đạo hàm": max_err,
        "% đổi dấu đạo hàm": round(sign_change_pct, 2) if not np.isnan(sign_change_pct) else np.nan,
        "Số khoảng MVT (gốc)": mvt_count_orig,
        "Số khoảng MVT (làm tròn)": mvt_count_round
    })

summary_df = pd.DataFrame(summary_rows)
st.markdown("**Bảng so sánh tóm tắt ảnh hưởng của làm tròn**")
st.dataframe(summary_df)

# Hiển thị chi tiết từng mức làm tròn trong các expander (biểu đồ & số liệu)
for k in rounding_levels:
    with st.expander(f"Chi tiết: làm tròn {k} chữ số sau dấu phẩy"):
        df_r = df.copy()
        df_r[col] = df_r[col].round(k)
        deriv_r = compute_derivative_series(df_r, col)

        # Biểu đồ so sánh dữ liệu gốc và làm tròn
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(df['timestamp'], df[col], marker='o', label='Dữ liệu gốc', alpha=0.8)
        axes[0].plot(df_r['timestamp'], df_r[col], marker='x', linestyle='--', label=f'Làm tròn {k}', alpha=0.9)
        axes[0].set_title(f'Dữ liệu gốc vs làm tròn ({k} chữ số)')
        axes[0].legend()
        axes[0].set_ylabel(col)

        # Biểu đồ đạo hàm so sánh
        axes[1].plot(df['timestamp'], deriv_orig, marker='o', label='Đạo hàm gốc')
        axes[1].plot(df_r['timestamp'], deriv_r, marker='x', linestyle='--', label=f'Đạo hàm làm tròn {k}')
        axes[1].set_title('Đạo hàm xấp xỉ (value / day)')
        axes[1].legend()
        axes[1].set_ylabel('đơn vị / day')
        plt.tight_layout()
        st.pyplot(fig)

        # Số liệu chi tiết
        mask = (~deriv_orig.isna()) & (~deriv_r.isna())
        if mask.sum() > 0:
            mae = np.nanmean(np.abs(deriv_orig[mask] - deriv_r[mask]))
            max_err = np.nanmax(np.abs(deriv_orig[mask] - deriv_r[mask]))
            sign_changes = int(np.sum((np.sign(deriv_orig[mask]) * np.sign(deriv_r[mask])) < 0))
            st.markdown(f"- **MAE đạo hàm:** {mae:.6g}")
            st.markdown(f"- **Max error đạo hàm:** {max_err:.6g}")
            st.markdown(f"- **Số điểm đổi dấu đạo hàm (trong vùng so sánh):** {sign_changes} / {mask.sum()}")
        else:
            st.markdown("Không có đủ giá trị đạo hàm hợp lệ để so sánh.")

# ---------------------------------------------------------
# Minh hoạ công thức xấp xỉ tuyến tính f(b) ≈ f(a) + f'(a)*(x-a)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔎 Minh hoạ công thức xấp xỉ tuyến tính")

st.markdown("Công thức xấp xỉ tuyến tính (tuyến tính hoá tại a):\n\n"
            r"$$f(b) \approx f(a) + f'(a)\,(x - a)$$\n\n"
            "Trong dữ liệu rời rạc, ta ước lượng f'(a) bằng sai phân trung tâm/tiến/lùi. "
            "Minh họa sau cho thấy vì sao công thức này hợp lý khi khoảng ngắn và dữ liệu mượt.")

# Chọn khoảng để minh hoạ
max_i = max(0, len(df) - 2)
idx = st.number_input("Chọn chỉ số i để minh họa (xét khoảng i → i+1)", min_value=0, max_value=max_i, value=0, step=1)
a_idx = int(idx)
b_idx = a_idx + 1

# Tính đạo hàm tại a (tính từ đạo hàm gốc tính bên trên)
f_a = float(df[col].iloc[a_idx])
f_b = float(df[col].iloc[b_idx])
t0 = df['timestamp'].iloc[0]
t_days = (df['timestamp'] - t0).dt.total_seconds() / 86400.0
dt_ab = t_days.iloc[b_idx] - t_days.iloc[a_idx]
fprime_a = deriv_orig.iloc[a_idx]

if np.isnan(fprime_a):
    st.warning("Không có giá trị đạo hàm tại điểm a để minh họa (có thể do dữ liệu quá ít hoặc trùng thời điểm). Chọn i khác.")
else:
    f_approx = f_a + fprime_a * dt_ab
    err = f_b - f_approx
    pct_err = (err / f_b * 100.0) if f_b != 0 else np.nan

    # Vẽ minh họa
    x_lin = np.linspace(t_days.iloc[a_idx] - dt_ab*0.2, t_days.iloc[b_idx] + dt_ab*0.2, 100)
    y_lin = f_a + fprime_a * (x_lin - t_days.iloc[a_idx])
    x_lin_ts = t0 + pd.to_timedelta(x_lin, unit='D')

    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.plot(df['timestamp'], df[col], marker='o', label='Dữ liệu gốc')
    ax3.scatter([df['timestamp'].iloc[a_idx]], [f_a], color='green', s=80, label='a (điểm tuyến tính hoá)')
    ax3.scatter([df['timestamp'].iloc[b_idx]], [f_b], color='red', s=80, label='b (giá trị thật)')
    ax3.plot(x_lin_ts, y_lin, linestyle='--', color='orange', label="Đường xấp xỉ tại a: f(a)+f'(a)(x-a)")
    ax3.set_title(f"Minh họa xấp xỉ tuyến tính cho khoảng {a_idx} → {b_idx}")
    ax3.set_ylabel(col)
    ax3.legend()
    st.pyplot(fig3)

    # Hiện kết quả số
    st.markdown(f"- Giá trị thật tại b: **f(b) = {f_b:.6g}**")
    st.markdown(f"- Giá trị xấp xỉ bằng phương pháp tuyến tính hoá tại a: **f_approx = {f_approx:.6g}**")
    st.markdown(f"- Sai số tuyệt đối: **{err:.6g}**, Sai số tương đối: **{pct_err:.3f}%**")

# ---------------------------------------------------------
# Tổng kết & diễn giải bằng văn bản
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### ✅ Kết luận (tóm tắt) về ảnh hưởng của làm tròn dữ liệu:")
st.markdown("""
- **Làm tròn làm mất chi tiết nhỏ**: các dao động cỡ nhỏ có thể bị triệt tiêu, khiến đạo hàm (dựa trên sai phân) giảm hoặc biến đổi.
- **Làm tròn có thể đổi dấu đạo hàm** ở nhiều điểm, điều này có thể khiến ta **không tìm được điểm MVT** trong một số khoảng (vì đạo hàm không còn chéo qua giá trị secant).
- **MAE / Max error của đạo hàm tăng theo mức làm tròn**: tức là thông tin về tốc độ thay đổi tức thời bị méo đi.
- **Ảnh hưởng tới dự báo**: nếu mô hình dự báo/ước lượng ngắn hạn phụ thuộc vào đạo hàm tức thời (hoặc gradient), việc làm tròn quá mức sẽ làm cho dự báo **mất nhạy** với biến đổi ngắn hạn hoặc thậm chí đưa đến dự báo sai hướng.
- **Lời khuyên thực tế**: tránh làm tròn trước khi tính đạo hàm; nếu cần, hãy làm mịn (smoothing) thay vì làm tròn thô, hoặc thực hiện phân tích nhạy (sensitivity) với các mức làm tròn như trên.
""")
