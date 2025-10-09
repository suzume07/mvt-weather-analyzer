import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# ================================
# 🌤 ỨNG DỤNG PHÂN TÍCH MVT THỜI TIẾT
# ================================

st.set_page_config(page_title="Phân tích MVT thời tiết", layout="wide")
st.title("🌤 ỨNG DỤNG PHÂN TÍCH DỮ LIỆU THỜI TIẾT THEO ĐỊNH LÝ GIÁ TRỊ TRUNG BÌNH (MVT)")
st.markdown("---")

# ============================================================
# 1️⃣ HÀM TIỆN ÍCH
# ============================================================

def lay_du_lieu_thoi_tiet(thanh_pho="Hanoi", api_key=None):
    """Lấy dữ liệu dự báo 5 ngày (mỗi 3 giờ) từ OpenWeatherMap"""
    if not api_key:
        st.error("⛔ Thiếu API key. Hãy thêm vào hoặc nhập trực tiếp.")
        return None

    url = f"https://api.openweathermap.org/data/2.5/forecast?q={thanh_pho}&units=metric&appid={api_key}&lang=vi"
    resp = requests.get(url)
    if resp.status_code != 200:
        st.error(f"Lỗi truy xuất API: {resp.status_code} — {resp.text}")
        return None

    data = resp.json()
    danh_sach = []
    for muc in data["list"]:
        danh_sach.append({
            "thoi_gian": muc["dt_txt"],
            "nhiet_do": muc["main"]["temp"],
            "do_am": muc["main"]["humidity"],
            "luong_mua": muc.get("rain", {}).get("3h", 0)
        })
    df = pd.DataFrame(danh_sach)
    df["thoi_gian"] = pd.to_datetime(df["thoi_gian"])
    df = df.sort_values("thoi_gian").reset_index(drop=True)
    df["thoi_gian_ngay"] = (df["thoi_gian"] - df["thoi_gian"].iloc[0]).dt.total_seconds() / 86400.0
    return df


# ============================================================
# 2️⃣ NHẬP DỮ LIỆU
# ============================================================

st.sidebar.header("🗂 NHẬP DỮ LIỆU")
lua_chon = st.sidebar.radio(
    "Chọn nguồn dữ liệu:",
    ["Dữ liệu mẫu", "Tải file CSV", "Lấy trực tiếp từ API"],
    index=0
)

if lua_chon == "Dữ liệu mẫu":
    df = pd.read_csv("data/sample_weather.csv")
    df["thoi_gian"] = pd.to_datetime(df["thoi_gian"])
    df["thoi_gian_ngay"] = (df["thoi_gian"] - df["thoi_gian"].iloc[0]).dt.total_seconds() / 86400.0

elif lua_chon == "Tải file CSV":
    tep_tai = st.sidebar.file_uploader("Tải file CSV", type=["csv"])
    if tep_tai is not None:
        df = pd.read_csv(tep_tai)
        df["thoi_gian"] = pd.to_datetime(df["thoi_gian"])
        df["thoi_gian_ngay"] = (df["thoi_gian"] - df["thoi_gian"].iloc[0]).dt.total_seconds() / 86400.0
    else:
        st.warning("⛔ Vui lòng tải file CSV hoặc chọn dữ liệu khác.")
        st.stop()

else:
    thanh_pho = st.sidebar.text_input("Nhập tên thành phố:", "Hanoi")
    api_key = st.sidebar.text_input("🔑 Nhập API key OpenWeatherMap:", type="password")
    if st.sidebar.button("📡 Lấy dữ liệu"):
        df = lay_du_lieu_thoi_tiet(thanh_pho, api_key)
        if df is not None:
            st.success(f"✅ Đã tải dữ liệu thời tiết của **{thanh_pho}** thành công!")
        else:
            st.stop()
    else:
        st.info("Nhập thành phố và API key rồi bấm **Lấy dữ liệu**.")
        st.stop()

st.subheader("📊 Dữ liệu đầu vào")
st.dataframe(df.head())

# ============================================================
# 3️⃣ BIỂU ĐỒ MINH HỌA
# ============================================================

st.subheader("📈 Biểu đồ minh họa dữ liệu thời tiết")
cot = st.selectbox("Chọn yếu tố để phân tích:", ["nhiet_do", "do_am", "luong_mua"], index=0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["thoi_gian"], df[cot], marker="o", label=cot)
ax.set_xlabel("Thời gian")
ax.set_ylabel(cot)
ax.legend()
st.pyplot(fig)

# ============================================================
# 4️⃣ TÍNH TOÁN CƠ BẢN
# ============================================================

st.subheader("🧮 Tính toán tốc độ thay đổi")

df["dao_ham_sai_phan"] = df[cot].diff()
df["dao_ham_xap_xi"] = (df[cot].shift(-1) - df[cot].shift(1)) / 2
toc_do_tb = np.mean(df["dao_ham_sai_phan"].dropna())
st.success(f"📈 Tốc độ thay đổi trung bình của {cot} ≈ {toc_do_tb:.3f}")

# ============================================================
# 5️⃣ HÀM PHỤ TRỢ CHO MVT
# ============================================================

def tinh_dao_ham(df, cot):
    t_ngay = df["thoi_gian_ngay"]
    y = df[cot]
    n = len(df)
    dao_ham = np.zeros(n)
    for i in range(n):
        if 0 < i < n - 1:
            dt = t_ngay.iloc[i + 1] - t_ngay.iloc[i - 1]
            dy = y.iloc[i + 1] - y.iloc[i - 1]
        elif i == 0:
            dt = t_ngay.iloc[1] - t_ngay.iloc[0]
            dy = y.iloc[1] - y.iloc[0]
        else:
            dt = t_ngay.iloc[-1] - t_ngay.iloc[-2]
            dy = y.iloc[-1] - y.iloc[-2]
        dao_ham[i] = dy / dt if dt != 0 else np.nan
    return pd.Series(dao_ham, name=f"d{cot}/dt")

def dem_khoang_mvt(df_local, cot, dao_ham):
    t0 = df_local['thoi_gian'].iloc[0]
    t_ngay = (df_local['thoi_gian'] - t0).dt.total_seconds() / 86400.0
    y = df_local[cot].to_numpy(dtype=float)
    n = len(y)
    dem = 0
    for i in range(n - 1):
        if (t_ngay[i + 1] - t_ngay[i]) == 0:
            continue
        S = (y[i + 1] - y[i]) / (t_ngay[i + 1] - t_ngay[i])
        d_i = dao_ham.iloc[i]
        d_ip1 = dao_ham.iloc[i + 1]
        if np.isnan(S) or np.isnan(d_i) or np.isnan(d_ip1):
            continue
        if (d_i - S) * (d_ip1 - S) < 0 or (d_i - S) == 0 or (d_ip1 - S) == 0:
            dem += 1
    return dem

# ============================================================
# 6️⃣ PHÂN TÍCH LÀM TRÒN DỮ LIỆU
# ============================================================

st.subheader("🧠 Ảnh hưởng của việc làm tròn dữ liệu đến đạo hàm & MVT")

muc_lam_tron = [0, 1, 2, 3]
dao_ham_goc = tinh_dao_ham(df, cot)
bang_tong_hop = []

for k in muc_lam_tron:
    df_r = df.copy()
    df_r[cot] = df_r[cot].round(k)
    dao_ham_r = tinh_dao_ham(df_r, cot)

    mat_na = (~dao_ham_goc.isna()) & (~dao_ham_r.isna())
    if mat_na.sum() > 0:
        sai_tb = float(np.nanmean(np.abs(dao_ham_goc[mat_na] - dao_ham_r[mat_na])))
        sai_max = float(np.nanmax(np.abs(dao_ham_goc[mat_na] - dao_ham_r[mat_na])))
        doi_dau = int(np.sum((np.sign(dao_ham_goc[mat_na]) * np.sign(dao_ham_r[mat_na])) < 0))
        doi_dau_pct = float(doi_dau / mat_na.sum() * 100.0)
    else:
        sai_tb, sai_max, doi_dau, doi_dau_pct = np.nan, np.nan, 0, np.nan

    so_khoang_goc = dem_khoang_mvt(df, cot, dao_ham_goc)
    so_khoang_lam_tron = dem_khoang_mvt(df_r, cot, dao_ham_r)

    bang_tong_hop.append({
        "Làm tròn (chữ số)": k,
        "Sai TB đạo hàm": sai_tb,
        "Sai max đạo hàm": sai_max,
        "% đổi dấu đạo hàm": round(doi_dau_pct, 2),
        "Khoảng MVT (gốc)": so_khoang_goc,
        "Khoảng MVT (làm tròn)": so_khoang_lam_tron
    })

summary_df = pd.DataFrame(bang_tong_hop)
st.markdown("**Bảng so sánh ảnh hưởng của làm tròn:**")
st.dataframe(summary_df)

# 🎚️ Slider trực quan
st.markdown("---")
st.subheader("🎚️ Minh họa trực quan mức làm tròn")

muc = st.slider("Chọn mức làm tròn:", 0, 3, 1, step=1)
df_lam_tron = df.copy()
df_lam_tron[cot] = df_lam_tron[cot].round(muc)

hien_thi_dao_ham = st.checkbox("Hiển thị đạo hàm trên biểu đồ", value=False)

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df["thoi_gian"], df[cot], label="Dữ liệu gốc", alpha=0.7)
ax2.plot(df_lam_tron["thoi_gian"], df_lam_tron[cot], "--", label=f"Làm tròn {muc} chữ số", color="orange")

if hien_thi_dao_ham:
    dao_ham_goc_plot = tinh_dao_ham(df, cot)
    dao_ham_lam_tron_plot = tinh_dao_ham(df_lam_tron, cot)
    ax2.plot(df["thoi_gian"], dao_ham_goc_plot, "g-", alpha=0.5, label="Đạo hàm (gốc)")
    ax2.plot(df["thoi_gian"], dao_ham_lam_tron_plot, "r--", alpha=0.5, label="Đạo hàm (làm tròn)")

ax2.set_title("Ảnh hưởng của việc làm tròn dữ liệu đến biến thiên & đạo hàm")
ax2.set_xlabel("Thời gian")
ax2.set_ylabel(cot)
ax2.legend()
st.pyplot(fig2)

st.caption("""
Kéo thanh trượt để quan sát cách mức làm tròn ảnh hưởng đến biến thiên và tốc độ thay đổi.
Làm tròn quá lớn khiến dữ liệu mất chi tiết, đạo hàm dao động mạnh, dẫn đến sai lệch khi dự báo.
""")

# ============================================================
# 🔎 MINH HỌA CÔNG THỨC XẤP XỈ
# ============================================================

st.markdown("---")
st.subheader("🔎 Minh họa công thức xấp xỉ tuyến tính f(b) ≈ f(a) + f'(a)(x-a)")

max_i = max(0, len(df) - 2)
idx = st.number_input("Chọn chỉ số i để minh họa (i → i+1)", min_value=0, max_value=max_i, value=0, step=1)
a_idx, b_idx = int(idx), int(idx) + 1

f_a = float(df[cot].iloc[a_idx])
f_b = float(df[cot].iloc[b_idx])
fprime_a = dao_ham_goc.iloc[a_idx]
t0 = df["thoi_gian"].iloc[0]
t_ngay = (df["thoi_gian"] - t0).dt.total_seconds() / 86400.0
dt_ab = t_ngay.iloc[b_idx] - t_ngay.iloc[a_idx]

if np.isnan(fprime_a):
    st.warning("Không có giá trị đạo hàm tại điểm a.")
else:
    f_xapxi = f_a + fprime_a * dt_ab
    sai_so = f_b - f_xapxi
    pct_sai = (sai_so / f_b * 100.0) if f_b != 0 else np.nan

    x_lin = np.linspace(t_ngay.iloc[a_idx] - dt_ab * 0.2, t_ngay.iloc[b_idx] + dt_ab * 0.2, 100)
    y_lin = f_a + fprime_a * (x_lin - t_ngay.iloc[a_idx])
    x_lin_ts = t0 + pd.to_timedelta(x_lin, unit="D")

    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.plot(df["thoi_gian"], df[cot], marker="o", label="Dữ liệu gốc")
    ax3.scatter([df["thoi_gian"].iloc[a_idx]], [f_a], color="green", s=80, label="Điểm a")
    ax3.scatter([df["thoi_gian"].iloc[b_idx]], [f_b], color="red", s=80, label="Điểm b (thực)")
    ax3.plot(x_lin_ts, y_lin, "--", color="orange", label="Đường xấp xỉ f(a)+f'(a)(x-a)")
    ax3.legend()
    st.pyplot(fig3)

    st.markdown(f"- **f(b)** = {f_b:.6g}")
    st.markdown(f"- **f xấp xỉ** = {f_xapxi:.6g}")
    st.markdown(f"- **Sai số tuyệt đối:** {sai_so:.6g}")
    st.markdown(f"- **Sai số tương đối:** {pct_sai:.3f}%")

# ============================================================
# ✅ KẾT LUẬN
# ============================================================

st.markdown("---")
st.markdown("### ✅ Kết luận:")
st.markdown("""
- Làm tròn làm mất chi tiết nhỏ và có thể đảo dấu đạo hàm.  
- Làm tròn quá mức khiến đạo hàm và điểm MVT bị sai lệch.  
- Sai số đạo hàm (MAE/Max Error) tăng theo mức làm tròn.  
- Nếu dùng đạo hàm để dự báo, việc làm tròn thô có thể gây sai hướng.  
- 👉 Hạn chế làm tròn trước khi tính đạo hàm; nên làm mịn dữ liệu (smoothing) thay vì làm tròn thô.
""")
