import seaborn as sns
import matplotlib.pyplot as plt

# Chọn các biến cần phân tích
corr_df = df[[
    "Nhiệt độ",
    "Độ ẩm",
    "Lượng mưa"
]].copy()

# Nếu có thêm pressure, wind, clouds thì thêm vào đây

corr = corr_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)
plt.title("Ma trận tương quan giữa các biến thời tiết")
plt.tight_layout()
plt.savefig("weather_correlation.png", dpi=300)
plt.show()
