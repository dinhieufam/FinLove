import yfinance as yf

# Ví dụ tải dữ liệu cho AAPL (hoặc thay bằng ticker khác như 'XLK' cho ETF)
data = yf.download('AAPL', start='2023-01-01', end='2025-10-09')

# Show ra 5 dòng đầu
print(data.head())

# Show ra toàn bộ dữ liệu (nếu dữ liệu ngắn), hoặc dùng describe để tóm tắt
print(data.describe())