# ==========================================
# 1. Builder Stage
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Cài đặt gcc và các thư viện cần thiết để build Python packages
RUN apt-get update && apt-get install -y gcc build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Cài đặt thư viện vào thư mục người dùng (.local)
RUN pip install --no-cache-dir --user -r requirements.txt

# ==========================================
# 2. Runtime Stage
# ==========================================
FROM python:3.11-slim AS runtime

# Cấu hình biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# Tạo user không có quyền root (Tăng tính bảo mật)
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy thư viện đã cài đặt từ Builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy mã nguồn dự án vào container
COPY . .

# Trao quyền sở hữu thư mục cho appuser
RUN chown -R appuser:appuser /app

# Chuyển sang user không phải root
USER appuser

EXPOSE 8501

# Lệnh khởi chạy ứng dụng Streamlit (Sử dụng biến $PORT nếu có, mặc định là 8501)
CMD sh -c 'streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0'
