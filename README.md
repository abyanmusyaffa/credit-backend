# Credit Prediction Backend (Flask)

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Pastikan model.pkl ada di folder ini
# Jalankan server
python app.py
```

Server berjalan di: http://localhost:5000

## Deploy ke Render

1. Push folder `credit-backend` ke GitHub repository
2. Buka [render.com](https://render.com) → New → Web Service
3. Connect GitHub repo
4. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Environment**: Python 3
5. Tambahkan file `model.pkl` ke repo (atau upload via environment)
6. Deploy!

## API Endpoints

### POST /predict
Menerima data pemohon kredit dan mengembalikan prediksi.

### GET /health
Cek status server.
