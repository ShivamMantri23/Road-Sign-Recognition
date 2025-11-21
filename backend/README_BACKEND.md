# Backend API - Road Sign Recognition (GTSRB)

This is a Flask API that loads the trained CNN model and performs predictions on uploaded images.

## 🚀 Deployment (Render)

1. Create a Render account → https://render.com
2. Click **New → Web Service**
3. Connect your GitHub repo
4. Choose the `backend/` folder
5. Configure:
   - Runtime: **Python**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
6. Deploy → Copy the API URL

Example:
