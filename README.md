# 🔐 PrivSecure – AI-Powered Secure Document Redaction & Verification

## 🎯 Problem Statement

Organizations frequently process identity documents containing sensitive personal information. Manual redaction is time-consuming and error-prone, while secure verification of document authenticity remains a challenge. PrivSecure addresses these issues by automating redaction and enabling tamper detection through a blockchain-inspired verification mechanism.

## 💡 Key Highlights

- Automated OCR-based document processing
- Privacy-preserving sensitive data redaction
- Secure cloud storage with Firebase
- Blockchain-inspired immutable verification ledger
- Tamper detection using SHA-256 hashing
- Cloud deployment on Railway
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-Web_App-green)
![Firebase](https://img.shields.io/badge/Firebase-Realtime_Database-orange)
![OCR](https://img.shields.io/badge/OCR-OCR.Space-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)

PrivSecure is an AI-powered document privacy platform that automatically detects, extracts, and redacts sensitive identity information from uploaded documents while preserving authenticity using an immutable blockchain-inspired verification ledger.

---

## 📌 Features

- 📄 Automatic OCR-based document text extraction
- 🤖 AI-powered identification of sensitive information
- 🔒 Automatic redaction of personal identity fields
- ☁️ Secure cloud storage using Firebase Realtime Database
- ⛓ Blockchain-inspired immutable verification ledger
- 🔍 Tamper detection using SHA-256 hashing
- 🔑 Secure Verification ID & Access Key system
- 🌐 Railway cloud deployment
- 📑 Downloadable redacted documents

---

# 🛠 Tech Stack

### Backend

- Python
- Flask

### OCR

- OCR.Space API

### Database

- Firebase Realtime Database

### Cloud

- Railway

### Security

- SHA-256 Hashing
- Blockchain-inspired Immutable Ledger

---

# 📂 Project Structure

```
PrivSecure/
│
├── app.py
├── storage_backend.py
├── data_redaction_testing.py
├── requirements.txt
├── templates/
├── static/
├── uploads/
├── local_results.json
└── README.md
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Harinimuthukumar05/PrivSecure.git
```

Go into the project

```bash
cd PrivSecure
```

Create virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python app.py
```

---

# 🔑 Environment Variables

Create a `.env` file.

```env
GROQ_API_KEY=YOUR_KEY
OCR_SPACE_API_KEY=YOUR_KEY
```

Firebase can be configured using either:

- `serviceAccountKey.json`
- or

```
FIREBASE_CREDENTIALS
```

---

# 🔒 Security Architecture

1. User uploads document.
2. OCR extracts text.
3. Sensitive identity information is detected.
4. Data is redacted.
5. Extracted information is securely stored in Firebase.
6. SHA-256 hash is generated.
7. Hash is stored in an immutable blockchain-inspired ledger.
8. Verification ID and Access Key are generated.
9. Future verification detects any tampering.

---

# 🔍 Verification System

Each uploaded document generates:

- Verification ID (Public)

```
VERIFY-XXXXXX
```

- Access Key (Private)

```
ACCESS-XXXXXXXXX
```

Verification ID is used to verify authenticity.

Access Key is required to retrieve stored information.

---

# ☁️ Deployment

The application is deployed on Railway.

---

# 📸 Screenshots

> Add screenshots of:

- Home Page
- Upload Document
- Redacted Output
- Verification Page
- Firebase Database

---

# 🚀 Future Improvements

- Multi-user authentication
- PDF digital signatures
- Blockchain integration (Ethereum/Hyperledger)
- Audit dashboard
- Role-based access control
- AI document classification

---

# 👨‍💻 Team

**Harini Muthukumar**

Backend Development
AI Integration
Firebase Integration
Cloud Deployment

**<Friend Name>**

Frontend Development
UI/UX
Testing
Documentation

---

# 📄 License

This project is developed for academic and research purposes.

---

# ⭐ If you found this project useful, consider giving it a star!