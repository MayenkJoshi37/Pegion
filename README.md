# 🕊️ Pegion

Pegion is a secure file encryption and decryption tool designed to protect sensitive data. It offers a simple command-line interface for encrypting files before sharing and decrypting received files, ensuring data confidentiality and integrity.

## 🚀 Features
- **File Encryption**: Securely encrypt any file using industry-standard algorithms.  
- **File Decryption**: Decrypt files that were encrypted with Pegion.  
- **Cross-Platform**: Works on Windows, macOS, and Linux.  
- **Lightweight & Fast**: Minimal dependencies and high performance.

## 🛠️ Installation

### Prerequisites
- Anaconda or Miniconda (for `environment.yml` support)

### Setup
```bash
git clone https://github.com/MayenkJoshi37/Pegion.git
cd Pegion
conda env create -f environment.yml
conda activate pegion
The environment.yml file lists all required dependencies and libraries.
```

📦 Usage
Encrypt a file

```bash
python encrypt_file.py \
  --input  /path/to/your/plain_file.txt \
  --output /path/to/save/encrypted_file.enc
```
Decrypt a file
```bash
python decrypt_file.py \
  --input  /path/to/your/encrypted_file.enc \
  --output /path/to/save/decrypted_file.txt
```
Replace /path/to/... with your actual file paths.

📁 Project Structure
```bash
Pegion/
├── encrypt_file.py        # Script to encrypt files
├── decrypt_file.py        # Script to decrypt files
├── environment.yml        # Conda environment specification
├── requirements.txt       # (Optional) pip requirements
└── README.md              # This file
```


🤝 Contributing
Fork the repository

Create a feature branch: git checkout -b feature/YourFeature

Commit your changes: git commit -m "Add YourFeature"

Push to your branch: git push origin feature/YourFeature

Open a Pull Request


📫 Contact
Author: Mayenk Joshi
Email: mayenk.joshi23@vit.edu
