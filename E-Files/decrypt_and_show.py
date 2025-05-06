import os
from cryptography.fernet import Fernet

KEY_FILE = "secret.key"
ENCRYPTED_DIR = "E-Files"

def load_key():
    with open(KEY_FILE, "rb") as f:
        return f.read()

def decrypt_and_display_files(directory=ENCRYPTED_DIR):
    key = load_key()
    fernet = Fernet(key)

    if not os.path.exists(directory):
        print("Encrypted directory not found.")
        return

    for filename in os.listdir(directory):
        if filename.endswith(".enc"):
            path = os.path.join(directory, filename)
            with open(path, "rb") as f:
                encrypted_data = f.read()
            try:
                decrypted_data = fernet.decrypt(encrypted_data)
                print(f"\n--- Decrypted content of {filename} ---")
                print(decrypted_data.decode())
            except Exception as e:
                print(f"Failed to decrypt {filename}: {e}")

# Example usage
decrypt_and_display_files()