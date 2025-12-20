import os

def main():
    os.makedirs("data/raw", exist_ok=True)
    print("Download datasets manually if not present.")
    print("DFDC: https://www.kaggle.com/c/deepfake-detection-challenge/data")
    print("FaceForensics++: https://github.com/ondyari/FaceForensics")

if __name__ == "__main__":
    main()