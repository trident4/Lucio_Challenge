import fitz
import sys

def verify_meta():
    path = "/tmp/META-Q3-2024-Earnings-Call-Transcript.pdf"
    doc = fitz.open(path)
    with open("/tmp/meta_q3.txt", "w") as f:
        for page in doc:
            f.write(page.get_text())
            
if __name__ == "__main__":
    verify_meta()
