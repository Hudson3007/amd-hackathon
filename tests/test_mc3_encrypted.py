"""An encrypted file must never contribute text, whatever its password setup.

Mini-Challenge 3 grades one question as unanswerable precisely because the
answer sits inside an encrypted PDF. A parser that reads it scores zero on
that question, and looks like a confident guess rather than a bug.

The dangerous case is not the locked file, it is the file encrypted with an
EMPTY user password: reader.decrypt("") succeeds on it, so any code that
tries a no-password decrypt and reads the result will happily index a document
it had no permission to open.
"""

import importlib.util
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mc3", "corpus.py")

spec = importlib.util.spec_from_file_location("mc3_corpus_enc", CORPUS)
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)

SECRET = "SECRET_UNIT_PRICE_4471B is 91.50 per unit"


def make_encrypted(owner_pw, user_pw):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), SECRET)
    data = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw=owner_pw, user_pw=user_pw
    )
    doc.close()
    return data


def main():
    try:
        import fitz  # noqa: F401
    except ImportError:
        print("SKIP PyMuPDF unavailable, cannot build an encrypted PDF")
        return 0

    cases = [
        ("empty user password (decrypt('') would succeed)", "owner", ""),
        ("real user password", "owner", "s3cret"),
        ("empty owner and user password", "", ""),
    ]

    failures = 0
    for label, owner, user in cases:
        text = corpus.parse_pdf(make_encrypted(owner, user))
        leaked = "91.50" in text
        if text.strip() == "" and not leaked:
            print(f"  PASS  refused: {label}")
        else:
            print(f"  FAIL  leaked {len(text)} chars: {label}")
            print(f"        {text[:100]!r}")
            failures += 1

    # An ordinary unencrypted PDF must still be readable, or the refusal above
    # would be indistinguishable from a parser that is simply broken.
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "max junction temperature 94 C")
    plain = doc.tobytes()
    doc.close()
    if "94" in corpus.parse_pdf(plain):
        print("  PASS  unencrypted PDF still parses")
    else:
        print("  FAIL  unencrypted PDF returned nothing - parser is broken")
        failures += 1

    print(f"\n{'all encrypted-PDF cases refused' if not failures else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
