# Low-Light Road Accident Detection — Thesis POC

Streamlit demo comparing three pipelines on CCTV frames:
Raw, Zero-DCE, and Zero-3DCE enhancement, each paired with a YOLO model
trained on the corresponding dataset variant.

## Setup

1. In Google Drive, share these files as "Anyone with the link" and copy each
   file ID (the string between `/d/` and `/view`):
   - `thesis_exports/raw_best.pt`
   - `thesis_exports/zerodce_best.pt`
   - `thesis_exports/zero3dce_best.pt`
   - `zero3dce/checkpoints/zero3dce_final.pth`
2. Put the IDs in Streamlit secrets (`.streamlit/secrets.toml`, see
   `secrets.toml.example`) — or paste them into `DEFAULT_DRIVE_IDS` in `app.py`.
   Zero-DCE weights download automatically from the official GitHub repo.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo (weights are gitignored — they download at runtime).
2. share.streamlit.io -> New app -> select repo, main file `app.py`.
3. App settings -> Secrets -> paste the `[drive_ids]` block.

First launch downloads ~60 MB of weights, then everything is cached.
CPU inference on a 640 px frame is ~1–2 s per pipeline.
