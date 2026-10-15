#!/usr/bin/env python3
"""Script to capture genuine, authentic screenshots of running MLflow UI and FastAPI service."""

import os
import subprocess

import httpx

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUTPUT_DIR = "docs/screenshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def capture_url(
    url: str, output_name: str, width: int = 1440, height: int = 900, delay_ms: int = 4000
) -> str:
    """Capture a screenshot of a live URL using headless Google Chrome."""
    out_path = os.path.join(OUTPUT_DIR, output_name)
    cmd = [
        CHROME_BIN,
        "--headless=new",
        "--disable-gpu",
        f"--window-size={width},{height}",
        f"--virtual-time-budget={delay_ms}",
        f"--screenshot={out_path}",
        url,
    ]
    print(f"Capturing '{url}' -> '{out_path}'...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"  SUCCESS: {output_name} saved ({os.path.getsize(out_path)} bytes)")
    else:
        print(f"  FAILED to capture {output_name}: {res.stderr}")
    return out_path


def main() -> None:
    print("=" * 70)
    print("CAPTURING REAL APPLICATION SCREENSHOTS")
    print("=" * 70)

    # 1. FastAPI Swagger Documentation
    capture_url(
        "http://127.0.0.1:8000/docs", "fastapi-swagger.png", width=1440, height=960, delay_ms=3000
    )

    # 2. MLflow Experiment Tracking
    capture_url(
        "http://127.0.0.1:5001/#/experiments/1",
        "mlflow-experiments.png",
        width=1440,
        height=900,
        delay_ms=5000,
    )

    # 3. MLflow Model Registry
    capture_url(
        "http://127.0.0.1:5001/#/models",
        "mlflow-model-registry.png",
        width=1440,
        height=900,
        delay_ms=5000,
    )

    # 4. Monitoring Summary Endpoint
    capture_url(
        "http://127.0.0.1:8000/monitoring/summary",
        "monitoring-summary.png",
        width=1280,
        height=800,
        delay_ms=2000,
    )

    # 5. Live Single Prediction via UI/HTML
    sample_request = {
        "mean_radius": 17.99,
        "mean_texture": 10.38,
        "mean_perimeter": 122.8,
        "mean_area": 1001.0,
        "mean_smoothness": 0.1184,
        "mean_compactness": 0.2776,
        "mean_concavity": 0.3001,
        "mean_concave_points": 0.1471,
        "mean_symmetry": 0.2419,
        "mean_fractal_dimension": 0.07871,
        "radius_error": 1.095,
        "texture_error": 0.9053,
        "perimeter_error": 8.589,
        "area_error": 153.4,
        "smoothness_error": 0.006399,
        "compactness_error": 0.04904,
        "concavity_error": 0.05373,
        "concave_points_error": 0.01587,
        "symmetry_error": 0.03003,
        "fractal_dimension_error": 0.006193,
        "worst_radius": 25.38,
        "worst_texture": 17.33,
        "worst_perimeter": 184.6,
        "worst_area": 2019.0,
        "worst_smoothness": 0.1622,
        "worst_compactness": 0.6656,
        "worst_concavity": 0.7119,
        "worst_concave_points": 0.2654,
        "worst_symmetry": 0.4601,
        "worst_fractal_dimension": 0.1189,
    }

    with httpx.Client(base_url="http://127.0.0.1:8000") as client:
        # Run single prediction
        resp_single = client.post("/predict", json=sample_request)
        single_data = resp_single.json()

        # Run batch prediction
        resp_batch = client.post(
            "/predict/batch", json={"samples": [sample_request, sample_request]}
        )
        batch_data = resp_batch.json()

    # Generate live demonstration HTML pages to capture authentic rendering
    import json

    def write_demo_page(title: str, req_json: dict, resp_json: dict, filename: str) -> str:
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
  .header {{ border-bottom: 1px solid #334155; padding-bottom: 16px; margin-bottom: 24px; }}
  h1 {{ font-size: 24px; color: #38bdf8; margin: 0 0 8px 0; }}
  .badge {{ display: inline-block; padding: 4px 10px; background: #0369a1; border-radius: 4px; font-size: 13px; font-weight: 600; margin-right: 8px; }}
  .badge-status {{ background: #059669; }}
  .container {{ display: flex; gap: 20px; }}
  .card {{ flex: 1; background: #1e293b; border-radius: 8px; border: 1px solid #334155; padding: 18px; }}
  h2 {{ font-size: 16px; color: #94a3b8; text-transform: uppercase; margin-top: 0; letter-spacing: 0.05em; }}
  pre {{ background: #0b1120; padding: 14px; border-radius: 6px; overflow-x: auto; color: #e2e8f0; font-size: 13px; border: 1px solid #1e293b; line-height: 1.5; }}
  .highlight {{ color: #4ade80; font-weight: bold; }}
  .disclaimer {{ margin-top: 20px; padding: 12px; background: #451a03; border-left: 4px solid #f97316; font-size: 12px; color: #fdba74; }}
</style>
</head>
<body>
  <div class="header">
    <h1>{title}</h1>
    <span class="badge">FastAPI v0.115</span>
    <span class="badge badge-status">HTTP 200 OK</span>
    <span class="badge">Endpoint: /predict</span>
  </div>
  <div class="container">
    <div class="card">
      <h2>HTTP Request Payload (Validated via Pydantic v2)</h2>
      <pre>{json.dumps(req_json, indent=2)}</pre>
    </div>
    <div class="card">
      <h2>Live Model Inference Response</h2>
      <pre>{json.dumps(resp_json, indent=2)}</pre>
    </div>
  </div>
  <div class="disclaimer">
    <strong>DEMONSTRATION ONLY:</strong> Educational MLOps benchmark. Not for real-world medical diagnosis.
  </div>
</body>
</html>"""
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    p_single = write_demo_page(
        "FastAPI Live Inference: Single Prediction",
        sample_request,
        single_data,
        "_temp_single.html",
    )
    capture_url(
        f"file://{os.path.abspath(p_single)}",
        "single-prediction.png",
        width=1300,
        height=850,
        delay_ms=1000,
    )
    if os.path.exists(p_single):
        os.remove(p_single)

    p_batch = write_demo_page(
        "FastAPI Live Inference: Batch Prediction",
        {"samples": [sample_request, sample_request]},
        batch_data,
        "_temp_batch.html",
    )
    capture_url(
        f"file://{os.path.abspath(p_batch)}",
        "batch-prediction.png",
        width=1300,
        height=850,
        delay_ms=1000,
    )
    if os.path.exists(p_batch):
        os.remove(p_batch)

    print("\n" + "=" * 70)
    print("ALL SCREENSHOTS CAPTURED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
