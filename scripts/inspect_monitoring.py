#!/usr/bin/env python3
"""CLI utility to inspect logged predictions and request distributions."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.monitoring import compute_monitoring_summary, read_prediction_logs


def main() -> None:
    print("=" * 70)
    print("INSPECTING PREDICTION & SERVICE MONITORING LOGS")
    print("=" * 70)

    summary = compute_monitoring_summary()
    print(f"Total Logged Events:     {summary['total_logged_events']}")
    print(f"Successful Predictions:  {summary['successful_predictions']}")
    print(f"Error / Invalid Events:  {summary['error_events']}")
    print(f"Average Latency:         {summary['average_latency_ms']} ms")
    print(f"Prediction Distribution: {summary['prediction_distribution']}")
    print(f"\nMonitoring Note: {summary['monitoring_note']}")

    logs = read_prediction_logs()
    if logs:
        print("\nLast 3 Logged Events:")
        for log in logs[-3:]:
            req_id = log.get("request_id", "")[:8]
            status_val = log.get("status")
            if status_val == "success":
                c_name = log.get("class_name")
                prob = log.get("probability")
                lat = log.get("latency_ms")
                print(
                    f"  [{log.get('timestamp')}] REQ:{req_id} -> {c_name} (Prob: {prob}, {lat}ms)"
                )
            else:
                err_t = log.get("error_type")
                print(f"  [{log.get('timestamp')}] REQ:{req_id} -> ERROR: {err_t}")

    print("=" * 70)


if __name__ == "__main__":
    main()
