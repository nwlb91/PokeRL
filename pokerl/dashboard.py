"""Lightweight web dashboard for monitoring PokeRL training.

Runs a Flask server in a background thread, serving a live dashboard
that polls a JSON API for training metrics. Designed to be started
alongside the Trainer and updated via the progress_callback interface.
"""

import json
import threading
import time
import logging
from pathlib import Path

from flask import Flask, Response, send_from_directory

logger = logging.getLogger(__name__)

# Shared state updated by the trainer's progress callback
_state = {
    "battle_count": 0,
    "total_battles": 0,
    "train_wr": [],          # [(battle_count, wr), ...]
    "greedy_wr": [],         # [(battle_count, wr), ...]
    "baseline_team1": [],    # [(battle_count, wr), ...]
    "baseline_team2": [],    # [(battle_count, wr), ...]
    "metrics": [],           # [{battle_count, policy_loss, value_loss, entropy, ...}, ...]
    "plateau": None,         # {is_plateau, streak, slope} or None
    "league_size": 0,
    "agent1_wins": 0,
    "agent1_losses": 0,
    "agent2_wins": 0,
    "agent2_losses": 0,
    "start_time": time.time(),
}
_lock = threading.Lock()


def update_dashboard(
    battle_count,
    total_battles,
    plateau_detector,
    metrics_history,
    greedy_eval_results,
    train_wr_history,
    baseline_team1=None,
    baseline_team2=None,
    league_size=0,
    agent1_wins=0,
    agent1_losses=0,
    agent2_wins=0,
    agent2_losses=0,
):
    """Progress callback compatible with Trainer._progress_callback signature."""
    with _lock:
        _state["battle_count"] = battle_count
        _state["total_battles"] = total_battles
        _state["train_wr"] = list(train_wr_history) if train_wr_history else []
        _state["greedy_wr"] = list(greedy_eval_results) if greedy_eval_results else []
        _state["baseline_team1"] = list(baseline_team1) if baseline_team1 else []
        _state["baseline_team2"] = list(baseline_team2) if baseline_team2 else []
        _state["metrics"] = [dict(m) for m in metrics_history] if metrics_history else []
        _state["league_size"] = league_size
        _state["agent1_wins"] = agent1_wins
        _state["agent1_losses"] = agent1_losses
        _state["agent2_wins"] = agent2_wins
        _state["agent2_losses"] = agent2_losses

        if plateau_detector is not None:
            info = plateau_detector.latest_info
            if info:
                _state["plateau"] = {
                    "is_plateau": info.is_plateau,
                    "streak": info.streak,
                    "slope": round(info.slope, 6),
                }
            else:
                _state["plateau"] = None


def _create_app() -> Flask:
    app = Flask(__name__)
    app.logger.setLevel(logging.WARNING)  # quiet Flask logs

    static_dir = Path(__file__).parent / "dashboard_static"

    @app.route("/")
    def index():
        return send_from_directory(str(static_dir), "index.html")

    @app.route("/api/metrics")
    def api_metrics():
        with _lock:
            data = json.dumps(_state)
        return Response(data, mimetype="application/json")

    return app


def start_dashboard(host: str = "0.0.0.0", port: int = 5555):
    """Start the dashboard web server in a background daemon thread."""
    app = _create_app()

    # Suppress Flask/Werkzeug startup banner
    wlog = logging.getLogger("werkzeug")
    wlog.setLevel(logging.WARNING)

    thread = threading.Thread(
        target=app.run,
        kwargs={"host": host, "port": port, "debug": False, "use_reloader": False},
        daemon=True,
    )
    thread.start()
    logger.info(f"Dashboard started at http://{host}:{port}")
    return thread
