"""Lightweight web dashboard for monitoring PokeRL training.

Runs a Flask server in a background thread, serving a live dashboard
that polls a JSON API for training metrics. Designed to be started
alongside the Trainer and updated via the progress_callback interface.
"""

import json
import math
import threading
import time
import logging
import traceback
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
    "last_update": time.time(),
}
_lock = threading.Lock()


def _sanitize(val):
    """Make a value JSON-safe (handle NaN, Inf, tensors, etc.)."""
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(val, dict):
        return {k: _sanitize(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_sanitize(v) for v in val]
    # Handle PyTorch tensors or numpy scalars
    if hasattr(val, 'item'):
        try:
            return _sanitize(val.item())
        except Exception:
            return None
    return val


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
    **kwargs,
):
    """Progress callback compatible with Trainer._progress_callback signature."""
    try:
        with _lock:
            _state["battle_count"] = battle_count
            _state["total_battles"] = total_battles
            _state["train_wr"] = list(train_wr_history) if train_wr_history else []
            _state["greedy_wr"] = list(greedy_eval_results) if greedy_eval_results else []
            _state["baseline_team1"] = list(baseline_team1) if baseline_team1 else []
            _state["baseline_team2"] = list(baseline_team2) if baseline_team2 else []
            _state["metrics"] = _sanitize([dict(m) for m in metrics_history]) if metrics_history else []
            _state["league_size"] = league_size
            _state["agent1_wins"] = agent1_wins
            _state["agent1_losses"] = agent1_losses
            _state["agent2_wins"] = agent2_wins
            _state["agent2_losses"] = agent2_losses
            _state["last_update"] = time.time()

            if plateau_detector is not None:
                try:
                    is_plateau = plateau_detector._streak >= plateau_detector.patience
                    _state["plateau"] = {
                        "is_plateau": is_plateau,
                        "streak": plateau_detector._streak,
                        "slope": 0.0,
                    }
                except AttributeError:
                    _state["plateau"] = None
            else:
                _state["plateau"] = None
    except Exception:
        logger.error(f"Dashboard update failed: {traceback.format_exc()}")


def _create_app() -> Flask:
    app = Flask(__name__)
    app.logger.setLevel(logging.WARNING)  # quiet Flask logs

    static_dir = Path(__file__).parent / "dashboard_static"

    @app.route("/")
    def index():
        return send_from_directory(str(static_dir), "index.html")

    @app.route("/api/metrics")
    def api_metrics():
        try:
            with _lock:
                data = json.dumps(_state)
            return Response(data, mimetype="application/json")
        except (ValueError, TypeError) as e:
            logger.error(f"Dashboard JSON serialization failed: {e}")
            # Return at least the battle count
            fallback = json.dumps({"battle_count": _state.get("battle_count", 0),
                                   "error": str(e)})
            return Response(fallback, status=500, mimetype="application/json")

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
