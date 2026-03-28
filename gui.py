#!/usr/bin/env python3
"""PokeRL GUI — Manage training, evaluation, and Showdown server."""

import asyncio
import atexit
import logging
import os
import queue
import signal
import shutil
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch

import torch

from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import (
    ServerConfiguration,
    ShowdownServerConfiguration,
)
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from pokerl.agent import PPOAgent
from pokerl.config import Config
from pokerl.env import RLPlayer, create_player, load_team
from pokerl.plateau import PlateauDetector
from pokerl.trainer import Trainer

logger = logging.getLogger("pokerl.gui")

# ---------------------------------------------------------------------------
# Async bridge — run coroutines from tkinter callbacks
# ---------------------------------------------------------------------------

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_loop_lock = threading.Lock()


def _ensure_event_loop():
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return
        _loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(target=_loop.run_forever, daemon=True)
        _loop_thread.start()


def run_async(coro):
    """Schedule *coro* on the background event loop, return a Future."""
    _ensure_event_loop()
    with _loop_lock:
        return asyncio.run_coroutine_threadsafe(coro, _loop)


# ---------------------------------------------------------------------------
# Log handler that pushes records into a queue for the GUI
# ---------------------------------------------------------------------------


class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            self.log_queue.put_nowait(self.format(record))
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

PADDING = {"padx": 6, "pady": 3}


class PokeRLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PokeRL \u2014 Pokemon Battle RL Trainer")
        self.geometry("960x920")
        self.minsize(800, 600)

        # --- State ---
        self._showdown_proc: Optional[subprocess.Popen] = None
        atexit.register(self._cleanup_server)
        self._trainer: Optional[Trainer] = None
        self._training_future = None
        self._training_stop = threading.Event()
        self._eval_future = None
        self._training_start_time: Optional[float] = None
        self.log_queue: queue.Queue = queue.Queue(maxsize=5000)

        # Install queue log handler on root logger
        qh = QueueLogHandler(self.log_queue)
        qh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(qh)
        logging.getLogger().setLevel(logging.INFO)
        # Silence poke-env's per-message WebSocket logging — it floods the
        # GUI log when running many concurrent battles.
        logging.getLogger("poke_env").setLevel(logging.WARNING)

        # Tkinter variables
        self.team1_var = tk.StringVar(value="teams/team1.txt")
        self.team2_var = tk.StringVar(value="teams/team2.txt")
        self.checkpoint_var = tk.StringVar(value="")
        self.fresh_start_var = tk.BooleanVar(value=False)
        self.output_var = tk.StringVar(value="checkpoints")
        self.format_var = tk.StringVar(value="gen9nationaldexmonotype")
        self.battles_var = tk.IntVar(value=100000)
        self.lr_var = tk.DoubleVar(value=3e-4)
        self.hidden_var = tk.IntVar(value=256)
        self.device_var = tk.StringVar(value="cpu")
        self.server_port_var = tk.IntVar(value=8000)
        self.showdown_path_var = tk.StringVar(value="pokemon-showdown")
        self.concurrent_battles_var = tk.IntVar(value=4)
        self.eval_n_battles_var = tk.IntVar(value=50)

        # Challenge tab variables
        self.challenge_checkpoint_var = tk.StringVar(value="")
        self.challenge_team_var = tk.StringVar(value="")
        self.challenge_server_var = tk.StringVar(value="local")  # "local" or "live"
        self.challenge_local_port_var = tk.IntVar(value=8000)
        self.challenge_username_var = tk.StringVar(value="")
        self.challenge_password_var = tk.StringVar(value="")
        self.challenge_opponent_var = tk.StringVar(value="")
        self.challenge_format_var = tk.StringVar(value="gen9nationaldexmonotype")
        self.challenge_n_var = tk.IntVar(value=1)
        self.challenge_team_select_var = tk.StringVar(value="agent1")
        self._challenge_stop = threading.Event()

        self._build_ui()
        self._poll_log_queue()

    # ----- UI construction --------------------------------------------------

    def _build_ui(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=4, pady=4)

        # --- Tab 1: Training ---
        train_frame = ttk.Frame(notebook)
        notebook.add(train_frame, text="  Training  ")
        self._build_training_tab(train_frame)

        # --- Tab 2: Evaluation ---
        eval_frame = ttk.Frame(notebook)
        notebook.add(eval_frame, text="  Evaluation  ")
        self._build_eval_tab(eval_frame)

        # --- Tab 3: Challenge ---
        challenge_frame = ttk.Frame(notebook)
        notebook.add(challenge_frame, text="  Challenge  ")
        self._build_challenge_tab(challenge_frame)

        # --- Tab 4: Server ---
        server_frame = ttk.Frame(notebook)
        notebook.add(server_frame, text="  Server  ")
        self._build_server_tab(server_frame)

        # --- Log area (always visible) ---
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, state="disabled", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, padx=2, pady=2)

    # -- Training tab --------------------------------------------------------

    def _build_training_tab(self, parent):
        # --- File selection ---
        files = ttk.LabelFrame(parent, text="Files")
        files.pack(fill="x", padx=6, pady=4)

        self._file_row(files, "Team 1:", self.team1_var, 0)
        self._file_row(files, "Team 2:", self.team2_var, 1)
        self._file_row(files, "Resume from checkpoint:", self.checkpoint_var, 2, optional=True)
        ttk.Checkbutton(
            files, text="Fresh start (reset exploration)",
            variable=self.fresh_start_var,
        ).grid(row=2, column=3, sticky="w", **PADDING)
        self._dir_row(files, "Output folder:", self.output_var, 3)

        # --- Hyperparameters ---
        hyper = ttk.LabelFrame(parent, text="Hyperparameters")
        hyper.pack(fill="x", padx=6, pady=4)

        row = 0
        ttk.Label(hyper, text="Battle format:").grid(row=row, column=0, sticky="e", **PADDING)
        ttk.Entry(hyper, textvariable=self.format_var, width=30).grid(row=row, column=1, sticky="w", **PADDING)
        ttk.Label(hyper, text="Device:").grid(row=row, column=2, sticky="e", **PADDING)
        ttk.Combobox(hyper, textvariable=self.device_var, values=["cpu", "cuda"], width=8, state="readonly").grid(row=row, column=3, sticky="w", **PADDING)

        row = 1
        ttk.Label(hyper, text="Total battles:").grid(row=row, column=0, sticky="e", **PADDING)
        ttk.Entry(hyper, textvariable=self.battles_var, width=12).grid(row=row, column=1, sticky="w", **PADDING)
        ttk.Label(hyper, text="Learning rate:").grid(row=row, column=2, sticky="e", **PADDING)
        ttk.Entry(hyper, textvariable=self.lr_var, width=12).grid(row=row, column=3, sticky="w", **PADDING)

        row = 2
        ttk.Label(hyper, text="Hidden size:").grid(row=row, column=0, sticky="e", **PADDING)
        ttk.Entry(hyper, textvariable=self.hidden_var, width=12).grid(row=row, column=1, sticky="w", **PADDING)
        ttk.Label(hyper, text="Concurrent battles:").grid(row=row, column=2, sticky="e", **PADDING)
        ttk.Spinbox(hyper, textvariable=self.concurrent_battles_var, from_=1, to=64, width=8).grid(row=row, column=3, sticky="w", **PADDING)

        # --- Controls ---
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill="x", padx=6, pady=6)

        self.btn_train = ttk.Button(ctrl, text="Start Training", command=self._on_start_training)
        self.btn_train.pack(side="left", padx=4)
        self.btn_stop_train = ttk.Button(ctrl, text="Stop Training", command=self._on_stop_training, state="disabled")
        self.btn_stop_train.pack(side="left", padx=4)

        self.train_status_var = tk.StringVar(value="Idle")
        ttk.Label(ctrl, textvariable=self.train_status_var, foreground="gray").pack(side="left", padx=12)

        # --- Progress ---
        prog_frame = ttk.LabelFrame(parent, text="Progress")
        prog_frame.pack(fill="x", padx=6, pady=4)

        self.train_progress = ttk.Progressbar(prog_frame, mode="determinate", maximum=100, value=0)
        self.train_progress.pack(fill="x", padx=6, pady=(4, 2))

        time_frame = ttk.Frame(prog_frame)
        time_frame.pack(fill="x", padx=6, pady=(0, 4))
        self.train_elapsed_var = tk.StringVar(value="Elapsed: --")
        self.train_eta_var = tk.StringVar(value="ETA: --")
        self.train_battles_var = tk.StringVar(value="")
        self.train_plateau_var = tk.StringVar(value="")
        ttk.Label(time_frame, textvariable=self.train_battles_var, foreground="gray").pack(side="left", padx=(0, 16))
        ttk.Label(time_frame, textvariable=self.train_elapsed_var).pack(side="left", padx=(0, 16))
        ttk.Label(time_frame, textvariable=self.train_eta_var).pack(side="left")
        self._plateau_label = ttk.Label(time_frame, textvariable=self.train_plateau_var, foreground="red")
        self._plateau_label.pack(side="right", padx=(16, 0))

        # --- Summary stats ---
        stats_frame = ttk.LabelFrame(parent, text="Training Stats")
        stats_frame.pack(fill="x", padx=6, pady=4)

        self.stat_greedy_wr_var = tk.StringVar(value="Greedy WR: --")
        self.stat_train_wr_var = tk.StringVar(value="Train WR: --")
        self.stat_policy_loss_var = tk.StringVar(value="Policy Loss: --")
        self.stat_value_loss_var = tk.StringVar(value="Value Loss: --")
        self.stat_entropy_var = tk.StringVar(value="Entropy: --")
        self.stat_expl_var_var = tk.StringVar(value="Expl. Var: --")
        self.stat_ep_return_var = tk.StringVar(value="Ep. Return: --")

        for i, var in enumerate([
            self.stat_greedy_wr_var, self.stat_train_wr_var,
            self.stat_policy_loss_var, self.stat_value_loss_var,
            self.stat_entropy_var, self.stat_expl_var_var,
            self.stat_ep_return_var,
        ]):
            ttk.Label(stats_frame, textvariable=var, font=("Consolas", 9)).grid(
                row=i // 4, column=i % 4, sticky="w", padx=8, pady=1,
            )

        # --- Charts: 2x2 grid (win rate, loss curves, entropy, explained variance) ---
        chart_frame = ttk.LabelFrame(parent, text="Training Charts")
        chart_frame.pack(fill="both", expand=True, padx=6, pady=4)

        self._fig, self._axes = plt.subplots(2, 2, figsize=(7, 4.2), dpi=90)
        self._fig.patch.set_facecolor("#f0f0f0")
        self._ax = self._axes[0, 0]  # keep reference for plateau chart compat
        self._fig.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.0)

        self._canvas = FigureCanvasTkAgg(self._fig, master=chart_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)

    # -- Evaluation tab ------------------------------------------------------

    def _build_eval_tab(self, parent):
        # --- Test group ---
        tg = ttk.LabelFrame(parent, text="Test Group (models to evaluate)")
        tg.pack(fill="both", expand=True, padx=6, pady=4)

        btn_frame = ttk.Frame(tg)
        btn_frame.pack(fill="x")
        ttk.Button(btn_frame, text="Add checkpoint...", command=lambda: self._add_to_listbox(self.test_listbox)).pack(side="left", **PADDING)
        ttk.Button(btn_frame, text="Remove selected", command=lambda: self._remove_from_listbox(self.test_listbox)).pack(side="left", **PADDING)

        self.test_listbox = tk.Listbox(tg, height=5, selectmode="extended")
        self.test_listbox.pack(fill="both", expand=True, padx=4, pady=2)

        # --- Opponents ---
        og = ttk.LabelFrame(parent, text="Opponents (evaluate against)")
        og.pack(fill="both", expand=True, padx=6, pady=4)

        btn_frame2 = ttk.Frame(og)
        btn_frame2.pack(fill="x")
        ttk.Button(btn_frame2, text="Add checkpoint...", command=lambda: self._add_to_listbox(self.opp_listbox)).pack(side="left", **PADDING)
        ttk.Button(btn_frame2, text="Remove selected", command=lambda: self._remove_from_listbox(self.opp_listbox)).pack(side="left", **PADDING)

        self.opp_listbox = tk.Listbox(og, height=5, selectmode="extended")
        self.opp_listbox.pack(fill="both", expand=True, padx=4, pady=2)

        # --- Eval settings ---
        settings = ttk.Frame(parent)
        settings.pack(fill="x", padx=6, pady=2)

        ttk.Label(settings, text="Battles per matchup:").pack(side="left", **PADDING)
        ttk.Entry(settings, textvariable=self.eval_n_battles_var, width=8).pack(side="left", **PADDING)
        ttk.Label(settings, text="Team 1:").pack(side="left", padx=(16, 2))
        ttk.Label(settings, textvariable=self.team1_var, foreground="gray").pack(side="left")
        ttk.Label(settings, text="Team 2:").pack(side="left", padx=(16, 2))
        ttk.Label(settings, textvariable=self.team2_var, foreground="gray").pack(side="left")

        # --- Controls ---
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill="x", padx=6, pady=6)

        self.btn_eval = ttk.Button(ctrl, text="Run Evaluation", command=self._on_run_eval)
        self.btn_eval.pack(side="left", padx=4)
        self.btn_stop_eval = ttk.Button(ctrl, text="Stop Evaluation", command=self._on_stop_eval, state="disabled")
        self.btn_stop_eval.pack(side="left", padx=4)

        self.eval_status_var = tk.StringVar(value="Idle")
        ttk.Label(ctrl, textvariable=self.eval_status_var, foreground="gray").pack(side="left", padx=12)

        # --- Results table ---
        res = ttk.LabelFrame(parent, text="Results")
        res.pack(fill="both", expand=True, padx=6, pady=(0, 4))

        cols = ("Test Model", "Opponent", "Wins", "Losses", "Win Rate")
        self.results_tree = ttk.Treeview(res, columns=cols, show="headings", height=6)
        for c in cols:
            self.results_tree.heading(c, text=c)
            self.results_tree.column(c, width=140, anchor="center")
        self.results_tree.pack(fill="both", expand=True, padx=2, pady=2)

    # -- Server tab ----------------------------------------------------------

    def _build_server_tab(self, parent):
        info = ttk.LabelFrame(parent, text="Pokemon Showdown Server")
        info.pack(fill="x", padx=6, pady=6)

        ttk.Label(info, text="Showdown directory:").grid(row=0, column=0, sticky="e", **PADDING)
        ttk.Entry(info, textvariable=self.showdown_path_var, width=50).grid(row=0, column=1, sticky="ew", **PADDING)
        ttk.Button(info, text="Browse...", command=self._browse_showdown_dir).grid(row=0, column=2, **PADDING)

        ttk.Label(info, text="Port:").grid(row=1, column=0, sticky="e", **PADDING)
        ttk.Entry(info, textvariable=self.server_port_var, width=8).grid(row=1, column=1, sticky="w", **PADDING)

        info.columnconfigure(1, weight=1)

        ctrl = ttk.Frame(parent)
        ctrl.pack(fill="x", padx=6, pady=6)

        self.btn_start_server = ttk.Button(ctrl, text="Start Server", command=self._on_start_server)
        self.btn_start_server.pack(side="left", padx=4)
        self.btn_stop_server = ttk.Button(ctrl, text="Stop Server", command=self._on_stop_server, state="disabled")
        self.btn_stop_server.pack(side="left", padx=4)

        self.server_status_var = tk.StringVar(value="Server: stopped")
        ttk.Label(ctrl, textvariable=self.server_status_var, foreground="gray").pack(side="left", padx=12)

    # -- Challenge tab -------------------------------------------------------

    def _build_challenge_tab(self, parent):
        # --- Model & Team ---
        model_frame = ttk.LabelFrame(parent, text="Model & Team")
        model_frame.pack(fill="x", padx=6, pady=4)

        self._file_row(model_frame, "Checkpoint:", self.challenge_checkpoint_var, 0)
        self._file_row(model_frame, "Team file:", self.challenge_team_var, 1)

        ttk.Label(model_frame, text="Agent team:").grid(row=2, column=0, sticky="e", **PADDING)
        team_combo = ttk.Combobox(
            model_frame,
            textvariable=self.challenge_team_select_var,
            values=["agent1", "agent2"],
            state="readonly",
            width=10,
        )
        team_combo.grid(row=2, column=1, sticky="w", **PADDING)
        ttk.Label(
            model_frame,
            text="(only used for full checkpoints with both agents)",
            foreground="gray",
        ).grid(row=2, column=2, sticky="w", **PADDING)

        # --- Server ---
        server_frame = ttk.LabelFrame(parent, text="Server")
        server_frame.pack(fill="x", padx=6, pady=4)

        row = 0
        ttk.Label(server_frame, text="Server:").grid(row=row, column=0, sticky="e", **PADDING)
        srv_inner = ttk.Frame(server_frame)
        srv_inner.grid(row=row, column=1, sticky="w", **PADDING)
        ttk.Radiobutton(srv_inner, text="Local", variable=self.challenge_server_var, value="local").pack(side="left", padx=(0, 8))
        ttk.Radiobutton(srv_inner, text="Pokemon Showdown (live)", variable=self.challenge_server_var, value="live").pack(side="left")

        row = 1
        ttk.Label(server_frame, text="Local port:").grid(row=row, column=0, sticky="e", **PADDING)
        ttk.Entry(server_frame, textvariable=self.challenge_local_port_var, width=8).grid(row=row, column=1, sticky="w", **PADDING)

        row = 2
        ttk.Label(server_frame, text="Username:").grid(row=row, column=0, sticky="e", **PADDING)
        ttk.Entry(server_frame, textvariable=self.challenge_username_var, width=30).grid(row=row, column=1, sticky="w", **PADDING)

        row = 3
        ttk.Label(server_frame, text="Password:").grid(row=row, column=0, sticky="e", **PADDING)
        pw_entry = ttk.Entry(server_frame, textvariable=self.challenge_password_var, width=30, show="*")
        pw_entry.grid(row=row, column=1, sticky="w", **PADDING)
        ttk.Label(server_frame, text="(required for live server)", foreground="gray").grid(row=row, column=2, sticky="w", **PADDING)

        server_frame.columnconfigure(1, weight=1)

        # --- Challenge settings ---
        ch_frame = ttk.LabelFrame(parent, text="Challenge")
        ch_frame.pack(fill="x", padx=6, pady=4)

        row = 0
        ttk.Label(ch_frame, text="Opponent username:").grid(row=row, column=0, sticky="e", **PADDING)
        ttk.Entry(ch_frame, textvariable=self.challenge_opponent_var, width=30).grid(row=row, column=1, sticky="w", **PADDING)

        row = 1
        ttk.Label(ch_frame, text="Battle format:").grid(row=row, column=0, sticky="e", **PADDING)
        ttk.Entry(ch_frame, textvariable=self.challenge_format_var, width=30).grid(row=row, column=1, sticky="w", **PADDING)

        row = 2
        ttk.Label(ch_frame, text="Number of challenges:").grid(row=row, column=0, sticky="e", **PADDING)
        ttk.Entry(ch_frame, textvariable=self.challenge_n_var, width=8).grid(row=row, column=1, sticky="w", **PADDING)

        ch_frame.columnconfigure(1, weight=1)

        # --- Controls ---
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill="x", padx=6, pady=6)

        self.btn_challenge = ttk.Button(ctrl, text="Send Challenge", command=self._on_send_challenge)
        self.btn_challenge.pack(side="left", padx=4)
        self.btn_stop_challenge = ttk.Button(ctrl, text="Stop", command=self._on_stop_challenge, state="disabled")
        self.btn_stop_challenge.pack(side="left", padx=4)

        self.challenge_status_var = tk.StringVar(value="Idle")
        ttk.Label(ctrl, textvariable=self.challenge_status_var, foreground="gray").pack(side="left", padx=12)

        # --- Results ---
        res_frame = ttk.LabelFrame(parent, text="Results")
        res_frame.pack(fill="both", expand=True, padx=6, pady=(0, 4))

        cols = ("Battle", "Result")
        self.challenge_tree = ttk.Treeview(res_frame, columns=cols, show="headings", height=8)
        self.challenge_tree.heading("Battle", text="Battle #")
        self.challenge_tree.heading("Result", text="Result")
        self.challenge_tree.column("Battle", width=100, anchor="center")
        self.challenge_tree.column("Result", width=200, anchor="center")
        self.challenge_tree.pack(fill="both", expand=True, padx=2, pady=2)

    # ----- Helpers -----------------------------------------------------------

    def _file_row(self, parent, label, var, row, optional=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", **PADDING)
        e = ttk.Entry(parent, textvariable=var, width=55)
        e.grid(row=row, column=1, sticky="ew", **PADDING)
        ttk.Button(parent, text="Browse...", command=lambda: self._browse_file(var)).grid(row=row, column=2, **PADDING)
        if optional:
            ttk.Button(parent, text="Clear", command=lambda: var.set("")).grid(row=row, column=3, **PADDING)
        parent.columnconfigure(1, weight=1)

    def _dir_row(self, parent, label, var, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", **PADDING)
        ttk.Entry(parent, textvariable=var, width=55).grid(row=row, column=1, sticky="ew", **PADDING)
        ttk.Button(parent, text="Browse...", command=lambda: self._browse_dir(var)).grid(row=row, column=2, **PADDING)
        parent.columnconfigure(1, weight=1)

    def _browse_file(self, var):
        path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("Checkpoint files", "*.pt"), ("All files", "*.*")]
        )
        if path:
            var.set(path)

    def _browse_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _browse_showdown_dir(self):
        path = filedialog.askdirectory(title="Select Pokemon Showdown directory")
        if path:
            self.showdown_path_var.set(path)

    def _add_to_listbox(self, listbox: tk.Listbox):
        paths = filedialog.askopenfilenames(
            filetypes=[("Checkpoint files", "*.pt"), ("All files", "*.*")]
        )
        for p in paths:
            if p and p not in listbox.get(0, "end"):
                listbox.insert("end", p)

    def _remove_from_listbox(self, listbox: tk.Listbox):
        for idx in reversed(listbox.curselection()):
            listbox.delete(idx)

    def _log(self, msg: str):
        logger.info(msg)

    _LOG_MAX_LINES = 10_000
    _LOG_TRIM_LINES = 1_000

    def _poll_log_queue(self):
        """Drain the log queue into the ScrolledText widget (max 20 msgs per tick)."""
        try:
            for _ in range(20):
                msg = self.log_queue.get_nowait()
                self.log_text.configure(state="normal")
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
        except queue.Empty:
            pass
        # Trim excess lines to prevent unbounded memory growth
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > self._LOG_MAX_LINES:
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", f"{self._LOG_TRIM_LINES}.0")
            self.log_text.configure(state="disabled")
        self.after(100, self._poll_log_queue)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m:02d}m {s:02d}s"
        if m:
            return f"{m}m {s:02d}s"
        return f"{s}s"

    def _on_progress_update(self, battle_count: int, total_battles: int,
                            plateau_detector: "PlateauDetector | None" = None,
                            metrics_history: "list | None" = None,
                            greedy_eval_results: "list | None" = None,
                            train_wr_history: "list | None" = None,
                            baseline_eval_results_team1: "list | None" = None,
                            baseline_eval_results_team2: "list | None" = None):
        """Update progress bar, time labels, and training charts."""
        pct = battle_count / total_battles * 100 if total_battles else 0
        self.train_progress["value"] = pct
        self.train_battles_var.set(f"{battle_count}/{total_battles} battles")

        if self._training_start_time is not None and battle_count > 0:
            elapsed = time.time() - self._training_start_time
            self.train_elapsed_var.set(f"Elapsed: {self._format_duration(elapsed)}")
            rate = battle_count / elapsed
            remaining = total_battles - battle_count
            eta = remaining / rate
            self.train_eta_var.set(f"ETA: {self._format_duration(eta)}")
        elif self._training_start_time is not None:
            elapsed = time.time() - self._training_start_time
            self.train_elapsed_var.set(f"Elapsed: {self._format_duration(elapsed)}")
            self.train_eta_var.set("ETA: --")

        # Update summary stat labels
        self._update_stat_labels(metrics_history, greedy_eval_results, train_wr_history,
                                 baseline_eval_results_team1, baseline_eval_results_team2)

        # Update all training charts
        self._update_training_charts(plateau_detector, metrics_history, greedy_eval_results,
                                     train_wr_history,
                                     baseline_eval_results_team1, baseline_eval_results_team2)

    def _update_stat_labels(self, metrics_history: "list | None",
                            greedy_eval_results: "list | None",
                            train_wr_history: "list | None" = None,
                            baseline_eval_results_team1: "list | None" = None,
                            baseline_eval_results_team2: "list | None" = None):
        """Update the summary stat labels with latest values."""
        wr_str = ""
        if greedy_eval_results:
            _, gwr = greedy_eval_results[-1]
            wr_str = f"Greedy: {gwr:.1%}"
        if baseline_eval_results_team1 and baseline_eval_results_team2:
            _, bwr1 = baseline_eval_results_team1[-1]
            _, bwr2 = baseline_eval_results_team2[-1]
            bl_str = f"BL T1={bwr1:.1%} T2={bwr2:.1%}"
            wr_str = f"{wr_str} | {bl_str}" if wr_str else bl_str
        if wr_str:
            self.stat_greedy_wr_var.set(wr_str)
        if train_wr_history:
            _, twr = train_wr_history[-1]
            self.stat_train_wr_var.set(f"Train WR: {twr:.1%}")
        if metrics_history:
            m = metrics_history[-1]
            self.stat_policy_loss_var.set(f"Policy Loss: {m['policy_loss']:.4f}")
            self.stat_value_loss_var.set(f"Value Loss: {m['value_loss']:.4f}")
            self.stat_entropy_var.set(f"Entropy: {m['entropy']:.4f}")
            self.stat_expl_var_var.set(f"Expl. Var: {m['explained_variance']:.4f}")
            self.stat_ep_return_var.set(f"Ep. Return: {m['mean_episode_return']:.4f}")

    def _update_training_charts(self, detector: "PlateauDetector | None",
                                metrics_history: "list | None",
                                greedy_eval_results: "list | None",
                                train_wr_history: "list | None" = None,
                                baseline_eval_results_team1: "list | None" = None,
                                baseline_eval_results_team2: "list | None" = None):
        """Redraw all four training charts."""
        ax_wr, ax_loss, ax_entropy, ax_ev = (
            self._axes[0, 0], self._axes[0, 1],
            self._axes[1, 0], self._axes[1, 1],
        )

        # --- Top-left: Win Rate & Plateau Detection ---
        ax_wr.clear()
        ax_wr.set_title("Win Rate", fontsize=9, fontweight="bold")
        ax_wr.set_xlabel("Battle", fontsize=8)
        ax_wr.set_ylabel("Win Rate", fontsize=8)
        ax_wr.set_ylim(-0.05, 1.05)
        ax_wr.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--")
        ax_wr.tick_params(labelsize=7)

        # Plot training win rate from dedicated history (updated every 50 battles)
        if train_wr_history:
            twr_battles = [r[0] for r in train_wr_history]
            twr_values = [r[1] for r in train_wr_history]
            ax_wr.plot(twr_battles, twr_values, color="#1f77b4", linewidth=1.5, label="Train WR")

        # Shade plateau regions from detector
        if detector is not None:
            bc_hist = detector._battle_counts
            wr_hist = detector._win_rates
            if wr_hist:
                regions = list(detector._plateau_regions)
                if detector._in_plateau_since is not None:
                    regions.append((detector._in_plateau_since, len(wr_hist) - 1))
                for start_idx, end_idx in regions:
                    start_idx = max(0, min(start_idx, len(bc_hist) - 1))
                    end_idx = max(0, min(end_idx, len(bc_hist) - 1))
                    ax_wr.axvspan(
                        bc_hist[start_idx], bc_hist[end_idx],
                        alpha=0.25, color="#d62728", zorder=0,
                    )

            # Plateau status label
            is_plateau = detector._streak >= detector.patience
            if is_plateau:
                self.train_plateau_var.set("PLATEAU DETECTED")
                self._plateau_label.configure(foreground="red")
            elif len(wr_hist) < detector.window:
                self.train_plateau_var.set(f"Collecting data ({len(wr_hist)}/{detector.window})")
                self._plateau_label.configure(foreground="gray")
            else:
                self.train_plateau_var.set("Learning")
                self._plateau_label.configure(foreground="green")

        # Overlay greedy eval win rate
        if greedy_eval_results and len(greedy_eval_results) > 0:
            ge_battles = [r[0] for r in greedy_eval_results]
            ge_wrs = [r[1] for r in greedy_eval_results]
            ax_wr.plot(ge_battles, ge_wrs, color="#ff7f0e", linewidth=1.5,
                       marker="o", markersize=3, label="Greedy WR")

        # Overlay per-team baseline eval win rates
        if baseline_eval_results_team1 and len(baseline_eval_results_team1) > 0:
            bl1_battles = [r[0] for r in baseline_eval_results_team1]
            bl1_wrs = [r[1] for r in baseline_eval_results_team1]
            ax_wr.plot(bl1_battles, bl1_wrs, color="#2ca02c", linewidth=1.5,
                       marker="s", markersize=3, label="Team1 vs BL")
        if baseline_eval_results_team2 and len(baseline_eval_results_team2) > 0:
            bl2_battles = [r[0] for r in baseline_eval_results_team2]
            bl2_wrs = [r[1] for r in baseline_eval_results_team2]
            ax_wr.plot(bl2_battles, bl2_wrs, color="#9467bd", linewidth=1.5,
                       marker="^", markersize=3, label="Team2 vs BL")

        handles = [
            plt.Line2D([0], [0], color="#1f77b4", linewidth=1.5, label="Train WR"),
            plt.Line2D([0], [0], color="#ff7f0e", linewidth=1.5, marker="o",
                       markersize=3, label="Greedy WR"),
            plt.Line2D([0], [0], color="#2ca02c", linewidth=1.5, marker="s",
                       markersize=3, label="Team1 vs BL"),
            plt.Line2D([0], [0], color="#9467bd", linewidth=1.5, marker="^",
                       markersize=3, label="Team2 vs BL"),
            Patch(facecolor="#d62728", alpha=0.25, label="Plateau"),
        ]
        ax_wr.legend(handles=handles, fontsize=7, loc="upper left")

        # --- Remaining charts require metrics_history ---
        if not metrics_history:
            self._fig.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.0)
            self._canvas.draw_idle()
            return

        battles = [m['battle_count'] for m in metrics_history]

        # --- Top-right: Policy Loss & Value Loss ---
        ax_loss.clear()
        ax_loss.set_title("Losses", fontsize=9, fontweight="bold")
        ax_loss.set_xlabel("Battle", fontsize=8)
        ax_loss.set_ylabel("Loss", fontsize=8)
        ax_loss.tick_params(labelsize=7)
        ax_loss.plot(battles, [m['policy_loss'] for m in metrics_history],
                     color="#d62728", linewidth=1.2, label="Policy Loss")
        ax_loss.plot(battles, [m['value_loss'] for m in metrics_history],
                     color="#9467bd", linewidth=1.2, label="Value Loss")
        ax_loss.legend(fontsize=7, loc="upper right")

        # --- Bottom-left: Entropy ---
        ax_entropy.clear()
        ax_entropy.set_title("Policy Entropy", fontsize=9, fontweight="bold")
        ax_entropy.set_xlabel("Battle", fontsize=8)
        ax_entropy.set_ylabel("Entropy", fontsize=8)
        ax_entropy.tick_params(labelsize=7)
        ax_entropy.plot(battles, [m['entropy'] for m in metrics_history],
                        color="#2ca02c", linewidth=1.2)

        # --- Bottom-right: Explained Variance & Episode Return ---
        # Reuse twin axis to avoid matplotlib memory leak from repeated twinx()
        if not hasattr(self, '_ax_return_twin'):
            self._ax_return_twin = ax_ev.twinx()
        else:
            self._ax_return_twin.clear()
        ax_ev.clear()
        ax_ev.set_title("Value Quality", fontsize=9, fontweight="bold")
        ax_ev.set_xlabel("Battle", fontsize=8)
        ax_ev.tick_params(labelsize=7)

        color_ev = "#1f77b4"
        ax_ev.set_ylabel("Explained Var.", fontsize=8, color=color_ev)
        ax_ev.plot(battles, [m['explained_variance'] for m in metrics_history],
                   color=color_ev, linewidth=1.2, label="Expl. Var.")
        ax_ev.tick_params(axis='y', labelcolor=color_ev, labelsize=7)
        ax_ev.set_ylim(-0.1, 1.1)

        # Secondary y-axis for mean episode return
        color_ret = "#ff7f0e"
        self._ax_return_twin.set_ylabel("Ep. Return", fontsize=8, color=color_ret)
        self._ax_return_twin.plot(battles, [m['mean_episode_return'] for m in metrics_history],
                                  color=color_ret, linewidth=1.2, label="Ep. Return")
        self._ax_return_twin.tick_params(axis='y', labelcolor=color_ret, labelsize=7)

        # Combined legend
        lines_ev = ax_ev.get_lines() + self._ax_return_twin.get_lines()
        labels_ev = [l.get_label() for l in lines_ev]
        ax_ev.legend(lines_ev, labels_ev, fontsize=7, loc="upper left")

        self._fig.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.0)
        self._canvas.draw_idle()

    def _make_config(self, **overrides) -> Config:
        """Build a Config from current GUI state."""
        kwargs = dict(
            team1_path=self.team1_var.get(),
            team2_path=self.team2_var.get(),
            battle_format=self.format_var.get(),
            total_battles=self.battles_var.get(),
            lr=self.lr_var.get(),
            hidden_size=self.hidden_var.get(),
            device=self.device_var.get(),
            checkpoint_dir=self.output_var.get(),
            server_port=self.server_port_var.get(),
            num_parallel_battles=self.concurrent_battles_var.get(),
        )
        ckpt = self.checkpoint_var.get().strip()
        if ckpt:
            kwargs["resume"] = True
            kwargs["resume_path"] = ckpt
            if self.fresh_start_var.get():
                kwargs["fresh_start"] = True
        kwargs.update(overrides)
        return Config(**kwargs)

    # ----- Server control ---------------------------------------------------

    def _on_start_server(self):
        sd_path = self.showdown_path_var.get().strip()
        if not sd_path:
            messagebox.showerror("Error", "Please select the Pokemon Showdown directory first.")
            return

        ps_main = Path(sd_path) / "pokemon-showdown"
        if not ps_main.exists():
            # Try the index.js fallback
            ps_main = Path(sd_path) / "index.js"
            if not ps_main.exists():
                messagebox.showerror("Error", f"Cannot find pokemon-showdown executable in {sd_path}")
                return

        port = self.server_port_var.get()
        node = shutil.which("node")
        if not node:
            messagebox.showerror("Error", "Node.js not found on PATH. Install Node.js first.")
            return

        try:
            self.btn_start_server.config(state="disabled")
            # Build and start in a background thread so the GUI stays responsive.
            threading.Thread(
                target=self._build_and_start_server,
                args=(node, str(ps_main), sd_path, port),
                daemon=True,
            ).start()
        except Exception as e:
            self.btn_start_server.config(state="normal")
            messagebox.showerror("Error", f"Failed to start server: {e}")

    def _build_and_start_server(self, node: str, ps_main: str, sd_path: str, port: int):
        """Build Showdown (if needed) then start the server.

        Separating build from start prevents esbuild's Go runtime from
        competing for memory with the running Node.js server process.
        """
        # Check whether a build is needed (dist/ is Showdown's esbuild output)
        dist_dir = Path(sd_path) / "dist"
        needs_build = not dist_dir.exists()

        if needs_build:
            self._log("Building Pokemon Showdown (first run)...")
            build_script = Path(sd_path) / "build"
            build_result = subprocess.run(
                [node, str(build_script)],
                cwd=sd_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if build_result.returncode != 0:
                self._log(f"[showdown] Build output:\n{build_result.stdout}")
                self._log("ERROR: Showdown build failed. Try running 'node build' in the pokemon-showdown directory manually.")
                self.after(0, lambda: self.btn_start_server.config(state="normal"))
                return
            self._log("Build completed successfully.")

        # Ensure Showdown config disables subprocesses to reduce memory usage.
        # PokeRL only runs 1 battle at a time, so forking worker processes for
        # battle simulation is unnecessary and causes OOM crashes on low-memory machines.
        config_dir = Path(sd_path) / "config"
        config_dir.mkdir(exist_ok=True)
        config_js = config_dir / "config.js"
        if not config_js.exists():
            config_js.write_text(
                "// Auto-generated by PokeRL for low-memory training.\n"
                "// Disable subprocesses to reduce memory usage.\n"
                "exports.pokemon = {};\n"
                "exports.subprocesses = 0;\n"
            )
            self._log("Created Showdown config with subprocesses disabled.")

        # Kill any leftover process holding the port (e.g. orphan from a previous session).
        self._free_port(port)

        # Start with --skip-build so the server process doesn't re-invoke esbuild.
        cmd = [node, ps_main, "start", "--no-security", "--skip-build", f"--port={port}"]
        popen_kwargs = dict(
            cwd=sd_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True
        self._showdown_proc = subprocess.Popen(cmd, **popen_kwargs)
        self._log(f"Starting Showdown server (PID {self._showdown_proc.pid}) on port {port}...")
        self.after(0, self._server_started)

        # Stream server output
        self._stream_server_output()

    def _server_started(self):
        if self._showdown_proc:
            self.server_status_var.set(f"Server: running (PID {self._showdown_proc.pid})")
            self.btn_stop_server.config(state="normal")

    def _stream_server_output(self):
        proc = self._showdown_proc
        if proc and proc.stdout:
            for line in proc.stdout:
                self._log(f"[showdown] {line.rstrip()}")
            proc.wait()
            self._log(f"Showdown server exited (code {proc.returncode})")
            self.after(0, self._server_stopped)

    def _server_stopped(self):
        self.server_status_var.set("Server: stopped")
        self.btn_start_server.config(state="normal")
        self.btn_stop_server.config(state="disabled")
        self._showdown_proc = None

    def _on_stop_server(self):
        if self._showdown_proc:
            self._log("Stopping Showdown server...")
            self.btn_stop_server.config(state="disabled")
            threading.Thread(target=self._cleanup_server, daemon=True).start()

    def _free_port(self, port: int):
        """Kill any process currently listening on *port*."""
        if os.name == "nt":
            # netstat -ano produces lines like:
            #   TCP    0.0.0.0:8000    0.0.0.0:0    LISTENING    12345
            try:
                out = subprocess.check_output(
                    ["netstat", "-ano", "-p", "TCP"],
                    text=True, stderr=subprocess.DEVNULL,
                )
            except Exception:
                return
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[3] == "LISTENING":
                    addr = parts[1]
                    if addr.endswith(f":{port}"):
                        pid = int(parts[4])
                        self._log(f"Killing leftover process on port {port} (PID {pid})...")
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(pid)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
        else:
            # Unix: lsof -ti :port returns PIDs
            try:
                out = subprocess.check_output(
                    ["lsof", "-ti", f":{port}"],
                    text=True, stderr=subprocess.DEVNULL,
                )
            except Exception:
                return
            for pid_str in out.split():
                pid = int(pid_str)
                self._log(f"Killing leftover process on port {port} (PID {pid})...")
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

    def _cleanup_server(self):
        """Terminate the showdown server and all its child processes."""
        proc = self._showdown_proc
        if proc is None or proc.poll() is not None:
            return
        if os.name == "nt":
            # Windows: taskkill /T kills the entire process tree
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc.wait()
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
                proc.wait()

    # ----- Training control -------------------------------------------------

    def _on_start_training(self):
        # Validate inputs
        t1 = self.team1_var.get().strip()
        t2 = self.team2_var.get().strip()
        if not t1 or not os.path.isfile(t1):
            messagebox.showerror("Error", f"Team 1 file not found: {t1}")
            return
        if not t2 or not os.path.isfile(t2):
            messagebox.showerror("Error", f"Team 2 file not found: {t2}")
            return

        config = self._make_config()
        self._training_stop.clear()
        self.btn_train.config(state="disabled")
        self.btn_stop_train.config(state="normal")
        self.train_status_var.set("Training...")

        self._log(f"Starting training: {config.total_battles} battles, format={config.battle_format}")
        self._training_start_time = time.time()
        self.train_progress["value"] = 0
        self.train_battles_var.set(f"0/{config.total_battles} battles")
        self.train_elapsed_var.set("Elapsed: 0s")
        self.train_eta_var.set("ETA: --")

        def _progress_cb(battle_count, total_battles, plateau_detector=None,
                         metrics_history=None, greedy_eval_results=None,
                         train_wr_history=None,
                         baseline_eval_results_team1=None,
                         baseline_eval_results_team2=None):
            self.after(0, lambda bc=battle_count, tb=total_battles, pd=plateau_detector,
                              mh=metrics_history, ge=greedy_eval_results,
                              twh=train_wr_history,
                              bt1=baseline_eval_results_team1,
                              bt2=baseline_eval_results_team2:
                       self._on_progress_update(bc, tb, pd, mh, ge, twh, bt1, bt2))

        def train_thread():
            try:
                trainer = Trainer(config, progress_callback=_progress_cb)
                self._trainer = trainer

                # Monkey-patch the trainer's train loop to check the stop flag
                original_run_battle = trainer._run_battle_batch

                async def stoppable_run_battle(n_battles):
                    if self._training_stop.is_set():
                        raise _TrainingStoppedError()
                    return await original_run_battle(n_battles)

                trainer._run_battle_batch = stoppable_run_battle

                future = run_async(trainer.train())
                future.result()  # blocks until done or error
                self._log("Training finished.")
            except _TrainingStoppedError:
                self._log("Training stopped by user.")
            except Exception as e:
                self._log(f"Training error: {e}")
            finally:
                self._trainer = None
                self.after(0, self._training_finished)

        threading.Thread(target=train_thread, daemon=True).start()

    def _on_stop_training(self):
        self._log("Requesting training stop...")
        self._training_stop.set()
        self.train_status_var.set("Stopping...")

    def _training_finished(self):
        self.btn_train.config(state="normal")
        self.btn_stop_train.config(state="disabled")
        self.train_status_var.set("Idle")
        self._training_start_time = None
        self.train_battles_var.set("")
        self.train_elapsed_var.set("Elapsed: --")
        self.train_eta_var.set("ETA: --")
        self.train_plateau_var.set("")
        self.stat_greedy_wr_var.set("Greedy WR: --")
        self.stat_train_wr_var.set("Train WR: --")
        self.stat_policy_loss_var.set("Policy Loss: --")
        self.stat_value_loss_var.set("Value Loss: --")
        self.stat_entropy_var.set("Entropy: --")
        self.stat_expl_var_var.set("Expl. Var: --")
        self.stat_ep_return_var.set("Ep. Return: --")

    # ----- Evaluation control -----------------------------------------------

    def _on_run_eval(self):
        test_models = list(self.test_listbox.get(0, "end"))
        opp_models = list(self.opp_listbox.get(0, "end"))

        if not test_models:
            messagebox.showerror("Error", "Add at least one test model.")
            return
        if not opp_models:
            messagebox.showerror("Error", "Add at least one opponent model.")
            return

        t1 = self.team1_var.get().strip()
        t2 = self.team2_var.get().strip()
        if not t1 or not os.path.isfile(t1):
            messagebox.showerror("Error", f"Team 1 file not found: {t1}")
            return
        if not t2 or not os.path.isfile(t2):
            messagebox.showerror("Error", f"Team 2 file not found: {t2}")
            return

        n_battles = self.eval_n_battles_var.get()
        config = self._make_config()

        self.btn_eval.config(state="disabled")
        self.btn_stop_eval.config(state="normal")
        self.eval_status_var.set("Running evaluation...")
        self._training_stop.clear()

        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        def eval_thread():
            try:
                team1_str = load_team(t1)
                team2_str = load_team(t2)

                for t_path in test_models:
                    if self._training_stop.is_set():
                        break

                    t_name = Path(t_path).stem
                    t_state = torch.load(t_path, map_location="cpu", weights_only=False)

                    for o_path in opp_models:
                        if self._training_stop.is_set():
                            break

                        o_name = Path(o_path).stem
                        o_state = torch.load(o_path, map_location="cpu", weights_only=False)

                        self._log(f"Evaluating {t_name} vs {o_name} ({n_battles} battles)...")
                        self.after(0, lambda: self.eval_status_var.set(f"{t_name} vs {o_name}..."))

                        wins, losses = self._run_eval_matchup(
                            config, team1_str, team2_str,
                            t_state, o_state, n_battles
                        )
                        wr = wins / max(wins + losses, 1)
                        self._log(f"  Result: {wins}W / {losses}L ({wr:.1%})")

                        # Insert into treeview on main thread
                        self.after(0, lambda tn=t_name, on=o_name, w=wins, l=losses, r=wr:
                            self.results_tree.insert("", "end", values=(tn, on, w, l, f"{r:.1%}"))
                        )

                self._log("Evaluation complete.")
            except Exception as e:
                self._log(f"Evaluation error: {e}")
            finally:
                self.after(0, self._eval_finished)

        threading.Thread(target=eval_thread, daemon=True).start()

    def _run_eval_matchup(
        self, config: Config, team1_str: str, team2_str: str,
        test_state: dict, opp_state: dict, n_battles: int
    ) -> Tuple[int, int]:
        """Run n_battles between two checkpoint states. Returns (wins, losses)."""
        total_wins = 0
        total_losses = 0

        for i in range(n_battles):
            if self._training_stop.is_set():
                break

            test_agent = PPOAgent(config, agent_id="eval_test")
            test_agent.load_state_dict(test_state["agent1"])
            test_agent.set_eval()

            opp_agent = PPOAgent(config, agent_id="eval_opp")
            opp_agent.load_state_dict(opp_state["agent2"])
            opp_agent.set_eval()

            async def run_one():
                p1 = create_player(
                    agent=test_agent, config=config, team_str=team1_str,
                    collect_data=False, deterministic=False,
                )
                p2 = create_player(
                    agent=opp_agent, config=config, team_str=team2_str,
                    collect_data=False, deterministic=False,
                )
                await p1.battle_against(p2, n_battles=1)
                return p1.n_won_battles > 0

            try:
                won = run_async(run_one()).result(timeout=120)
                if won:
                    total_wins += 1
                else:
                    total_losses += 1
            except Exception as e:
                self._log(f"  Battle {i+1} error: {e}")
                total_losses += 1

        return total_wins, total_losses

    def _on_stop_eval(self):
        self._log("Requesting evaluation stop...")
        self._training_stop.set()
        self.eval_status_var.set("Stopping...")

    def _eval_finished(self):
        self.btn_eval.config(state="normal")
        self.btn_stop_eval.config(state="disabled")
        self.eval_status_var.set("Idle")

    # ----- Challenge control ------------------------------------------------

    def _on_send_challenge(self):
        ckpt_path = self.challenge_checkpoint_var.get().strip()
        team_path = self.challenge_team_var.get().strip()
        opponent = self.challenge_opponent_var.get().strip()
        username = self.challenge_username_var.get().strip()

        if not ckpt_path or not os.path.isfile(ckpt_path):
            messagebox.showerror("Error", f"Checkpoint not found: {ckpt_path}")
            return
        if not team_path or not os.path.isfile(team_path):
            messagebox.showerror("Error", f"Team file not found: {team_path}")
            return
        if not opponent:
            messagebox.showerror("Error", "Please enter an opponent username.")
            return
        if not username:
            messagebox.showerror("Error", "Please enter your username.")
            return

        server_type = self.challenge_server_var.get()
        password = self.challenge_password_var.get().strip()
        if server_type == "live" and not password:
            messagebox.showerror("Error", "Password is required for the live server.")
            return

        battle_format = self.challenge_format_var.get().strip()
        n_challenges = self.challenge_n_var.get()

        self._challenge_stop.clear()
        self.btn_challenge.config(state="disabled")
        self.btn_stop_challenge.config(state="normal")
        self.challenge_status_var.set("Connecting...")

        # Clear previous results
        for item in self.challenge_tree.get_children():
            self.challenge_tree.delete(item)

        def challenge_thread():
            try:
                # Build server configuration
                if server_type == "live":
                    server_cfg = ShowdownServerConfiguration
                else:
                    port = self.challenge_local_port_var.get()
                    server_cfg = ServerConfiguration(
                        f"ws://localhost:{port}/showdown/websocket",
                        f"http://localhost:{port}/action.php?",
                    )

                # Load team
                team_str = load_team(team_path)

                # Load checkpoint and create agent
                config = Config(
                    battle_format=battle_format,
                    device="cpu",
                )
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                agent = PPOAgent(config, agent_id="challenger")
                # Load agent weights (inference-only, no optimizer state needed)
                if "agent" in state:
                    # Per-team checkpoint (best_team1.pt / best_team2.pt)
                    agent.load_weights_only(state["agent"])
                elif "agent1" in state or "agent2" in state:
                    # Full checkpoint — use team selector
                    team_key = self.challenge_team_select_var.get()
                    if team_key not in state:
                        # Fall back to whichever key exists
                        team_key = "agent1" if "agent1" in state else "agent2"
                    agent.load_weights_only(state[team_key])
                else:
                    raise ValueError(
                        "Checkpoint does not contain agent, agent1, or agent2 keys."
                    )
                agent.set_eval()

                acct = AccountConfiguration(username, password or None)

                wins = 0
                losses = 0

                for i in range(n_challenges):
                    if self._challenge_stop.is_set():
                        break

                    self.after(0, lambda idx=i+1: self.challenge_status_var.set(
                        f"Battle {idx}/{n_challenges}..."
                    ))

                    async def run_challenge():
                        player = RLPlayer(
                            agent=agent,
                            config=config,
                            collect_data=False,
                            deterministic=False,
                            account_configuration=acct,
                            battle_format=battle_format,
                            team=ConstantTeambuilder(team_str),
                            server_configuration=server_cfg,
                            max_concurrent_battles=1,
                        )
                        await player.send_challenges(opponent, n_challenges=1)
                        won = player.n_won_battles > 0
                        return won

                    try:
                        won = run_async(run_challenge()).result(timeout=300)
                        if won:
                            wins += 1
                            result_text = "Win"
                        else:
                            losses += 1
                            result_text = "Loss"
                    except Exception as e:
                        losses += 1
                        result_text = f"Error: {e}"

                    self._log(f"Challenge {i+1}/{n_challenges}: {result_text}")
                    self.after(0, lambda idx=i+1, r=result_text:
                        self.challenge_tree.insert("", "end", values=(idx, r))
                    )

                self._log(f"Challenges complete: {wins}W / {losses}L")
            except Exception as e:
                self._log(f"Challenge error: {e}")
            finally:
                self.after(0, self._challenge_finished)

        threading.Thread(target=challenge_thread, daemon=True).start()

    def _on_stop_challenge(self):
        self._log("Requesting challenge stop...")
        self._challenge_stop.set()
        self.challenge_status_var.set("Stopping...")

    def _challenge_finished(self):
        self.btn_challenge.config(state="normal")
        self.btn_stop_challenge.config(state="disabled")
        self.challenge_status_var.set("Idle")

    # ----- Cleanup ----------------------------------------------------------

    def destroy(self):
        self._cleanup_server()
        if _loop and _loop.is_running():
            _loop.call_soon_threadsafe(_loop.stop)
        super().destroy()


class _TrainingStoppedError(Exception):
    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = PokeRLApp()
    app.protocol("WM_DELETE_WINDOW", app.destroy)
    app.mainloop()


if __name__ == "__main__":
    main()
