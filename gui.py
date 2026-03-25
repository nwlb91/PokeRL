#!/usr/bin/env python3
"""PokeRL GUI — Manage training, evaluation, and Showdown server."""

import asyncio
import json
import logging
import os
import platform
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch

import torch

from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ServerConfiguration,
    ShowdownServerConfiguration,
)
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from pokerl.agent import PPOAgent
from pokerl.checkpoint import CheckpointManager
from pokerl.config import Config
from pokerl.env import RLPlayer, create_player, load_team
from pokerl.league import League
from pokerl.plateau import PlateauDetector
from pokerl.trainer import Trainer
from pokerl.win_probability import WinProbabilityEstimator

logger = logging.getLogger("pokerl.gui")

# ---------------------------------------------------------------------------
# Async bridge — run coroutines from tkinter callbacks
# ---------------------------------------------------------------------------

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None


def _ensure_event_loop():
    global _loop, _loop_thread
    if _loop is not None and _loop.is_running():
        return
    _loop = asyncio.new_event_loop()
    _loop_thread = threading.Thread(target=_loop.run_forever, daemon=True)
    _loop_thread.start()


def run_async(coro):
    """Schedule *coro* on the background event loop, return a Future."""
    _ensure_event_loop()
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

        # Tkinter variables
        self.team1_var = tk.StringVar(value="teams/team1.txt")
        self.team2_var = tk.StringVar(value="teams/team2.txt")
        self.checkpoint_var = tk.StringVar(value="")
        self.output_var = tk.StringVar(value="checkpoints")
        self.format_var = tk.StringVar(value="gen9nationaldexmonotype")
        self.battles_var = tk.IntVar(value=100000)
        self.lr_var = tk.DoubleVar(value=3e-4)
        self.hidden_var = tk.IntVar(value=256)
        self.device_var = tk.StringVar(value="cpu")
        self.server_port_var = tk.IntVar(value=8000)
        self.showdown_path_var = tk.StringVar(value="pokemon-showdown")
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

        # --- Win-rate chart with plateau visualisation ---
        chart_frame = ttk.LabelFrame(parent, text="Win Rate & Plateau Detection")
        chart_frame.pack(fill="both", expand=True, padx=6, pady=4)

        self._fig, self._ax = plt.subplots(figsize=(7, 2.4), dpi=90)
        self._fig.patch.set_facecolor("#f0f0f0")
        self._ax.set_xlabel("Battle")
        self._ax.set_ylabel("Win Rate")
        self._ax.set_ylim(-0.05, 1.05)
        self._ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--")
        self._fig.tight_layout(pad=1.5)

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
                            plateau_detector: "PlateauDetector | None" = None):
        """Update progress bar, time labels, and plateau chart."""
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

        # Update the plateau chart
        if plateau_detector is not None:
            self._update_plateau_chart(plateau_detector)

    def _update_plateau_chart(self, detector: "PlateauDetector"):
        """Redraw the win-rate chart with plateau shading."""
        wr_hist = detector._win_rates
        bc_hist = detector._battle_counts
        if not wr_hist:
            return

        ax = self._ax
        ax.clear()

        # Style
        ax.set_xlabel("Battle", fontsize=9)
        ax.set_ylabel("Win Rate", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--")
        ax.tick_params(labelsize=8)

        # Win rate line
        ax.plot(bc_hist, wr_hist, color="#1f77b4", linewidth=1.5, label="Win Rate")

        # Shade plateau regions in red
        regions = list(detector._plateau_regions)
        if detector._in_plateau_since is not None:
            regions.append((detector._in_plateau_since, len(wr_hist) - 1))

        for start_idx, end_idx in regions:
            start_idx = max(0, min(start_idx, len(bc_hist) - 1))
            end_idx = max(0, min(end_idx, len(bc_hist) - 1))
            ax.axvspan(
                bc_hist[start_idx], bc_hist[end_idx],
                alpha=0.25, color="#d62728", zorder=0,
            )

        # Legend
        handles = [
            plt.Line2D([0], [0], color="#1f77b4", linewidth=1.5, label="Win Rate"),
            Patch(facecolor="#d62728", alpha=0.25, label="Plateau"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="upper left")

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

        self._fig.tight_layout(pad=1.5)
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
        )
        ckpt = self.checkpoint_var.get().strip()
        if ckpt:
            kwargs["resume"] = True
            kwargs["resume_path"] = ckpt
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
            cmd = [node, str(ps_main), "start", "--no-security", f"--port={port}"]
            self._showdown_proc = subprocess.Popen(
                cmd,
                cwd=sd_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self._log(f"Starting Showdown server (PID {self._showdown_proc.pid}) on port {port}...")
            self.server_status_var.set(f"Server: running (PID {self._showdown_proc.pid})")
            self.btn_start_server.config(state="disabled")
            self.btn_stop_server.config(state="normal")

            # Stream server output in background
            threading.Thread(target=self._stream_server_output, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start server: {e}")

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
            self._showdown_proc.terminate()
            threading.Thread(target=self._wait_kill_server, daemon=True).start()

    def _wait_kill_server(self):
        proc = self._showdown_proc
        if proc:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
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

        def _progress_cb(battle_count, total_battles, plateau_detector=None):
            self.after(0, lambda bc=battle_count, tb=total_battles, pd=plateau_detector:
                       self._on_progress_update(bc, tb, pd))

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
                # Try agent1 key first, fall back to agent2
                if "agent1" in state:
                    agent.load_state_dict(state["agent1"])
                elif "agent2" in state:
                    agent.load_state_dict(state["agent2"])
                else:
                    raise ValueError("Checkpoint does not contain agent1 or agent2 keys.")
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
        if self._showdown_proc:
            self._showdown_proc.terminate()
            try:
                self._showdown_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._showdown_proc.kill()
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
