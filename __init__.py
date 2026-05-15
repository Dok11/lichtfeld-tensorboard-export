"""Export full LichtFeld training metrics for external analysis."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import lichtfeld as lf
from lfs_plugins.props import BoolProperty, IntProperty, PropSubtype, PropertyGroup, StringProperty
from lfs_plugins.ui.state import AppState

try:
    from tensorboardX import SummaryWriter
except Exception as error:  # pragma: no cover - depends on plugin-local venv state
    SummaryWriter = None
    SUMMARY_WRITER_IMPORT_ERROR = error
else:
    SUMMARY_WRITER_IMPORT_ERROR = None


__version__ = "0.1.0"
CONFIG_SCHEMA_VERSION = 1
DEFAULT_LOG_DIR = str(Path.home() / "LichtFeldTensorBoardRuns")
CSV_FIELDS = [
    "wall_time",
    "iteration",
    "max_iterations",
    "progress",
    "loss_raw",
    "loss_mean_since_last_write",
    "loss_min_since_last_write",
    "loss_max_since_last_write",
    "loss_ema_100",
    "loss_ema_500",
    "num_gaussians",
    "iters_per_second",
    "elapsed_seconds",
    "eta_seconds",
    "phase",
    "strategy",
]


class TensorBoardExportSettings(PropertyGroup):
    enabled = BoolProperty(default=False, name="Enabled")
    csv_enabled = BoolProperty(default=True, name="Write CSV")
    every_n_steps = IntProperty(default=500, min=1, max=100000, name="Every N steps")
    log_dir = StringProperty(default=DEFAULT_LOG_DIR, subtype=PropSubtype.DIR_PATH, name="Log dir")
    run_name = StringProperty(default="", name="Run name")


class TensorBoardExportPanel(lf.ui.Panel):
    id = "tensorboard_export.panel"
    label = "TensorBoard Export"
    parent = "lfs.training"
    order = 900

    def __init__(self) -> None:
        self.settings = TensorBoardExportSettings.get_instance()

    @classmethod
    def poll(cls, context) -> bool:
        del context
        return AppState.has_trainer.value or lf.has_trainer()

    def draw(self, ui) -> None:
        ui.heading("TensorBoard Export")

        if SUMMARY_WRITER_IMPORT_ERROR is not None:
            ui.text_colored(
                f"tensorboardX import failed: {SUMMARY_WRITER_IMPORT_ERROR}",
                (1.0, 0.35, 0.35, 1.0),
            )

        ui.prop(self.settings, "enabled")
        ui.prop(self.settings, "csv_enabled")
        changed, every_n_steps = ui.input_int("Every N steps", int(self.settings.every_n_steps))
        if changed:
            self.settings.every_n_steps = max(1, min(100000, int(every_n_steps)))
        ui.prop(self.settings, "log_dir")
        ui.prop(self.settings, "run_name")

        ui.separator()
        if _logger is not None:
            ui.label(f"Writer: {_logger.status}")
            ui.label(f"Rows written: {_logger.rows_written}")
            ui.label(f"Last logged iteration: {_logger.last_logged_iteration if _logger.last_logged_iteration > 0 else '-'}")
            ui.label(f"Last write: {_logger.last_write_label}")
            if _logger.last_error:
                ui.text_colored(f"Last error: {_logger.last_error}", (1.0, 0.35, 0.35, 1.0))

        ui.separator()
        ui.label(f"State: {AppState.trainer_state.value}")
        ui.label(f"Iteration: {AppState.iteration.value:,} / {AppState.max_iterations.value:,}")
        ui.label(f"Loss: {AppState.loss.value:.6f}")
        ui.label(f"Gaussians: {AppState.num_gaussians.value:,}")

        if _logger is not None and _logger.run_dir is not None:
            ui.separator()
            ui.text_wrapped(f"Run dir: {_logger.run_dir}")

        ui.separator()
        if ui.button("Rotate run now", (-1, 0)):
            _start_run(force=True, reason="manual")
        if ui.button("Flush and close run", (-1, 0)):
            _close_run(reason="manual")
        if ui.button("Open run folder", (-1, 0)):
            _open_run_folder()
        if ui.button("Copy TensorBoard command", (-1, 0)):
            _copy_tensorboard_command()
        if ui.button("Open TensorBoard", (-1, 0)):
            lf.ui.open_url("http://localhost:6006/?darkMode=true")


class MetricsLogger:
    def __init__(self) -> None:
        self.settings = TensorBoardExportSettings.get_instance()
        self.writer: Any | None = None
        self.run_dir: Path | None = None
        self.csv_file: Any | None = None
        self.csv_writer: csv.DictWriter | None = None
        self.last_logged_iteration = -1
        self.last_rate_iteration: int | None = None
        self.last_rate_time: float | None = None
        self.config_logged = False
        self.rows_written = 0
        self.last_write_time: float | None = None
        self.last_error = ""
        self.status = "closed"
        self.loss_window_count = 0
        self.loss_window_sum = 0.0
        self.loss_window_min: float | None = None
        self.loss_window_max: float | None = None
        self.loss_ema_100: float | None = None
        self.loss_ema_500: float | None = None

    @property
    def last_write_label(self) -> str:
        if self.last_write_time is None:
            return "-"
        return time.strftime("%H:%M:%S", time.localtime(self.last_write_time))

    def close(self, reason: str = "") -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()

        self.writer = None
        self.csv_file = None
        self.csv_writer = None
        self.last_rate_iteration = None
        self.last_rate_time = None
        self.status = "closed" if not reason else f"closed ({reason})"
        self._reset_loss_window()

    def start(self, force: bool = False, reason: str = "") -> None:
        if not self.settings.enabled:
            return
        if (self.writer is not None or self.csv_writer is not None) and not force:
            return

        self.close(reason="rotated" if force else "")
        self.last_logged_iteration = -1
        self.rows_written = 0
        self.last_write_time = None
        self.last_error = ""
        self.config_logged = False
        self.loss_ema_100 = None
        self.loss_ema_500 = None
        self._reset_loss_window()

        root = Path(self.settings.log_dir).expanduser()
        run_name = self.settings.run_name.strip() or _default_run_name()
        self.run_dir = _unique_run_dir(root / _sanitize_path_part(run_name))
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if SummaryWriter is not None:
            self.writer = SummaryWriter(logdir=str(self.run_dir))
            self.status = "active"
        else:
            lf.log.warn(f"tensorboardX is not available; writing CSV only: {SUMMARY_WRITER_IMPORT_ERROR}")
            self.status = "csv-only"

        if self.settings.csv_enabled:
            self.csv_file = (self.run_dir / "metrics.csv").open("a", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS)
            if self.csv_file.tell() == 0:
                self.csv_writer.writeheader()

        if self.writer is not None and self.csv_writer is not None:
            self.status = "active (TensorBoard + CSV)"
        elif self.writer is not None:
            self.status = "active (TensorBoard)"
        elif self.csv_writer is not None:
            self.status = "csv-only"
        else:
            self.status = "inactive"

        self._write_config_snapshot()
        suffix = f" ({reason})" if reason else ""
        lf.log.info(f"TensorBoard export run started{suffix}: {self.run_dir}")

    def log_step(self) -> None:
        if not self.settings.enabled:
            return

        ctx = lf.context()
        iteration = int(ctx.iteration)
        if iteration <= 0 or iteration == self.last_logged_iteration:
            return

        loss = float(ctx.loss)
        self._observe_loss(loss)

        every_n_steps = max(1, int(self.settings.every_n_steps))
        if iteration % every_n_steps != 0:
            return

        self.start()
        if self.writer is None and self.csv_writer is None:
            return

        max_iterations = int(ctx.max_iterations)
        if max_iterations <= 0:
            max_iterations = int(AppState.max_iterations.value)
        loss_stats = self._loss_window_stats(loss)
        num_gaussians = int(ctx.num_gaussians)
        progress = iteration / max_iterations if max_iterations > 0 else None
        iters_per_second = self._calculate_rate(iteration)
        elapsed_seconds = _safe_float_call(lf.trainer_elapsed_seconds)
        eta_seconds = _safe_float_call(lf.trainer_eta_seconds)
        phase = str(getattr(ctx, "phase", ""))
        strategy = str(getattr(ctx, "strategy", ""))

        if self.writer is not None:
            self.writer.add_scalar("train/loss_raw", loss, iteration)
            self.writer.add_scalar(
                "train/loss_mean_since_last_write",
                loss_stats["mean"],
                iteration,
            )
            self.writer.add_scalar("train/loss_min_since_last_write", loss_stats["min"], iteration)
            self.writer.add_scalar("train/loss_max_since_last_write", loss_stats["max"], iteration)
            self.writer.add_scalar("train/num_gaussians", num_gaussians, iteration)

            if self.loss_ema_100 is not None:
                self.writer.add_scalar("train/loss_ema_100", self.loss_ema_100, iteration)
            if self.loss_ema_500 is not None:
                self.writer.add_scalar("train/loss_ema_500", self.loss_ema_500, iteration)
            if progress is not None:
                self.writer.add_scalar("train/progress", progress, iteration)
            if iters_per_second is not None:
                self.writer.add_scalar("performance/iters_per_second", iters_per_second, iteration)
            if elapsed_seconds is not None:
                self.writer.add_scalar("trainer/elapsed_seconds", elapsed_seconds, iteration)
            if eta_seconds is not None and eta_seconds >= 0:
                self.writer.add_scalar("trainer/eta_seconds", eta_seconds, iteration)

            if not self.config_logged:
                self._log_config_text(iteration)
                self.config_logged = True
            self.writer.flush()

        if self.csv_writer is not None and self.csv_file is not None:
            self.csv_writer.writerow(
                {
                    "wall_time": time.time(),
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "progress": progress,
                    "loss_raw": loss,
                    "loss_mean_since_last_write": loss_stats["mean"],
                    "loss_min_since_last_write": loss_stats["min"],
                    "loss_max_since_last_write": loss_stats["max"],
                    "loss_ema_100": self.loss_ema_100,
                    "loss_ema_500": self.loss_ema_500,
                    "num_gaussians": num_gaussians,
                    "iters_per_second": iters_per_second,
                    "elapsed_seconds": elapsed_seconds,
                    "eta_seconds": eta_seconds,
                    "phase": phase,
                    "strategy": strategy,
                }
            )
            self.csv_file.flush()

        self.last_logged_iteration = iteration
        self.rows_written += 1
        self.last_write_time = time.time()
        self._reset_loss_window()

    def _calculate_rate(self, iteration: int) -> float | None:
        now = time.monotonic()
        if self.last_rate_iteration is None or self.last_rate_time is None:
            self.last_rate_iteration = iteration
            self.last_rate_time = now
            return None

        iteration_delta = iteration - self.last_rate_iteration
        time_delta = now - self.last_rate_time
        self.last_rate_iteration = iteration
        self.last_rate_time = now

        if iteration_delta <= 0 or time_delta <= 0:
            return None
        return iteration_delta / time_delta

    def _observe_loss(self, loss: float) -> None:
        if not math.isfinite(loss):
            return

        self.loss_window_count += 1
        self.loss_window_sum += loss
        self.loss_window_min = loss if self.loss_window_min is None else min(self.loss_window_min, loss)
        self.loss_window_max = loss if self.loss_window_max is None else max(self.loss_window_max, loss)
        self.loss_ema_100 = _ema(self.loss_ema_100, loss, 100)
        self.loss_ema_500 = _ema(self.loss_ema_500, loss, 500)

    def _loss_window_stats(self, fallback: float) -> dict[str, float]:
        if self.loss_window_count <= 0:
            return {"mean": fallback, "min": fallback, "max": fallback}
        return {
            "mean": self.loss_window_sum / self.loss_window_count,
            "min": self.loss_window_min if self.loss_window_min is not None else fallback,
            "max": self.loss_window_max if self.loss_window_max is not None else fallback,
        }

    def _reset_loss_window(self) -> None:
        self.loss_window_count = 0
        self.loss_window_sum = 0.0
        self.loss_window_min = None
        self.loss_window_max = None

    def _log_config_text(self, iteration: int) -> None:
        if self.writer is None:
            return

        payload = _config_payload()
        self.writer.add_text(
            "config/session",
            "```json\n" + json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n```",
            iteration,
        )

    def _write_config_snapshot(self) -> None:
        if self.run_dir is None:
            return
        try:
            (self.run_dir / "config.json").write_text(
                json.dumps(_config_payload(), indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
        except Exception as error:
            self.last_error = str(error)
            lf.log.error(f"TensorBoard export config snapshot failed: {error}")


def _read_property_group(factory: Any) -> dict[str, Any]:
    try:
        params = factory()
        if params is None or not params.has_params():
            return {}
    except Exception:
        return {}

    values: dict[str, Any] = {}
    for item in _property_items(params):
        name = _property_name(item)
        if not name:
            continue
        try:
            values[name] = _to_jsonable(getattr(params, name))
        except Exception:
            try:
                values[name] = _to_jsonable(params.get(name))
            except Exception:
                pass
    return values


def _property_items(params: Any) -> list[Any]:
    try:
        descriptors = params.get_all_properties()
        if isinstance(descriptors, dict):
            return list(descriptors.keys())
    except Exception:
        pass
    try:
        return list(params.properties())
    except Exception:
        return []


def _property_name(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("id", "name", "identifier"):
            value = item.get(key)
            if value:
                return str(value)
    for key in ("id", "name", "identifier"):
        value = getattr(item, key, None)
        if value:
            return str(value)
    return ""


def _config_payload() -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "plugin_version": __version__,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "trainer_state": _safe_str_call(lf.trainer_state),
        "strategy_type": _safe_str_call(lf.trainer_strategy_type),
        "max_gaussians": _safe_int_call(lf.trainer_max_gaussians),
        "total_iterations": _safe_int_call(lf.trainer_total_iterations),
        "optimization_params": _read_property_group(lf.optimization_params),
        "dataset_params": _read_property_group(lf.dataset_params),
    }


def _default_run_name() -> str:
    parts = [time.strftime("%Y%m%d_%H%M%S")]
    dataset_name = _dataset_basename()
    strategy = _safe_str_call(lf.trainer_strategy_type)
    total_iterations = _safe_int_call(lf.trainer_total_iterations)

    if dataset_name:
        parts.append(dataset_name)
    if strategy:
        parts.append(strategy)
    if total_iterations and total_iterations > 0:
        parts.append(f"{total_iterations}iter")

    return "_".join(parts)


def _dataset_basename() -> str:
    try:
        params = lf.dataset_params()
        if params is None or not params.has_params():
            return ""
        data_path = str(getattr(params, "data_path", "") or "")
    except Exception:
        return ""
    if not data_path:
        return ""
    try:
        return Path(data_path).expanduser().resolve().name
    except Exception:
        return Path(data_path).name


def _unique_run_dir(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(2, 1000):
        candidate = path.with_name(f"{path.name}_{index:02d}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{path.name}_{int(time.time())}")


def _tensorboard_command() -> str:
    log_dir = str(Path(TensorBoardExportSettings.get_instance().log_dir).expanduser())
    return f'tensorboard --logdir "{log_dir}" --port 6006'


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "value"):
        return _to_jsonable(value.value)
    return str(value)


def _ema(previous: float | None, value: float, span: int) -> float:
    if previous is None:
        return value
    alpha = 2.0 / (span + 1.0)
    return alpha * value + (1.0 - alpha) * previous


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def _safe_int_call(callback: Any) -> int | None:
    try:
        return int(callback())
    except Exception:
        return None


def _safe_float_call(callback: Any) -> float | None:
    try:
        return _safe_float(callback())
    except Exception:
        return None


def _safe_str_call(callback: Any) -> str:
    try:
        return str(callback())
    except Exception:
        return ""


def _sanitize_path_part(value: str) -> str:
    cleaned = "".join("_" if char in '<>:"/\\|?*' else char for char in value).strip(" .")
    return cleaned or time.strftime("lichtfeld_%Y%m%d_%H%M%S")


def _on_post_step(_hook: Any) -> None:
    if _logger is None:
        return
    try:
        _logger.log_step()
    except Exception as error:
        _logger.last_error = str(error)
        _logger.status = "error"
        lf.log.error(f"TensorBoard export failed: {error}")


def _on_training_start(_hook: Any) -> None:
    if _logger is not None:
        try:
            _logger.start(force=True, reason="training start")
        except Exception as error:
            _logger.last_error = str(error)
            _logger.status = "error"
            lf.log.error(f"TensorBoard export start failed: {error}")


def _on_training_end(_hook: Any) -> None:
    if _logger is not None:
        try:
            _logger.close(reason="training end")
        except Exception as error:
            _logger.last_error = str(error)
            _logger.status = "error"
            lf.log.error(f"TensorBoard export close failed: {error}")


def _start_run(force: bool = False, reason: str = "") -> None:
    if _logger is not None:
        _logger.start(force=force, reason=reason)


def _close_run(reason: str = "") -> None:
    if _logger is not None:
        _logger.close(reason=reason)


def _open_run_folder() -> None:
    if _logger is None or _logger.run_dir is None:
        return
    try:
        os.startfile(str(_logger.run_dir))
    except Exception as error:
        _logger.last_error = str(error)
        lf.log.error(f"TensorBoard export failed to open run folder: {error}")


def _copy_tensorboard_command() -> None:
    try:
        lf.ui.set_clipboard_text(_tensorboard_command())
    except Exception as error:
        if _logger is not None:
            _logger.last_error = str(error)
        lf.log.error(f"TensorBoard export failed to copy command: {error}")


_classes = [TensorBoardExportPanel]
_logger: MetricsLogger | None = None
_handler: Any | None = None


def on_load() -> None:
    global _logger
    global _handler

    _logger = MetricsLogger()
    for cls in _classes:
        lf.register_class(cls)

    _handler = lf.ScopedHandler()
    _handler.on_training_start(_on_training_start)
    _handler.on_post_step(_on_post_step)
    _handler.on_training_end(_on_training_end)
    lf.log.info("TensorBoard export plugin loaded")


def on_unload() -> None:
    global _logger
    global _handler

    if _handler is not None:
        _handler.clear()
        _handler = None
    if _logger is not None:
        _logger.close(reason="plugin unload")
        _logger = None

    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("TensorBoard export plugin unloaded")
