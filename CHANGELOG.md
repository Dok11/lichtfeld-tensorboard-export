# Changelog

## 0.1.0

- Initial publish-ready release.
- Export TensorBoard event files and CSV from LichtFeld training hooks.
- Start a unique run automatically on training start and close it on training end.
- Add explicit writer status in the plugin panel.
- Add run folder opening, TensorBoard command copying, and TensorBoard URL opening.
- Write `config.json` snapshots with schema and plugin versions.
- Record raw, windowed, and EMA loss metrics.
- Skip unavailable/noisy metrics instead of writing zero-value graphs.
- Stop exporting PSNR because current LichtFeld runtimes often expose it as an empty zero signal.
