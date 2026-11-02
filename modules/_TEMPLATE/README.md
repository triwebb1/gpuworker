# Template module

Copy this directory to `modules/<your-module>/`, rename `module.toml.example` to
`module.toml`, then fill in `module.toml`, `provision.sh` and `src/`. (The registry
picks up any `modules/*/module.toml`; this template is skipped only because its file is
named `.example`. The image build skips directories starting with `_`.) See
[docs/MODULE_GUIDE.md](../../docs/MODULE_GUIDE.md) and the working example in
[modules/echo](../echo).
