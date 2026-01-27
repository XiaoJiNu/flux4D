# Repository Guidelines

This repository contains research artifacts and notes under `docs/`. Contributions should keep the root clean, add source material in well-named folders, and avoid committing derived binaries without their sources.

## Project Structure & Module Organization

- `docs/论文/202512_Flux4D.pdf`: primary document (treat as generated output if a source is added later).
- `docs/论文/Flux4D-基于光流的无监督4D重建-翻译.md`: translated notes.
- `docs/需求/`: requirements and prompts.
- `docs/开发记录/`: plans and progress notes.
- `docs/skills/`: local skill references.
- Recommended additions (create as needed):
  - `configs/`: training/eval configuration files.
  - `data/`: metadata indices and optional caches (avoid raw datasets).
  - `paper/`: LaTeX/Markdown sources used to generate the PDF (e.g., `paper/main.tex`).
  - `assets/`: figures, tables, and media referenced by the paper (e.g., `assets/figures/`).
  - `src/`: implementation code (if/when added).
  - `scripts/`: entry points for preprocessing/training/inference.
  - `tools/`: visualization and debugging utilities.
  - `tests/`: automated tests (if/when added).
  - `third_party/`: external dependencies kept in-repo as needed.

## Build, Test, and Development Commands

There is no build/test tooling in the current tree. If you add sources or code, include a short, reproducible command set in `README.md` (and preferably a `Makefile`). Examples:

- `make paper`: build `202512_Flux4D.pdf` from `paper/`.
- `make lint`: run format/lint checks for `src/`.
- `make test`: run the test suite in `tests/`.

## Coding Style & Naming Conventions

- Indentation: 4 spaces for Python; keep lines ≤ 100 chars.
- Naming: `snake_case.py` files, `snake_case` functions, `PascalCase` classes.
- Prefer automated formatting/linting when code exists (e.g., `ruff` + `black`) and keep config at repo root.
- Python 代码需遵循 Google Python Style Guide，保持一致的模块结构与导入顺序。
- 所有函数/类必须写中文 Docstring，推荐 Google 风格，至少包含：功能说明、参数（含类型/含义）、返回值、可能抛出的异常（如有）、实现方法要点（简述思路/步骤）。
- 必须使用类型标注：函数参数、返回值、关键成员变量均需显式标注，复杂类型优先用 `typing`。
- 复杂逻辑需添加简短中文行内注释，解释“为什么/怎么做”，避免重复代码本身的含义。
- 禁止静默吞异常；需要明确异常处理策略或日志记录。
- 新增可复用函数/模块需补充最小单元测试（`tests/` 下 `test_*.py`）。

## Testing Guidelines

No tests exist yet. If you introduce code, add `pytest`-style tests under `tests/` using `test_*.py` naming, and ensure new functionality is covered by a minimal unit test.

## Commit & Pull Request Guidelines

No Git history is present in this checkout. If/when versioned, use Conventional Commits for clarity:

- `docs: revise Flux4D abstract`
- `feat: add training pipeline skeleton`
- `fix: correct coordinate transform`

PRs should include a brief description, rationale for changes, and (for document updates) a short summary of what changed in the PDF (screenshots or a diff tool output if available).

## Data & Large Files

Avoid committing datasets, checkpoints, or other large binaries. Document download steps and use `.gitignore`/Git LFS when large artifacts are unavoidable.

## Agent-Specific Instructions

- Prefer editing sources under `paper/` and regenerating the PDF via a build command; avoid direct binary edits to `docs/论文/202512_Flux4D.pdf`.
- Keep diffs focused and update this file when the project layout/tooling changes.

## 用中文和我交流

## 版本管理规范

- 每个阶段完成后必须提交一次；阶段可定义为：完成一个功能/实验、修复一个 bug、更新一批文档或数据说明。
- 提交前必须更新 `docs/开发记录/`，记录动机、关键变更、结论与下一步。
- 提交消息使用 Conventional Commits（如 `feat: add flow field loader`、`docs: update experiment notes`、`fix: handle empty input`）。
- 提交粒度小且单一目的，避免混合“代码+大量无关文档”。
- 提交前检查 `git status` 干净；若有测试，至少运行最小相关测试。
- 禁止提交大体积二进制/数据；如必须，使用 LFS 并在 `README.md` 记录来源与获取方式。
- 禁止使用 `--amend` 和 `--force` 覆写历史（除非明确说明）。
