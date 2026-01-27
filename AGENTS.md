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

### General Python Rules

- **Style Guide**: 代码必须严格遵循 **Google Python Style Guide**。
- **Formatter**: 使用 `black` 格式化代码，行宽限制 100 字符；使用 `isort` 管理导入。
- **Naming**:
  - 文件名: `snake_case.py`
  - 变量/函数: `snake_case`
  - 类名: `PascalCase`
  - 私有成员: `_leading_underscore`
  - 常量: `UPPER_CASE`

### Type Hinting (强制)

- 所有函数参数、返回值必须有类型标注。
- 复杂类型使用 `typing` 模块 (e.g., `List`, `Dict`, `Optional`, `Union`, `Tuple`)。
- 避免使用 `Any`，除非完全不可预知；尽量使用具体类型或 `Protocol`。

### Docstrings (中文 Google 风格)

所有模块、类、函数必须包含中文 Docstring，格式如下：

1. **Summary**: 第一行简述功能，通过空行与详细描述分隔。
2. **Args**: 列出参数名和描述。
   - **不要**在 Args 里重复写 `(int)` 这种类型，因为函数签名里已经有了。
   - 如果参数有默认值或特定限制，在描述中说明。
3. **Returns**: 描述返回值。
   - 如果返回 `None`，**省略**此部分。
   - 如果是生成器，使用 **Yields**。
4. **Raises**: **仅当**函数显式抛出异常时编写此部分。
   - **禁止**写 `Raises: None` 或 `Raises: 无`。
5. **Note** (可选): 如果有复杂的实现细节、算法引用或特别注意事项，放在这里，不要创造自定义的 Header（如“实现要点”）。

**Docstring Example (严格参考):**

```python
def fetch_data(url: str, timeout: int = 10) -> Dict[str, Any]:
    """从指定 URL 获取数据并解析为字典。

    会对请求进行重试，如果三次失败则抛出异常。

    Args:
        url: 目标 API 的地址。
        timeout: 请求超时时间（秒）。

    Returns:
        包含响应数据的字典对象。

    Raises:
        ConnectionError: 网络连接失败。
        ValueError: 响应无法解析为 JSON。
    """
    pass
```

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

# README.md维护

完成重要阶段后，你要自行判断是否需要更新README.md
