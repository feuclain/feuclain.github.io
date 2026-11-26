# Personal Academic & Engineering Site (Astro + KaTeX)

A clean, minimal, fast personal website built for an incoming Kyoto University NLP Master's student — designed to be **zero-framework-knowledge friendly** to maintain and deploy to GitHub Pages.

## Features / 核心特性

- **Home (`/`)**: Personal bio, status badge, focus areas, latest notes, live sprint progress, and recent timeline milestones.
- **Writing (`/writing`)**: Plain Markdown articles with **KaTeX math** (`$...$`, `$$...$$`), syntax highlighting (light/dark dual theme), one-click copy buttons, sticky Table of Contents, category tabs (`AI / LLM`, `Algorithms`, `Essays`), and instant search.
- **Timeline (`/timeline`)**: Filterable chronological milestone log powered by a single JSON file (`src/data/timeline.json`).
- **Now & Plan (`/roadmap`)**: Active sprint progress bars + multi-phase future roadmap with status filtering (`Done`, `In Progress`, `Planned`), powered by `src/data/roadmap.json`.
- **Dark / Light Mode**: Warm ivory light theme + deep obsidian dark theme with zero flash on load.

---

## 日常如何维护（完全不需要懂前端框架！）

你日常只需要改这 **4 个地方**：

| 你想做什么 | 只需要编辑哪里 |
| :--- | :--- |
| **写一篇新文章 / 题解 / 杂谈** | 在 `src/content/writing/` 里新建一个 `.md` 文件（参考现有文章的顶部 frontmatter 格式） |
| **修改个人简介、姓名、GitHub 链接** | 编辑 `src/data/site.json` |
| **往时间线加一条新里程碑** | 在 `src/data/timeline.json` 的 `milestones` 数组最前面加一条对象 |
| **更新近期进度或未来计划表** | 编辑 `src/data/roadmap.json` |

### 写新文章模板（`src/content/writing/my-new-post.md`）

```markdown
---
title: "Your Article Title in English"
date: 2026-12-01
category: ai-llm   # 可选: ai-llm | algorithms | essays
summary: "One or two sentences summarizing what this post covers."
tags: [llm, pytorch]
---

Write your Markdown here. Inline math like $E = mc^2$ and block math:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$
```

---

## 本地预览命令

```bash
npm install
npm run dev
```

浏览器打开终端提示的 `http://localhost:4321` 即可实时预览。

---

## 如何免费部署到 GitHub Pages

1. `astro.config.mjs` 中的 `site` 已配置为 `https://feuclain.github.io`。
   - 如果迁移到非 `<用户名>.github.io` 的普通子仓库（如 `blog`），再额外加一行 `base: '/blog'`。
2. 把代码 push 到 GitHub 仓库的 `main` 分支。
3. 打开 GitHub 仓库页面 → **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。
4. 仓库内置的 `.github/workflows/deploy.yml` 会自动构建并发布，以后每次 `git push` 都会自动更新网站！
