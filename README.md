# Personal Academic & Engineering Site (`feuclain.github.io`)

Minimal, fast personal website built with **Astro + Markdown + KaTeX** for incoming Kyoto University NLP Master's studies.

## 页面结构与日常维护指南

| 板块 | 维护方式 |
| :--- | :--- |
| **Home (`/`)** | 编辑 `src/data/site.json`（简介、Research Interests、Projects、Activities 单行动态、可选 Publications） |
| **Writing (`/writing`)** | 在 `src/content/writing/` 新建 `.md` 文章（支持 KaTeX 数学公式、代码高亮、可选 `cover` 封面图） |
| **Timeline (`/timeline`)** | 在 `src/content/timeline/` 新建 `.md` 条目（支持无标题短动态、单图/多图画廊放大、公式、代码块、地点标签） |
| **Plan (`/roadmap`)** | **支持直接在网页上手动编辑！** 点击右上角 `Edit Plan` 输入一次仅限本仓库的 GitHub Token 即可在线打勾、增删改并一键保存 |

---

## 1. 如何在网页上直接修改 Plan（权限与首次配置）

访客打开 `https://feuclain.github.io/roadmap` 时为**纯只读模式**。你自己在常用浏览器首次配置只需 30 秒：

1. 打开 GitHub [Fine-grained Personal Access Tokens 创建页](https://github.com/settings/personal-access-tokens/new)。
2. **Repository access** 选择 **Only select repositories** → 勾选 `feuclain/feuclain.github.io`。
3. **Permissions → Repository permissions** 找到 **Contents**，设为 **Read and write**（其余保持默认无需勾选），生成 Token。
4. 打开你的网站 `/roadmap` 页面 → 点击右上角 **`Edit Plan`** → 粘贴该 Token。
   - Token **仅保存在你当前浏览器的 `localStorage`**，绝不会上传或暴露给任何访客。
   - 解锁后可直接在网页上勾选完成、修改聚焦说明、增删或上下移动阶段与任务，按 `Ctrl+S` 或点击 **`Save to GitHub`** 即可直接写回仓库并实时生效！

---

## 2. 如何往 Timeline 插入图片或多元化卡片

1. 把图片放到 `public/images/timeline/` 目录下（例如 `public/images/timeline/kyoto.jpg`）。
2. 在 `src/content/timeline/` 下新建 `.md` 文件（标题 `title`、地点 `location`、图片 `images`、链接 `links` 全部可选，想写多简短或多丰富都可以）：

```markdown
---
date: 2027-04-01
displayDate: "2027.04.01"
category: japan          # 可选: ai-llm | algorithms | academic | japan | life
title: "Arrived in Kyoto" # 可选：删掉这行就会渲染成轻量的无标题动态卡片
location: "Kyoto, Japan"
images:
  - src: "/images/timeline/kyoto.jpg"
    alt: "Kyoto campus"
    caption: "First spring morning near Yoshida Campus."
links:
  - label: "Read Essay"
    url: "/writing/from-se-to-kyoto-nlp"
---

正文支持完整 Markdown、**加粗**、引用、代码块、行内图片 `![](...)` 以及公式 $E = mc^2$。
```

---

## 3. 本地开发与预览

```bash
npm install
npm run dev
```

访问 `http://localhost:4321` 即可实时预览；推送到 `main` 分支后 GitHub Actions 会自动构建部署。
