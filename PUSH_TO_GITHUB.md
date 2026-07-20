# How to push this project to GitHub

## Step 1 — Copy the project folder to your computer
Download / move the `resume-agent/` folder to wherever you want to keep it (e.g. Desktop).

## Step 2 — Open a terminal in that folder, then run:

```bash
cd path/to/resume-agent

git init
git add .
git commit -m "Initial commit: Intelligent Resume Analysis Agent (Qwen2.5 + RAG)"

git remote add origin https://github.com/Clara-xiao/Intelligent-Resume-Analysis-and-Recruitment-Agent-System-Based-on-Qwen2.git
git branch -M main
git push -u origin main
```

If the repo already has a README on GitHub (created when you made the repo), add `--force` to the last line:
```bash
git push -u origin main --force
```

## That's it — your repo will be live at:
https://github.com/Clara-xiao/Intelligent-Resume-Analysis-and-Recruitment-Agent-System-Based-on-Qwen2
