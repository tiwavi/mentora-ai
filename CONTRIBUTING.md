# Collaboration & Git Workflow Guide (Team DataPulse)

## 1. Branch Ownership
No team member commits directly to main. All development happens on dedicated role branches:
- **Shreyansh Tiwari (Team Lead):** eature/ml-backend (works inside src/ and pp.py)
- **Shivansh Mittal (Frontend Lead):** eature/web-dashboard (works inside web/)
- **Aryan Jain (Data & QA Lead):** eature/content-and-tests (works inside data/content/, 	ests/, and docs/)

## 2. Standard Commit & Pull Request Workflow
1. Sync with main: git checkout main; git pull origin main
2. Switch to your branch: git checkout <your-branch-name> and git merge main
3. Stage only your owned files: git add <folder>
4. Commit with a clear prefix ([ML], [UI], [DATA], or [TEST])
5. Push your branch: git push origin <your-branch-name>
6. Open a Pull Request (PR) on GitHub targeting main and request review from Shreyansh Tiwari.
