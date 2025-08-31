# GitHub Setup Instructions

Follow these steps to push your project to GitHub:

## 1. Create a new repository on GitHub

Go to [GitHub.com](https://github.com) and create a new repository. Do NOT initialize the repository with a README, license, or .gitignore file since you already have these files locally.

## 2. Initialize local Git repository

```bash
# Initialize Git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: Biodiversity Criteria Evaluator"
```

## 3. Link and push to GitHub repository

Replace `YOUR_USERNAME` with your actual GitHub username and `YOUR_REPO` with your repository name.

```bash
# Add the remote repository
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push to GitHub
git push -u origin main
```

If your default branch is named `master` instead of `main`, use:

```bash
git push -u origin master
```

## 4. Verify GitHub repository

Visit your GitHub repository at `https://github.com/YOUR_USERNAME/YOUR_REPO` to ensure everything has been pushed correctly.

## Need to update your repository?

When you make changes to your code, push them to GitHub with:

```bash
# Add changes
git add .

# Commit changes with a descriptive message
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Important Notice

Make sure to check that your `.env` file is correctly ignored and not pushed to GitHub, as it contains sensitive API keys. You can verify this by looking at your GitHub repository - the `.env` file should not appear in the file list.
