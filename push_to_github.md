# GitHub Repository Setup Commands

After creating your repository on GitHub, run these commands in your PowerShell:

## Replace YOUR_USERNAME with your actual GitHub username
```powershell
# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/pneumonia-detection-deep-learning.git

# Verify the remote was added correctly
git remote -v

# Push to GitHub (you'll be prompted for your GitHub credentials)
git push -u origin master
```

## Alternative: If you prefer to use main branch instead of master
```powershell
# Rename branch to main (optional, modern convention)
git branch -M main

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/pneumonia-detection-deep-learning.git

# Push to main branch
git push -u origin main
```

## If you need to authenticate:
- Use your GitHub username
- For password, use a Personal Access Token (not your GitHub password)
- To create a token: GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)

## Repository Details for GitHub:
- **Name**: pneumonia-detection-deep-learning
- **Description**: Deep Learning project for pneumonia detection from chest X-rays using CNN, ResNet50, and DenseNet121 (CheXNet). Best model achieves 85%+ accuracy.
- **Topics**: machine-learning, deep-learning, computer-vision, medical-imaging, pneumonia-detection, tensorflow, keras, densenet, resnet, cnn