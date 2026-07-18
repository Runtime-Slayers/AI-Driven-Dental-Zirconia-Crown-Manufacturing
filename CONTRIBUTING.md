# Contributing to AI-Driven Dental Zirconia Crown Manufacturing

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold respectful and inclusive behaviour.

## How Can I Contribute?

### 🐛 Reporting Bugs

- Use the GitHub Issues tab to report bugs.
- Include steps to reproduce, expected vs. actual behaviour, and your environment details.

### 💡 Suggesting Enhancements

- Open an issue with the `enhancement` label.
- Describe the feature, its motivation, and any implementation ideas.

### 🔧 Code Contributions

We welcome contributions across all stages and languages:

| Stage | Language | What You Can Improve |
|-------|----------|----------------------|
| Stage 1 | Python | Segmentation model, preprocessing |
| Stage 2 | Python | Transformer architecture, physics loss |
| Stage 3 | Python | Decision network, optimisation algorithms |
| Stage 4 | MATLAB | Robotics kinematics, milling paths |
| Stage 5 | Python | RL agents, G-code generation |

## Development Setup

```bash
# Fork and clone
git clone https://github.com/<your-username>/AI-Driven-Dental-Zirconia-Crown-Manufacturing.git
cd AI-Driven-Dental-Zirconia-Crown-Manufacturing

# Create environment
python -m venv venv
source venv/bin/activate
pip install torch numpy scipy h5py matplotlib
```

## Coding Standards

### Python
- Follow **PEP 8** style guidelines
- Use type hints where practical
- Include docstrings for all public functions and classes

### MATLAB
- Use function-level documentation with `%` comments
- Keep functions modular and well-named

### General
- Write meaningful commit messages
- Keep pull requests focused on a single change
- Add comments explaining non-obvious logic

## Submitting Changes

1. **Fork** the repository
2. Create a **feature branch**: `git checkout -b feature/my-improvement`
3. Make your changes and **test** them
4. **Commit** with a clear message: `git commit -m "feat: improve Stage2 transformer accuracy"`
5. **Push** your branch: `git push origin feature/my-improvement`
6. Open a **Pull Request** against `main`

### Commit Message Convention

We loosely follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` — New features
- `fix:` — Bug fixes
- `docs:` — Documentation changes
- `refactor:` — Code restructuring without behaviour change
- `test:` — Adding or updating tests
- `chore:` — Maintenance tasks

---

Thank you for helping improve this project! 🦷
