# Contributing to WaveDiff

WaveDiff is a software development project aimed at modeling the point spread function of an optical system using auto differentiation techniques. These guidelines are designed to facilitate contributions to WaveDiff by providing clear instructions on issue reporting, pull request submission, and code style standards.

## Contents

1. [Introduction](#introduction)
2. [Workflow](#workflow)
3. [Issues](#issues)
4. [Pull Requests](#pull-requests)
   1. [Before Opening a PR](#before-opening-a-pr)
   2. [Opening a PR](#opening-a-pr)
   3. [Revising a PR](#revising-a-pr)
   4. [After a PR is Merged](#after-a-pr-is-merged)
5. [Content](#content)
6. [CI Tests](#ci-tests)
7. [Style Guide](#style-guide)

## Introduction

Welcome to the WaveDiff project! We appreciate your interest in contributing to our software development efforts. Before you get started, please take a moment to review and adhere to these guidelines and our [code of conduct](./CODE_OF_CONDUCT.md).

## Workflow

The WaveDiff project follows a [specific workflow](./DEV_WORKFLOW.md) to streamline the development process. Familiarize yourself with the following workflow guidelines:

1. **Milestone Definition**: Define milestones to mark the completion of project cycles, such as feature development or testing phases. Milestones are typically scheduled releases, such as monthly releases, and can include different types such as minor, major, or patch releases.

2. **Git Branching Model**: WaveDiff follows a Git branching model to manage code changes efficiently. The main branches include:
   - `main`: Stores official release history, with each commit tagged with a version number.
   - `develop`: Integration branch for features.
   - `feature`: Branches for developing new features, created from the `develop` branch.
   - `bug`: Branches for fixing specific bugs, created from the `develop` branch.
   - `hotfix`: Branches for quickly correcting bugs in the latest release version, created from the `main` branch.

3. **Branch Naming Conventions**: Use specific naming conventions for branches to indicate their purpose and associated issue. For example:
   - Feature branch: `feature/issue-<issue_number>/short-description`
   - Bug branch: `bug/issue-<issue_number>/short-description`
   - Hotfix branch: `hotfix/issue-<issue_number>/short-description`

4. **Pull Request Process**: Follow a structured process when opening and reviewing pull requests. Include steps for opening a PR, revising it based on feedback, and merging it into the appropriate branch.

5. **Preparing for a Release**: Outline the steps to prepare for a release, including merging changes into the main branch and tagging the release.

These workflow guidelines ensure consistency and efficiency in the development process and help maintain a high-quality codebase for the WaveDiff project.

## Issues

Reporting issues is a valuable way to contribute to the improvement of WaveDiff. Follow these guidelines when reporting issues:

- Use the [issue tracker](https://github.com/CosmoStat/wf-psf/issues/new/choose) to report bugs, request features, or ask questions.
- Provide clear and descriptive titles and descriptions for your issues.
- Search existing issues to avoid duplicates before submitting a new one.

## Pull Requests

To contribute code changes or new features to WaveDiff, please follow these steps:

### Before Opening a PR

1. Log in to your GitHub account or create one if you don't have one already.
2. Navigate to the [WaveDiff repository](https://github.com/CosmoStat/wf-psf/).
3. Check if an [issue](#issues) exists for the changes you plan to make. If not, [open one](https://github.com/CosmoStat/wf-psf/issues/new/choose) and specify that you plan to open a PR to address it.
4. Fork the repository to create your own copy.
5. Clone your fork of WaveDiff to your local machine.
```bash
git clone git@github.com:<your-username>/wf-psf.git
```
6. Add the original repository (upstream) as a remote.
```bash
git remote add upstream git@github.com:CosmoStat/wf-psf.git
```

### Opening a PR

1. Pull the latest updates from the original repository.
```bash
git pull upstream develop
```
2. Create a new branch for your modifications.
```bash
git checkout -b <branch-name>
```
3. Make the desired modifications to the codebase.
4. Add and commit your changes.
```bash
git add .
git commit -m "Description of the changes"
```
5. Push your changes to your fork on GitHub.
```bash
git push origin <branch-name>
```
6. Open a [pull request](https://github.com/CosmoStat/wf-psf/compare) with a clear description of your changes and reference any related issues.

### Revising a PR

1. After opening a PR, a project maintainer will review your changes and provide feedback.
2. Address any comments or requested changes in your PR.
3. Push your changes to the same branch in your fork.
4. The reviewer will merge your PR into the main branch once it's approved.

### After a PR is Merged

After your PR is merged, follow these steps to keep your fork up to date:

1. Switch to your local develop branch.
```bash
git checkout develop
```
2. Delete the branch you used for the PR.
```bash
git branch -d <branch-name>
```
3. Pull the latest updates from the original repository.
```bash
git pull upstream develop
```
4. Push the changes to your fork.
```bash
git push origin develop
```

## Content

Every PR should correspond to a specific issue in the issue tracker. Tag the issue that the PR resolves (e.g., "This PR closes #1"). Keep PR content concise and focused on resolving a single issue. Additional changes should be made in separate PRs.

## CI Tests

All PRs must pass CI tests before being merged. Resolve any issues causing test failures and justify any modifications to unit tests in the PR description.

## Style Guide

Adhere to the following style guidelines when contributing to WaveDiff:

1. Compatibility: Ensure compatibility with specified Python package versions.
2. PEP8 Standards: Follow PEP8 standards for code formatting.
3. Docstrings: Provide docstrings for all modules, methods, and classes following numpydoc standards.
4. Existing Code: Use existing code as a reference for style conventions.
5. String Formatting: Prefer single quotes over double quotes for strings, and use explicit floats.
6. Line Length: Split long lines (>79 characters) using parentheses for readability.
7. String Concatenation: Use f-strings for string formatting and explicit concatenation with a `+` at the beginning of each line.

These guidelines will help maintain code consistency and readability throughout the project.

Feel free to customize these guidelines further to better suit the specific needs and conventions of the WaveDiff project.
