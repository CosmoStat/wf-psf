# **WaveDiff Development Plan**

## 1.  **Define a Milestone**
    

  * **Purpose**: Milestones mark the completion of project cycles and achievement of predefined goals. A milestone is a timeline point marking the end of a project cycle, where a set of goals defined at the start of the cycle have been completed.  Goals could be tasks associated with feature development, testing phase completion, etc. List & Open issues. 
    
  * **Types of Milestones:** Differentiate between major, minor and patch release releases to track progress effectively.
    

Make a milestone related to a release on possibly a minor release schedule (e.g. monthly schedule). Differentiate between the different types of releases (minor, major, patch).

## 2.  **Git Workflow Branching Model (Made with** [**mermaid**](https://mermaid.js.org)**)**
![](assets/Git-Branching-Model.png)

   *   **main**: Stores official release history with tagged version numbers (see top row in the diagram above).
    
   *   **develop**: This branch is the integration branch for features.
    
   *   **feature**: Branch off of `develop` for new feature development.  Features get merged to `develop` (never to main).
    
   *   **bug**: For fixing specific bugs introduced during the production phase.
    
   *   **hotfix**: Quickly corrects bugs in the latest release version; branched off `main`.

   *   **release/vX.Y.Z**: Temporary branch created from `develop`; used to:
         * Collect Scriv changelog fragments here.
         * Apply final QA or documentation updates.
         * Merge into `main` when ready and tag the release.
    

## 3.  **New Branch naming conventions**
    

Branches created for the purpose of a Pull Request should be directly related to an open issue within the project’s issue tracker.  This ensures that all code changes are tied to specific tasks or features, facilitating better organisation and tracking of contributions. Below are the branch naming conventions:

   *   **Feature**: `feature/<issue number>/short-description`
    
   *   **Bug**: `bug/<issue number>/short-description`
    
   *   **Hotfix**: `hotfix/<issue number>/short-description`

   *   **Release**: `release/v<MAJOR>.<MINOR>.<PATCH>` 

**Notes:**
- Replace `<ISSUE_NUMBER>` with the tracker issue number.
- Keep short-description concise.
- Release branches are short-lived and should not include ongoing development.
    

Replace with the corresponding issue number and provide a brief description of the changes in \`short\_description\`.

## 4.  **Feature branch creation** 
    

*   Pull the latest changes from the remote repository
    
*   Checkout `develop`
    
*   Create a new feature branch
    
*   Ensure the feature branch’s commits align with the defined scope
    

**Tip**: Keep feature branch development focused on the defined scope outlined in the corresponding issue. Avoid introducing unrelated changes. Instead, open a new issue for out-of-scope features or bugs, creating a separate branch for their development.

## 5.  **Completion of development** 
    

*   Run training and metrics validation tests locally to confirm no breaking behaviour or errors were introduced.  Include test reports as per the PR template.
    
*   Open a Pull Request to start the review process, ensuring to map the branch correctly to the target branch (i.e. `feature` -> `develop` or `hotfix` -> `main`).
    
*   In the description of the Pull Request, explicitly state whether the PR resolves or closes an open issue by using one of the following formats:
    
    *   "Solves #"
        
    *   "Closes #"
        

Example: "Solves #12345" or "Closes #8679".

*   Ensure that the PR meets the defined requirements and passes Continuous Integration (CI) tests.
    
*   Assign a reviewer, assign yourself as assignee, select the correct project, choose the correct label that describes the nature of the issue (e.g. bug, enhancement, etc), choose Milestone if applicable if the issue is associated with a specific milestone or target release.
    
*   Address reviewer feedback in threads and iterate as needed, implementing the requested changes or explaining why the task is implemented in the manner it is.  Respect the rule: Whoever opens the thread/conversation in the Pull Request is responsible for closing it.
    

## 6.  **Merging Pull Requests** 
    
*   Approval and Merging: Once the reviewer approves the PR and all feedback is addressed, merge the feature branch into develop.  Note, it is the reviewer who is responsible for merging the PR when satisfied with the changes.
    
## 7.  **Changelog and release notes**

WaveDiff uses **Scriv** to manage its changelog and release notes.

Instead of editing `CHANGELOG.md` directly during development, contributors add small **changelog fragments** describing user-visible changes. These fragments are committed alongside pull requests and collected automatically when preparing a release.

This approach:
- Avoids merge conflicts on the changelog
- Ensures consistent, structured release notes
- Keeps release notes focused on user impact

### Changelog fragments

- Fragments are stored in `changelog.d/`
- Each fragment corresponds to a single pull request
- Fragments must be **staged and committed** as part of the PR
- Contributors create fragments using:
   ```bash
   scriv create
   ```
**Notes**: 
- Do not collect fragments in your PR; leave [Unreleased] in CHANGELOG.md.
- [Unreleased] is always at the top; contributors never write directly into it.

Fragment creation and writing guidelines are documented in [CONTRIBUTING.md](./CONTRIBUTING.md).

### Branching and scope
- Changelog fragments should reflect **what is merged**, not how it is implemented.
- Feature branches should include only feature-related fragments
- Bug fixes discovered during feature development should be moved to a **separate branch and pull request**, with their own fragment
This separation keeps release notes accurate and enables clean backports

## 8.  **Preparing for a Release**
    

*   Each milestone targets a release (feature, patch, etc).
    
*   Open a PR from `develop` to `release/vX.Y.Z` upon completing a milestone
    
* Collect changelog fragments and generate release notes using:

   ```bash
   scriv collect --version vX.Y.Z

*   Ensure all checklist items for the release are completed.

*   Merge the PR into main and tag the release.
    

## 9.  **Continuous Improvement**
    

*   Regularly review and refine the workflow based on team feedback and lessons learned.
    
*   Encourage collaboration and communication among team members to streamline processes and enhance productivity.
    

## 10.  **Documentation and Training**
    

*   Maintain up-to-date documentation outlining the development workflow and procedures.
    
*   Provide training to new team members and ensure existing members are familiar with the workflow and best practices.


## 11.  **Automation and Tooling**
    

*   Explore automation tools to streamline repetitive tasks, such as testing, code formatting, and deployment.
    
*   Integrate with CI/CD pipelines for automated testing and deployment processes.
    
