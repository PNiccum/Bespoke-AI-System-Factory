# Private repo with CI on your own laptop

GitHub Actions minutes are free and unlimited for **self-hosted runners**, so the repo can stay
private. CI only runs while the laptop is on, online, and the runner service is running; otherwise
PR checks wait as "queued" until it is back.

## 1. One-time laptop setup
Install: Git for Windows (gives `bash`), Python 3.11 and 3.12 (tick "Add to PATH"), and optionally
Docker Desktop (Linux containers) so the sandbox container tests run; without it they skip.
Open PowerShell and check: `git --version; py -3.11 --version; py -3.12 --version; docker version`.

## 2. Register the runner as a service
1. GitHub repo > Settings > Actions > Runners > **New self-hosted runner** > Windows x64.
2. Run the shown download/extract commands in a folder such as `C:\actions-runner` (not in Dropbox).
3. `./config.cmd --url https://github.com/PNiccum/Bespoke-AI-System-Factory --token <token shown>`
   Accept defaults; when asked "run as service?" answer **Y** (or later `./svc.cmd install; ./svc.cmd start`).
4. The runner should show **Idle** on the Runners page.

## 3. Point CI at it
Repo > Settings > Secrets and variables > Actions > **Variables** > New repository variable:
name `CI_RUNNER`, value `["self-hosted","windows"]`. Delete the variable to return to GitHub-hosted runners.

## 4. Make the repo private
Settings > General > Danger Zone > **Change visibility** > Private. Do this after step 3 so CI keeps running.
(Anyone who cloned while it was public keeps their copy; keys and lab data were never committed.)

## Notes
- Only run the runner on a private repo with no outside collaborators; self-hosted runners execute PR code.
- The Claude review workflow uses the same runner; it needs Node on the laptop and the `CLAUDE_CODE_OAUTH_TOKEN` secret.
  If it misbehaves on Windows, it just fails; it never blocks tests.
- If the laptop is off, leave PRs queued or set `CI_RUNNER` back to `"ubuntu-latest"` when minutes reset.

## Dropbox backup (optional)
`scripts/backup_bundle.ps1` writes a timestamped `.bundle` (full history, all branches) into
`...\Final Project\repo-backups` and keeps the newest 14. Schedule it daily (PowerShell, from the repo folder):

    schtasks /Create /SC DAILY /ST 21:00 /TN "ASPIRE repo backup" /TR "powershell -NoProfile -File C:\path\to\repo\scripts\backup_bundle.ps1"

Restore: `git clone aspire-<stamp>.bundle restored`. GitHub stays the source of truth.
