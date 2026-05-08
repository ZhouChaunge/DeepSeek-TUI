//! Skill-directory file-system watcher for live hot-reload (#19).
//!
//! Uses `notify` v6 in recommended (native) mode. The watcher runs on a
//! background thread and sends a `SkillReloadEvent` over a channel whenever
//! a `SKILL.md` file inside the watched directory is created, modified, or
//! removed.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Duration;

use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};

/// Event emitted to the main thread when skills should be reloaded.
#[derive(Debug, Clone)]
pub struct SkillReloadEvent {
    /// The file that triggered the reload (for logging).
    pub trigger_path: PathBuf,
}

/// Start watching `skills_dir` for `SKILL.md` changes.
///
/// Returns a `SkillWatcherHandle` that yields `SkillReloadEvent`s whenever a
/// re-scan is needed. The watcher runs until the handle is dropped.
///
/// Returns `None` if the directory does not exist or if the OS watcher
/// could not be initialised — both are treated as silent no-ops so the TUI
/// still starts normally.
pub fn start_watcher(skills_dir: PathBuf) -> Option<SkillWatcherHandle> {
    if !skills_dir.exists() {
        return None;
    }

    let (tx, rx): (Sender<SkillReloadEvent>, Receiver<SkillReloadEvent>) = channel();

    let watcher = build_watcher(tx, &skills_dir)?;

    Some(SkillWatcherHandle {
        _watcher: watcher,
        rx,
        skills_dir,
    })
}

/// Opaque handle that keeps the watcher alive.
/// Drop this to stop watching.
pub struct SkillWatcherHandle {
    _watcher: RecommendedWatcher,
    pub rx: Receiver<SkillReloadEvent>,
    pub skills_dir: PathBuf,
}

impl std::fmt::Debug for SkillWatcherHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkillWatcherHandle")
            .field("skills_dir", &self.skills_dir)
            .finish()
    }
}

fn build_watcher(tx: Sender<SkillReloadEvent>, skills_dir: &Path) -> Option<RecommendedWatcher> {
    let mut last_event: Option<std::time::Instant> = None;
    let debounce = Duration::from_millis(300);

    let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
        let Ok(event) = res else { return };

        // Only care about SKILL.md files.
        let relevant = event.paths.iter().any(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.eq_ignore_ascii_case("SKILL.md"))
                .unwrap_or(false)
        });

        if !relevant {
            return;
        }

        // Debounce: skip if we just fired within the debounce window.
        let now = std::time::Instant::now();
        if let Some(last) = last_event {
            if now.duration_since(last) < debounce {
                return;
            }
        }
        last_event = Some(now);

        let trigger_path = event.paths.first().cloned().unwrap_or_default();
        let _ = tx.send(SkillReloadEvent { trigger_path });
    })
    .ok()?;

    watcher.watch(skills_dir, RecursiveMode::Recursive).ok()?;

    Some(watcher)
}
