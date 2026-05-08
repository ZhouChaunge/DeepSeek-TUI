//! Per-alias secondary-model cost side-channel (#18).
//!
//! Mirrors the [`crate::cost_status`] pattern: `llm_call` reports each
//! call here; the TUI sidebar drains and displays the per-alias breakdown.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

/// Per-alias cost entry accumulated since last drain.
#[derive(Debug, Clone, Default)]
pub struct AliasCostEntry {
    pub calls: u32,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_usd: f64,
}

type AliasMap = BTreeMap<String, AliasCostEntry>;

static PENDING: OnceLock<Mutex<AliasMap>> = OnceLock::new();

fn cell() -> &'static Mutex<AliasMap> {
    PENDING.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Called by `llm_call` after each successful secondary-model invocation.
/// `cost_usd` is estimated via a simple per-token rate; pass `0.0` if
/// pricing is unknown.
pub fn report(alias: &str, tokens_in: u64, tokens_out: u64, cost_usd: f64) {
    if let Ok(mut map) = cell().lock() {
        let entry = map.entry(alias.to_string()).or_default();
        entry.calls += 1;
        entry.tokens_in += tokens_in;
        entry.tokens_out += tokens_out;
        entry.cost_usd += cost_usd;
    }
}

/// Drain the accumulated per-alias costs and reset to zero.
/// Called by the TUI sidebar render loop.
pub fn drain() -> AliasMap {
    let Ok(mut map) = cell().lock() else {
        return BTreeMap::new();
    };
    std::mem::replace(&mut *map, BTreeMap::new())
}

/// Peek at the current accumulated costs without resetting.
/// Used to display the running total without losing data mid-frame.
pub fn peek() -> AliasMap {
    cell()
        .lock()
        .map(|m| m.clone())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reset() {
        if let Ok(mut m) = cell().lock() {
            m.clear();
        }
    }

    #[test]
    fn report_and_peek() {
        reset();
        report("reviewer", 100, 50, 0.001);
        let snap = peek();
        assert_eq!(snap["reviewer"].calls, 1);
        assert_eq!(snap["reviewer"].tokens_in, 100);
        let drained = drain();
        assert_eq!(drained["reviewer"].calls, 1);
        let after_drain = peek();
        assert!(after_drain.is_empty());
    }
}
