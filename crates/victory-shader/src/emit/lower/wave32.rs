//! Wave32 emulation and uniform-branch planning for wave64 compute shaders.

use std::collections::{HashMap, HashSet};

use super::Lower;
use super::super::bitcode::{Attr, Bin, V, pred};
use super::super::shader::flags;
use crate::ir::Program;
use crate::structure::{Atom, LaneExit, Structured};

/// Per-block thread-group uniformity (`true` when every thread in the group
/// executes block `b` in lockstep, assuming group entry is uniform).
#[derive(Debug, Clone, Copy)]
pub(super) struct TwoSidedJoin {
    pub(super) then_b: usize,
    pub(super) else_b: usize,
    pub(super) enter_else: V,
    pub(super) act_before: V,
    pub(super) else_check: usize,
    pub(super) else_check_tail: usize,
    pub(super) br_dxil: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum UniformBranch {
    /// One target is a trivial `End` block (`if (ret_cond) return;`).
    EarlyReturn { stay: usize, ret: usize },
    /// One target is `ipdom[b]` (`if (body_cond) { body } skip`).
    OneSidedIf { body: usize, skip: usize },
    /// Two disjoint arms reconverging at `join = ipdom[b]`, serialized as `then` then `else`.
    TwoSidedIf {
        then_b: usize,
        else_b: usize,
        join: usize,
    },
    /// One target is a loop back-edge (`while (loop_cond) { ... }`).
    DivergentLoop { header: usize, exit: usize },
}

pub(super) fn ir_dominates(idom: &[usize], h: usize, mut u: usize) -> bool {
    while u != usize::MAX {
        if u == h {
            return true;
        }
        if idom[u] == u {
            break;
        }
        u = idom[u];
    }
    false
}

pub(super) fn uniform_blocks(p: &Program, st: &Structured) -> Vec<bool> {
    uniform_blocks_with(p, st, &HashSet::new()).0
}

type UniformAnalysis = (Vec<bool>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>);

fn uniform_blocks_with(
    p: &Program,
    st: &Structured,
    ignored_divergent: &HashSet<usize>,
) -> UniformAnalysis {
    let n = p.blocks.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut index = vec![usize::MAX; n];
    for (pos, &b) in order.iter().enumerate() {
        index[b] = pos;
    }
    let reachable = |b: usize| index[b] != usize::MAX;
    let succs = |b: usize| -> [Option<usize>; 2] {
        match st.exits[b] {
            LaneExit::End | LaneExit::Indirect => [None, None],
            LaneExit::Jump(t) => [Some(t), None],
            LaneExit::Branch {
                taken, not_taken, ..
            } => [Some(taken), Some(not_taken)],
        }
    };

    // Reverse-CFG post-dominators using virtual exit node `n`.
    let v_exit = n;
    let mut r_succs = vec![Vec::new(); n + 1];
    for &b in &order {
        let [s0, s1] = succs(b);
        match (s0, s1) {
            (None, None) => r_succs[v_exit].push(b),
            _ => {
                for s in [s0, s1].into_iter().flatten() {
                    r_succs[s].push(b);
                }
            }
        }
    }
    let mut seen = vec![false; n + 1];
    let mut post = Vec::with_capacity(n + 1);
    fn dfs_rev(
        u: usize,
        r_succs: &[Vec<usize>],
        seen: &mut [bool],
        post: &mut Vec<usize>,
    ) {
        if seen[u] {
            return;
        }
        seen[u] = true;
        for &v in &r_succs[u] {
            dfs_rev(v, r_succs, seen, post);
        }
        post.push(u);
    }
    dfs_rev(v_exit, &r_succs, &mut seen, &mut post);
    post.reverse();
    let mut r_index = vec![usize::MAX; n + 1];
    for (pos, &b) in post.iter().enumerate() {
        r_index[b] = pos;
    }
    let mut r_preds = vec![Vec::new(); n + 1];
    for &b in &order {
        let [s0, s1] = succs(b);
        match (s0, s1) {
            (None, None) => r_preds[b].push(v_exit),
            _ => {
                for s in [s0, s1].into_iter().flatten() {
                    r_preds[b].push(s);
                }
            }
        }
    }
    let mut ipdom = vec![usize::MAX; n + 1];
    if !post.is_empty() {
        ipdom[v_exit] = v_exit;
        let intersect = |ipdom: &[usize], mut a: usize, mut b: usize| {
            while a != b {
                while r_index[a] > r_index[b] {
                    a = ipdom[a];
                }
                while r_index[b] > r_index[a] {
                    b = ipdom[b];
                }
            }
            a
        };
        let mut changed = true;
        while changed {
            changed = false;
            for &b in &post[1..] {
                let mut new = usize::MAX;
                for &q in r_preds[b].iter().filter(|&&q| ipdom[q] != usize::MAX) {
                    new = if new == usize::MAX {
                        q
                    } else {
                        intersect(&ipdom, q, new)
                    };
                }
                if ipdom[b] != new {
                    ipdom[b] = new;
                    changed = true;
                }
            }
        }
    }

    let is_divergent_branch = |b: usize| {
        !ignored_divergent.contains(&b)
            && match st.exits[b] {
                LaneExit::Branch { when, .. } => st
                    .bools
                    .support(when)
                    .iter()
                    .any(|a| !matches!(a, Atom::Uniform(_))),
                _ => false,
            }
    };

    let mut uniform: Vec<bool> = (0..n)
        .map(|b| reachable(b) && ipdom[b] != usize::MAX)
        .collect();

    // Mark blocks between each divergent branch and its immediate post-dominator.
    for &b in &order {
        if !is_divergent_branch(b) {
            continue;
        }
        let stop = ipdom[b];
        let mut work = Vec::new();
        let mut visited = vec![false; n];
        for s in succs(b).into_iter().flatten() {
            if s != stop && !visited[s] {
                visited[s] = true;
                work.push(s);
            }
        }
        while let Some(u) = work.pop() {
            uniform[u] = false;
            for s in succs(u).into_iter().flatten() {
                if s != stop && !visited[s] {
                    visited[s] = true;
                    work.push(s);
                }
            }
        }
    }

    // Natural loops whose header/latch becomes non-uniform or that have a
    // divergent exit branch lose lockstep across the whole loop body.
    let idom = crate::ir::idoms(&p.blocks, &order, &index);
    let mut loops: Vec<(usize, HashSet<usize>)> = Vec::new();
    for &u in &order {
        for h in succs(u).into_iter().flatten() {
            if !ir_dominates(&idom, h, u) {
                continue;
            }
            let mut body = HashSet::new();
            body.insert(h);
            if u != h {
                let mut stack = vec![u];
                body.insert(u);
                while let Some(cur) = stack.pop() {
                    for &q in &p.blocks[cur].preds {
                        if reachable(q) && body.insert(q) {
                            stack.push(q);
                        }
                    }
                }
            }
            loops.push((h, body));
        }
    }
    loop {
        let mut changed = false;
        for (h, body) in &loops {
            let latch_divergent = p.blocks[*h]
                .preds
                .iter()
                .any(|&q| body.contains(&q) && !uniform[q]);
            let exit_divergent = body.iter().any(|&x| {
                is_divergent_branch(x)
                    && succs(x).into_iter().flatten().any(|s| !body.contains(&s))
            });
            if !uniform[*h] || latch_divergent || exit_divergent {
                for &x in body {
                    if std::mem::replace(&mut uniform[x], false) {
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    (uniform, idom, ipdom, order, index)
}

/// Plans workgroup-uniform predication for divergent branches that surround
/// wave64-emulated operations (`need_uniform`). Returns `Some(plan)` if every
/// required divergent branch can be converted to `EarlyReturn`, `OneSidedIf`,
/// or `DivergentLoop` so all needed blocks (and the uniformized branches
/// themselves) run in workgroup-uniform control flow.
pub(super) fn plan_uniform_branches(
    p: &Program,
    st: &Structured,
    need_uniform: &[bool],
) -> Option<(HashMap<usize, UniformBranch>, Vec<usize>)> {
    let n = p.blocks.len();
    let mut ignored = HashSet::new();
    let mut plan: HashMap<usize, UniformBranch> = HashMap::new();
    let is_trivial_end = |t: usize| {
        st.exits[t] == LaneExit::End
            && p.blocks[t].phis.is_empty()
            && p.blocks[t].insts.iter().all(|i| i.name() == "s_endpgm")
    };
    let succs = |b: usize| -> [Option<usize>; 2] {
        match st.exits[b] {
            LaneExit::End | LaneExit::Indirect => [None, None],
            LaneExit::Jump(t) => [Some(t), None],
            LaneExit::Branch {
                taken, not_taken, ..
            } => [Some(taken), Some(not_taken)],
        }
    };
    loop {
        let (uniform, idom, ipdom, order, rpo) = uniform_blocks_with(p, st, &ignored);
        let all_ok = (0..n).all(|b| (!need_uniform[b] && !plan.contains_key(&b)) || uniform[b]);
        if all_ok {
            return Some((plan, idom));
        }
        // Find divergent branches whose region (or loop) affects any still-non-uniform needed block.
        let mut added = false;
        for &br in &order {
            if ignored.contains(&br) {
                continue;
            }
            let LaneExit::Branch {
                when,
                taken,
                not_taken,
            } = st.exits[br]
            else {
                continue;
            };
            let is_div = st
                .bools
                .support(when)
                .iter()
                .any(|a| !matches!(a, Atom::Uniform(_)));
            if !is_div {
                continue;
            }
            // Check whether ignoring `br` makes progress or `br`'s region touches a needed non-uniform block.
            let stop = ipdom[br];
            let mut touches = false;
            let mut work = Vec::new();
            let mut visited = vec![false; n];
            for s in [taken, not_taken] {
                if s != stop && !visited[s] {
                    visited[s] = true;
                    work.push(s);
                }
            }
            while let Some(u) = work.pop() {
                if (need_uniform[u] || plan.contains_key(&u)) && !uniform[u] {
                    touches = true;
                    break;
                }
                for s in succs(u).into_iter().flatten() {
                    if s != stop && !visited[s] {
                        visited[s] = true;
                        work.push(s);
                    }
                }
            }
            // Also check if `br` is a divergent exit of a loop containing a needed non-uniform block.
            if !touches {
                for &u in &order {
                    if (need_uniform[u] || plan.contains_key(&u))
                        && !uniform[u]
                        && (ir_dominates(&idom, u, br) || ir_dominates(&idom, br, u))
                    {
                        touches = true;
                        break;
                    }
                }
            }
            if !touches {
                continue;
            }
            let kind = if is_trivial_end(taken) && !is_trivial_end(not_taken) && rpo[not_taken] > rpo[br] {
                Some(UniformBranch::EarlyReturn {
                    stay: not_taken,
                    ret: taken,
                })
            } else if is_trivial_end(not_taken) && !is_trivial_end(taken) && rpo[taken] > rpo[br] {
                Some(UniformBranch::EarlyReturn {
                    stay: taken,
                    ret: not_taken,
                })
            } else if rpo[taken] <= rpo[br] && rpo[not_taken] > rpo[br] && idom[taken] != usize::MAX && idom[taken] != taken {
                Some(UniformBranch::DivergentLoop {
                    header: taken,
                    exit: not_taken,
                })
            } else if rpo[not_taken] <= rpo[br] && rpo[taken] > rpo[br] && idom[not_taken] != usize::MAX && idom[not_taken] != not_taken {
                Some(UniformBranch::DivergentLoop {
                    header: not_taken,
                    exit: taken,
                })
            } else if rpo[taken] > rpo[br] && rpo[not_taken] > rpo[br] && taken == ipdom[br] && idom[not_taken] == br {
                Some(UniformBranch::OneSidedIf {
                    body: not_taken,
                    skip: taken,
                })
            } else if rpo[taken] > rpo[br] && rpo[not_taken] > rpo[br] && not_taken == ipdom[br] && idom[taken] == br {
                Some(UniformBranch::OneSidedIf {
                    body: taken,
                    skip: not_taken,
                })
            } else if rpo[taken] > rpo[br]
                && rpo[not_taken] > rpo[br]
                && stop < n
                && taken != stop
                && not_taken != stop
                && idom[taken] == br
                && idom[not_taken] == br
            {
                let check_arm = |start: usize| -> bool {
                    let mut seen = HashSet::new();
                    let mut stack = vec![start];
                    seen.insert(start);
                    while let Some(u) = stack.pop() {
                        if !ir_dominates(&idom, start, u) {
                            return false;
                        }
                        if matches!(st.exits[u], LaneExit::End | LaneExit::Indirect) {
                            return false;
                        }
                        for s in succs(u).into_iter().flatten() {
                            if s == stop {
                                continue;
                            }
                            if !ir_dominates(&idom, start, s) {
                                return false;
                            }
                            if seen.insert(s) {
                                stack.push(s);
                            }
                        }
                    }
                    true
                };
                if check_arm(taken) {
                    Some(UniformBranch::TwoSidedIf {
                        then_b: taken,
                        else_b: not_taken,
                        join: stop,
                    })
                } else if check_arm(not_taken) {
                    Some(UniformBranch::TwoSidedIf {
                        then_b: not_taken,
                        else_b: taken,
                        join: stop,
                    })
                } else {
                    None
                }
            } else {
                None
            };
            let Some(kind) = kind else {
                // The dominance check above can also select unrelated branches.
                // Try the remaining candidates before declaring the plan impossible.
                continue;
            };
            plan.insert(br, kind);
            ignored.insert(br);
            added = true;
        }
        if !added {
            return None;
        }
    }
}

impl<'a> Lower<'a> {
    /// Exchanges one `i32` per host wave32 subwave via the emulation TGSM pair
    /// for each console wave64; returns `(subwave0, has_hi ? subwave1 : fallback)`.
    pub(super) fn emu_exchange_pair(&mut self, at: usize, val: V, fallback: V) -> (V, V) {
        let w = self.console_wave();
        let sub = self.console_subwave();
        let has_hi = self.console_has_hi();
        let lds = self.s.lds();
        let base = self.s.m.i32(self.emu_lds_base);
        let one = self.s.m.i32(1);
        let w2 = self.s.m.bin(at, Bin::Shl, w, one);
        let w2_base = self.s.m.bin(at, Bin::Add, base, w2);
        let my_idx = self.s.m.bin(at, Bin::Add, w2_base, sub);
        let my_ptr = self.s.m.element(at, lds, my_idx);
        self.s.m.store(at, my_ptr, val);
        self.emit_lds_barrier_at(at);
        let p0 = self.s.m.element(at, lds, w2_base);
        let v0 = self.s.m.load(at, p0);
        let hi_idx = self.s.m.bin(at, Bin::Add, w2_base, one);
        let p1 = self.s.m.element(at, lds, hi_idx);
        let v1_raw = self.s.m.load(at, p1);
        let v1 = self.s.m.select(at, has_hi, v1_raw, fallback);
        self.emit_lds_barrier_at(at);
        (v0, v1)
    }

    /// 64-lane ballot `(lo, hi)` emulated across wave32 subwaves at block `at`.
    pub(super) fn emulate_ballot(&mut self, at: usize, bit: V) -> (V, V) {
        self.s.flags |= flags::WAVE_OPS;
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        let h = self.s.m.extract(at, r, 0);
        let z = self.s.m.i32(0);
        if self.emu_total <= 32 {
            (h, z)
        } else {
            self.emu_exchange_pair(at, h, z)
        }
    }

    /// Workgroup-wide `any(cond)` via 1 TGSM dword (`emu_wg_slot`) and
    /// `waveActiveBallot`, guaranteeing uniform control flow across all wave32
    /// subwaves in the workgroup. Advances `self.cur`.
    pub(super) fn wg_any_true(&mut self, cond: V) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let slot = self.emu_wg_slot.expect("emu_wg_slot");
        let lds = self.s.lds();
        let idx = self.s.m.i32(slot);
        let z = self.s.m.i32(0);
        let one = self.s.m.i32(1);
        let flat = self.console_flat();
        let is0 = self.s.m.cmp(self.cur, pred::EQ, flat, z);
        let g0 = self.begin_guarded(Some(is0));
        let ptr0 = self.s.m.element(self.cur, lds, idx);
        self.s.m.store(self.cur, ptr0, z);
        self.end_guarded(g0, &[]);
        self.emit_lds_barrier();
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let b = self
            .s
            .op(self.cur, "waveActiveBallot", Attr::NoUnwind, four, 116, &[cond]);
        let mask = self.s.m.extract(self.cur, b, 0);
        let w_any = self.s.m.cmp(self.cur, pred::NE, mask, z);
        let g1 = self.begin_guarded(Some(w_any));
        let ptr1 = self.s.m.element(self.cur, lds, idx);
        self.s.m.store(self.cur, ptr1, one);
        self.end_guarded(g1, &[]);
        self.emit_lds_barrier();
        let ptr2 = self.s.m.element(self.cur, lds, idx);
        let val = self.s.m.load(self.cur, ptr2);
        let res = self.s.m.cmp(self.cur, pred::NE, val, z);
        self.emit_lds_barrier();
        res
    }
    pub(super) fn emu_read_first_lane(&mut self, at: usize, a: V, guard: Option<V>) -> V {
        let z = self.s.m.i32(0);
        let lane = match guard {
            None => z,
            Some(g) => {
                let (lo, hi) = self.emulate_ballot(at, g);
                let i32t = self.s.t.i32;
                let k32 = self.s.m.i32(32);
                let flo = self.s.op(at, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[lo]);
                let fhi = self.s.op(at, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[hi]);
                let fhi32 = self.s.m.bin(at, Bin::Add, fhi, k32);
                let lo_nz = self.s.m.cmp(at, pred::NE, lo, z);
                let first = self.s.m.select(at, lo_nz, flo, fhi32);
                let both = self.s.m.bin(at, Bin::Or, lo, hi);
                let any_nz = self.s.m.cmp(at, pred::NE, both, z);
                self.s.m.select(at, any_nz, first, z)
            }
        };
        self.read_lane(a, lane)
    }
}

