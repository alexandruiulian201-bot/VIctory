//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the subresource footprints of the host
/// texture, and what was last detiled (with a transient upload buffer alive
/// only while a dirty texture awaits its prelude GPU copy).
pub(crate) struct Staging {
    total: u64,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// Whether `memory` is write-watched: 0 unknown, 1 yes, 2 no.
    watched: AtomicU8,
    /// Whether the staging buffer has been GPU-copied into the host texture at least once.
    uploaded: AtomicBool,
    last: Mutex<Last>,
}

const NOT_WATCHED: u8 = 2;

/// What a texture's staging state holds: the hash of the memory detiled
/// (memory not watched), the write-watch asking it was detiled after, and
/// the transient upload buffer populated for the current submission prelude.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    asked: u64,
    upload: Option<ID3D12Resource>,
}

const MAX_READ_SCRATCH: usize = 4 << 20;

/// Single shared temporary buffer for copying write-watched tiled texture
/// memory before detiling, instead of pinning one `Vec<u8>` per texture.
static READ_SCRATCH: Mutex<Vec<u8>> = Mutex::new(Vec::new());

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    set.get_or_insert_with(Default::default)
        .insert(resource as usize);
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Defers a texture's staging refresh and GPU copy to submission time: only
/// textures whose contents changed (or that have not yet been uploaded) record
/// `CopyTextureRegion` in the submission's prelude, skipping PCIe copies and
/// `COPY_DEST` transitions for static textures.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    let host_written = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if host_written || !list.first_use(resource as usize) {
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    let Some(copy) = crate::resource::host_copy(resource) else {
        return;
    };
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging, copy);
}

impl Staging {
    /// Records `CopyTextureRegion` for all subresources from the transient
    /// staging buffer into `texture` (which must already be in `COPY_DEST`),
    /// returning the upload buffer to keep alive until the prelude finishes.
    pub(crate) fn record_copy(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        texture: &ID3D12Resource,
    ) -> Option<ID3D12Resource> {
        let buffer = self
            .last
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .upload
            .take()?;
        for (i, footprint) in self.footprints.iter().enumerate() {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: i as u32,
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(&buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: *footprint,
                },
            };
            // SAFETY: an open prelude command list; both locations name live resources.
            unsafe { cmd.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
        self.uploaded.store(true, Ordering::Release);
        Some(buffer)
    }

    /// Detiles the texture memory into a transient staging buffer if it changed
    /// since the last time (and is mapped at all). Returns `true` if the host
    /// texture has a staged upload ready for `CopyTextureRegion` in the prelude.
    pub(crate) fn refresh(&self) -> bool {
        let started = std::time::Instant::now();
        let (read, ready) = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        preparing[3].fetch_add(read as u64, Ordering::Relaxed);
        ready
    }

    /// Returns `(bytes_read, upload_ready)`.
    fn refresh_now(&self) -> (usize, bool) {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(start, start + size, false) {
                self.watched.store(1, Ordering::Relaxed);
                if last.asked != 0
                    && written.iter().all(|&(_, w)| w <= last.asked)
                    && (self.uploaded.load(Ordering::Acquire) || last.upload.is_some())
                {
                    return (0, last.upload.is_some());
                }
                // Read through the copy, which stands unmapped pages: memory of
                // big pools is too slow to ask page by page.
                let mut scratch = READ_SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
                scratch.resize(size, 0);
                crate::host::copy_mapped(scratch.as_mut_ptr(), self.memory, size, false);
                let uploaded = self.detile(&scratch);
                if scratch.capacity() > MAX_READ_SCRATCH {
                    *scratch = Vec::new();
                }
                let Some(buf) = uploaded else {
                    return (0, last.upload.is_some());
                };
                last.asked = now;
                last.upload = Some(buf);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
                return (size, true);
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return (0, last.upload.is_some());
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash)
            && (self.uploaded.load(Ordering::Acquire) || last.upload.is_some())
        {
            return (size, last.upload.is_some());
        }
        let Some(buf) = self.detile(tiled) else {
            return (size, last.upload.is_some());
        };
        last.hash = Some(hash);
        last.upload = Some(buf);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        (size, true)
    }

    /// Allocates a transient upload buffer and detiles `tiled` into it.
    fn detile(&self, tiled: &[u8]) -> Option<ID3D12Resource> {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        let (buffer, data) = host()?.upload_buffer(self.total)?;
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear =
                unsafe { std::slice::from_raw_parts_mut(data.add(footprint.Offset as usize), len) };
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return None;
            }
        }
        Some(buffer)
    }
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let s = Arc::new(Staging {
        total,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        watched: AtomicU8::new(0),
        uploaded: AtomicBool::new(false),
        last: Mutex::default(),
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}
