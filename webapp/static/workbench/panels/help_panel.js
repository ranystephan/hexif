// Slide-out help panel (help system).
//
// Right-side panel that renders the markdown help pages from
// /api/help/{id}. The panel is dismissable via Esc / close button /
// outside click. Pages are loaded lazily on first tab click and
// cached per-session so re-opening is instant.
//
// External links rendered inside the panel are decorated with
// target="_blank" + rel="noopener noreferrer" by the *server-side*
// markdown renderer (see webapp/help.py::_install_link_target_blank).
// We don't post-process the injected HTML here.
//
// Internal anchors (href="#section") trigger a scroll within the
// panel's content container, *not* the document — the panel hijacks
// the click and calls scrollIntoView() on the resolved anchor element.

import { fetchHelpPages, fetchHelpPageHtml } from '../api.js';

/**
 * @typedef {Object} HelpPage
 * @property {string} id
 * @property {string} title
 */

/**
 * @typedef {Object} HelpPanelHandle
 * @property {() => void} open                     Open to the current/first page.
 * @property {(pageId: string, anchor?: string|null) => Promise<void>} openTo
 *   Open the panel to a specific page id; optionally scroll to ``#anchor``.
 * @property {() => void} close
 * @property {() => boolean} isOpen
 */

/**
 * Mount the help panel as a sibling of ``parent``. If a ``#help-panel``
 * already exists, reuse it (so calling mountHelpPanel twice is a no-op
 * — handy for hot-reload during dev).
 *
 * @param {HTMLElement} parent  Where to attach the panel (typically document.body).
 * @returns {HelpPanelHandle}
 */
export function mountHelpPanel(parent) {
  // Reuse path: if the help panel was already mounted (e.g., a previous
  // call from main.js boot), return a handle that drives the existing
  // DOM rather than rebuilding it.
  const existing = parent.querySelector('aside#help-panel');
  if (existing && existing.__hexifHelpHandle) {
    return existing.__hexifHelpHandle;
  }

  /** @type {HTMLElement} */
  const aside = existing || document.createElement('aside');
  aside.id = 'help-panel';
  aside.className = 'help-panel';
  aside.setAttribute('role', 'dialog');
  aside.setAttribute('aria-label', 'HEXIF Workbench help');
  aside.setAttribute('aria-modal', 'false');
  aside.hidden = true;

  // Header + tab strip + content container. Built once; tab clicks
  // mutate which tab is highlighted and which page is rendered.
  aside.innerHTML = `
    <div class="help-header">
      <strong class="help-title">Help</strong>
      <button type="button" class="help-close" aria-label="Close help panel">×</button>
    </div>
    <nav class="help-tabs" role="tablist" aria-label="Help pages"></nav>
    <div class="help-content" tabindex="0"></div>
  `;
  if (!existing) parent.appendChild(aside);

  /** @type {HTMLElement} */
  const tabsEl = aside.querySelector('.help-tabs');
  /** @type {HTMLElement} */
  const contentEl = aside.querySelector('.help-content');
  /** @type {HTMLButtonElement} */
  const closeBtn = aside.querySelector('.help-close');

  /** @type {HelpPage[]} */
  let pages = [];
  /** @type {string|null} */
  let activePageId = null;
  /** @type {Map<string, string>} */
  const htmlCache = new Map();
  let listLoaded = false;
  let listLoadPromise = null;

  function isOpen() {
    return !aside.hidden;
  }

  async function ensureList() {
    if (listLoaded) return;
    if (listLoadPromise) return listLoadPromise;
    listLoadPromise = (async () => {
      try {
        const resp = await fetchHelpPages();
        pages = (resp && Array.isArray(resp.pages)) ? resp.pages : [];
        renderTabs();
        listLoaded = true;
      } catch (err) {
        contentEl.textContent = `Could not load help index: ${err.message || err}`;
        // Why: a failed boot of the help index is surfaced to the
        // user instead of silently swallowed, because the help panel
        // itself is the user's only path to the docs.
        console.warn('fetchHelpPages failed:', err);
      } finally {
        listLoadPromise = null;
      }
    })();
    return listLoadPromise;
  }

  function renderTabs() {
    tabsEl.innerHTML = '';
    for (const p of pages) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'help-tab';
      btn.dataset.pageId = p.id;
      btn.setAttribute('role', 'tab');
      btn.setAttribute('aria-selected', String(p.id === activePageId));
      btn.textContent = p.title;
      btn.addEventListener('click', () => {
        if (p.id === activePageId) return;  // No-op: clicking the active tab.
        void selectTab(p.id);
      });
      tabsEl.appendChild(btn);
    }
  }

  function syncTabSelected() {
    for (const tab of tabsEl.querySelectorAll('.help-tab')) {
      tab.setAttribute('aria-selected', String(tab.dataset.pageId === activePageId));
    }
  }

  async function selectTab(pageId, anchor) {
    activePageId = pageId;
    syncTabSelected();
    // Cache hit — render immediately so the user sees no flash.
    if (htmlCache.has(pageId)) {
      contentEl.innerHTML = htmlCache.get(pageId);
      scrollToAnchor(anchor);
      return;
    }
    contentEl.textContent = 'Loading…';
    try {
      const html = await fetchHelpPageHtml(pageId);
      htmlCache.set(pageId, html);
      // Defensive guard: ensure the user didn't switch tabs while we
      // were fetching this one.
      if (activePageId === pageId) {
        contentEl.innerHTML = html;
        attachAnchorHandlers();
        scrollToAnchor(anchor);
      }
    } catch (err) {
      contentEl.textContent = `Could not load help page "${pageId}": ${err.message || err}`;
      console.warn(`fetchHelpPageHtml(${pageId}) failed:`, err);
    }
  }

  function scrollToAnchor(anchor) {
    if (!anchor) {
      // Default: scroll the content container to the top so a fresh
      // page open lands at the H1, not wherever the previous read left
      // the scroll offset.
      contentEl.scrollTop = 0;
      return;
    }
    const id = anchor.startsWith('#') ? anchor.slice(1) : anchor;
    // Use #-escape-safe ID selector to handle ids with leading digits etc.
    const target = contentEl.querySelector(`[id="${cssEscape(id)}"]`);
    if (target && typeof target.scrollIntoView === 'function') {
      target.scrollIntoView({ behavior: 'instant', block: 'start' });
    } else {
      // The anchor doesn't exist on this page — fall back to top so the
      // user sees something useful instead of stale state.
      contentEl.scrollTop = 0;
    }
  }

  function cssEscape(s) {
    // The native CSS.escape isn't available in older WebKit builds we
    // still support; fall back to a manual escaper for the small set
    // of chars that appear in markdown-it slugs (digits, hyphens,
    // underscores, lowercase letters — already safe — only quotes
    // and backslashes need escaping for the attribute-selector form).
    if (window.CSS && typeof window.CSS.escape === 'function') return window.CSS.escape(s);
    return String(s).replace(/["\\]/g, '\\$&');
  }

  function attachAnchorHandlers() {
    // Hijack in-panel anchor clicks so they scroll the content
    // container rather than the document. External links (target=
    // _blank, set server-side) are not touched — their default click
    // semantics still apply.
    const links = contentEl.querySelectorAll('a[href^="#"]');
    for (const a of links) {
      a.addEventListener('click', (ev) => {
        ev.preventDefault();
        const href = a.getAttribute('href') || '';
        scrollToAnchor(href);
      });
    }
  }

  function open() {
    if (!aside.hidden && aside.classList.contains('open')) return;
    aside.hidden = false;
    // Force a reflow so the transition kicks in even when we just
    // unhid the element on the same frame.
    void aside.offsetWidth;
    aside.classList.add('open');
    document.addEventListener('keydown', onKeyDown, true);
    document.addEventListener('mousedown', onDocMouseDown, true);
    void ensureList().then(() => {
      if (!activePageId && pages.length > 0) {
        void selectTab(pages[0].id);
      }
    });
  }

  async function openTo(pageId, anchor) {
    open();
    await ensureList();
    if (!pages.length) return;
    // If the requested id isn't present, fall back to the first page
    // and leave the anchor empty (a deep-link to a missing page is
    // worth not crashing — the user still sees the docs index).
    const target = pages.find((p) => p.id === pageId) ? pageId : pages[0].id;
    await selectTab(target, anchor || null);
  }

  function close() {
    if (aside.hidden) return;
    aside.classList.remove('open');
    // Match the CSS transition duration (0.2s) before hiding so the
    // slide-out animation actually plays.
    setTimeout(() => {
      if (!aside.classList.contains('open')) aside.hidden = true;
    }, 220);
    document.removeEventListener('keydown', onKeyDown, true);
    document.removeEventListener('mousedown', onDocMouseDown, true);
  }

  function onKeyDown(ev) {
    if (ev.key !== 'Escape') return;
    // UX convention: Esc closes the topmost popover only. The color-by
    // popover in panels/controls.js is a sibling capture-phase listener
    // on document — without this guard both listeners fire on the same
    // keypress and both popovers close at once. The color-by popover's
    // open state is encoded as `hidden=false` on a `.cb-popover` node;
    // if one is present, let its handler take Esc and bail here.
    if (document.querySelector('.cb-popover:not([hidden])')) return;
    ev.stopPropagation();
    close();
  }

  function onDocMouseDown(ev) {
    if (aside.contains(ev.target)) return;
    // Ignore clicks on the discoverability ? icons — they open the
    // panel themselves; the resulting capture-phase mousedown should
    // not also close the panel before the click handler runs.
    const opener = ev.target.closest && ev.target.closest('[data-help-anchor]');
    if (opener) return;
    close();
  }

  closeBtn.addEventListener('click', close);

  const handle = { open, openTo, close, isOpen };
  aside.__hexifHelpHandle = handle;
  return handle;
}
