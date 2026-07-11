import { useState, useEffect, useMemo, useRef } from 'react';
import {
  Button,
  Empty,
  Input,
  Segmented,
  Tooltip,
  message
} from 'antd';
import {
  CommentOutlined,
  HighlightOutlined,
  LinkOutlined,
  CloudDownloadOutlined,
  ReloadOutlined,
  RightOutlined,
  DownOutlined
} from '@ant-design/icons';
import { EditorContent, useEditor } from '@tiptap/react';
import { BubbleMenu } from '@tiptap/react/menus';
import { Extension } from '@tiptap/core';
import StarterKit from '@tiptap/starter-kit';
import Highlight from '@tiptap/extension-highlight';
import Underline from '@tiptap/extension-underline';
import Link from '@tiptap/extension-link';
import Placeholder from '@tiptap/extension-placeholder';
import { TextStyle } from '@tiptap/extension-text-style';
import { Color } from '@tiptap/extension-color';
import {
  apiJson,
  reportSeedHtml,
  timeText,
  downloadBlob,
  sourceTitle,
  signalEvidenceGraph
} from '../utils/helpers';
import {
  citationMarkAttrs,
  editorJsonToReportIr,
  reportIrOutline,
  reportIrToEditorJson
} from '../utils/reportIrEditor';

function slugify(value, fallback) {
  const slug = String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  return slug || fallback;
}

function removeDuplicateOverviewChapter(doc) {
  const main = doc.querySelector('main');
  if (!main) return;
  const firstChapter = Array.from(main.children).find((element) => element.classList?.contains('chapter'));
  if (!firstChapter) return;
  const text = String(firstChapter.textContent || '').replace(/\s+/g, ' ').trim().toLowerCase();
  const hasOverview = text.includes('report overview:');
  const hasScaffold = (text.includes('summary and highlights') || text.includes('hero summary'))
    && (text.includes('performance indicators') || text.includes('key performance indicators') || text.includes('recommended actions') || text.includes('actions'));
  if (hasOverview && hasScaffold) firstChapter.remove();
}

function normalizeHeroActions(doc) {
  doc.querySelectorAll('.hero-section-combined .hero-actions').forEach((actions) => {
    const buttons = Array.from(actions.querySelectorAll('button'));
    if (!buttons.length) return;
    const replacement = doc.createElement('div');
    replacement.className = 'hero-actions';
    const label = doc.createElement('span');
    label.className = 'hero-actions-label';
    label.textContent = 'Recommended Follow-up';
    const list = doc.createElement('ul');
    buttons.forEach((button) => {
      const item = doc.createElement('li');
      item.textContent = String(button.textContent || '').trim();
      list.appendChild(item);
    });
    replacement.append(label, list);
    actions.replaceWith(replacement);
  });
}

function injectPreviewOverrides(doc) {
  const style = doc.createElement('style');
  style.setAttribute('data-review-preview-overrides', 'true');
  style.textContent = [
    '.header-actions, #theme-toggle-btn, #print-btn, #export-btn, nav.toc, .toc, .report-header .tagline { display: none !important; }',
    '.report-header { justify-content: flex-start !important; }',
    '.report-header .subtitle { color: rgba(33,37,41,.66) !important; }',
    '.hero-section-combined { display: grid; gap: 22px; padding: clamp(28px, 5vw, 52px); margin-bottom: 34px; border: 1px solid var(--border-color, #dee2e6); border-radius: 18px; background: var(--card-bg, #fff); box-shadow: 0 14px 34px rgba(0,0,0,.06); }',
    '.hero-header { display: grid; gap: 8px; padding-bottom: 18px; border-bottom: 1px solid var(--border-color, #dee2e6); }',
    '.hero-hint { margin: 0; color: rgba(33,37,41,.62); font-size: .82rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }',
    '.hero-title { margin: 0; color: var(--text-color, #212529); font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.08; letter-spacing: 0; }',
    '.hero-subtitle { margin: 0; max-width: 880px; color: rgba(33,37,41,.68); font-size: 1.02rem; line-height: 1.5; }',
    '.hero-body { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(240px, .65fr); gap: 22px; align-items: start; }',
    '.hero-section-combined .hero-content { display: grid; gap: 16px; min-width: 0; }',
    '.hero-section-combined .hero-side { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; min-width: 0; margin-top: 0; }',
    '.hero-overview-card { display: grid; gap: 8px; padding: 16px 18px; border: 1px solid rgba(0,0,0,.06); border-radius: 14px; background: rgba(0,0,0,.018); }',
    '.hero-overview-label, .hero-actions-label { color: rgba(33,37,41,.62); font-size: .78rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }',
    '.hero-overview-card .hero-summary { margin: 0; color: var(--text-color, #212529); font-size: 1.02rem; font-weight: 400; line-height: 1.68; }',
    '.hero-section-combined .hero-actions { display: grid; gap: 8px; padding: 14px 16px; border: 1px solid rgba(0,0,0,.06); border-radius: 14px; background: rgba(0,0,0,.014); }',
    '.hero-actions ul { margin: 0; padding-left: 18px; }',
    '.hero-actions li { margin: 4px 0; }',
    '@media (max-width: 900px) { .hero-body { grid-template-columns: 1fr; } }'
  ].join('\n');
  doc.head.appendChild(style);
}

function normalizeReportHtml(html) {
  if (!html || typeof window === 'undefined' || !window.DOMParser) return html || '';
  try {
    const doc = new window.DOMParser().parseFromString(html, 'text/html');
    doc.querySelectorAll('#theme-toggle-btn, #print-btn, #export-btn, .header-actions, nav.toc, .toc, .report-header .tagline').forEach((element) => element.remove());
    removeDuplicateOverviewChapter(doc);
    normalizeHeroActions(doc);
    injectPreviewOverrides(doc);
    doc.querySelectorAll('h1, h2, h3').forEach((heading, index) => {
      if (!heading.id) heading.id = slugify(heading.textContent, 'report-section-' + index);
    });
    return '<!doctype html>\n' + doc.documentElement.outerHTML;
  } catch {
    return html;
  }
}

function extractReportOutline(html) {
  if (!html || typeof window === 'undefined' || !window.DOMParser) return [];
  try {
    const doc = new window.DOMParser().parseFromString(html, 'text/html');
    return Array.from(doc.querySelectorAll('h1, h2, h3'))
      .filter((heading) => !heading.closest('.report-header, .toc, .hero-section-combined'))
      .map((heading, index) => ({
        id: heading.id || slugify(heading.textContent, 'report-section-' + index),
        level: heading.tagName.toLowerCase(),
        title: String(heading.textContent || '').replace(/\s+/g, ' ').trim()
      }))
      .filter((item) => item.title)
      .slice(0, 80);
  } catch {
    return [];
  }
}

function outlineRankFromTitle(title) {
  const match = String(title || '').trim().match(/^(\d+(?:\.\d+)*)(?:\.|\s)/);
  if (!match) return null;
  const depth = match[1].split('.').filter(Boolean).length;
  return Math.min(6, Math.max(2, depth + 1));
}

function outlineRank(level, title) {
  const inferred = outlineRankFromTitle(title);
  if (inferred) return inferred;
  const rank = Number(String(level || '').replace(/[^0-9]/g, ''));
  return rank >= 1 && rank <= 6 ? rank : 2;
}

function buildOutlineTree(items) {
  const roots = [];
  const stack = [];
  (items || []).forEach((item, index) => {
    const rank = outlineRank(item.level, item.title);
    const node = {
      ...item,
      rank,
      key: item.irPath || item.id || 'outline-' + index,
      children: []
    };
    while (stack.length && stack[stack.length - 1].rank >= rank) stack.pop();
    if (stack.length) stack[stack.length - 1].children.push(node);
    else roots.push(node);
    stack.push(node);
  });
  return roots;
}

function buildCitationSources(output) {
  const graph = signalEvidenceGraph(output);
  const sources = [
    ...(output?.source_data?.query_agent?.top_sources || []),
    ...(output?.coordinator_intelligence?.report_contract?.top_sources || []),
    ...(graph?.evidence_items || [])
  ];
  const seen = new Set();
  return sources.filter((source) => {
    const key = String(source?.url || source?.citation_span_id || source?.canonical_item_id || source?.title || '').trim();
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

const IrBlockAttributes = Extension.create({
  name: 'irBlockAttributes',
  addGlobalAttributes() {
    return [{
      types: ['paragraph', 'heading', 'bulletList', 'orderedList', 'listItem', 'blockquote', 'codeBlock', 'horizontalRule'],
      attributes: {
        irPath: {
          default: null,
          parseHTML: (element) => element.getAttribute('data-ir-path'),
          renderHTML: (attributes) => attributes.irPath ? { 'data-ir-path': attributes.irPath } : {}
        },
        irBlockType: {
          default: null,
          parseHTML: (element) => element.getAttribute('data-ir-block-type'),
          renderHTML: (attributes) => attributes.irBlockType ? { 'data-ir-block-type': attributes.irBlockType } : {}
        },
        irLocked: {
          default: false,
          parseHTML: (element) => element.getAttribute('data-ir-locked') === 'true',
          renderHTML: (attributes) => attributes.irLocked ? { 'data-ir-locked': 'true', contenteditable: 'false', class: 'ir-locked-block' } : {}
        }
      }
    }];
  }
});

const CitationLink = Link.extend({
  addAttributes() {
    const parentAttributes = this.parent?.() || {};
    return {
      ...parentAttributes,
      citationId: {
        default: null,
        parseHTML: (element) => element.getAttribute('data-citation-id'),
        renderHTML: (attributes) => attributes.citationId ? { 'data-citation-id': attributes.citationId } : {}
      },
      citationIndex: {
        default: null,
        parseHTML: (element) => element.getAttribute('data-citation-index'),
        renderHTML: (attributes) => attributes.citationIndex ? { 'data-citation-index': attributes.citationIndex } : {}
      }
    };
  }
});

export default function ReviewEditor({ output, reportHtml, onReportHtmlChange, documentIr, onDocumentIrChange, annotations, setAnnotations, hasGeneratedReport, reportTaskId }) {
  const [commentDraft, setCommentDraft] = useState('');
  const [bubbleNoteOpen, setBubbleNoteOpen] = useState(false);
  const [pendingSelection, setPendingSelection] = useState(null);
  const [viewMode, setViewMode] = useState(hasGeneratedReport ? 'preview' : 'edit');
  const [reportPage, setReportPage] = useState({ current: 1, total: 1 });
  const [citationQuery, setCitationQuery] = useState('');
  const [citationPage, setCitationPage] = useState(1);
  const [renderingPreview, setRenderingPreview] = useState(false);
  const [expandedOutlineKeys, setExpandedOutlineKeys] = useState(() => new Set());
  const reportFrameRef = useRef(null);
  const structuredEditRef = useRef(false);
  const documentIrRef = useRef(documentIr || null);
  const loadedEditorSourceRef = useRef('');

  const canPreview = Boolean(hasGeneratedReport && reportHtml);
  const canStructuredEdit = Boolean(hasGeneratedReport && documentIr);
  const normalizedReportHtml = useMemo(() => canPreview ? normalizeReportHtml(reportHtml) : '', [canPreview, reportHtml]);
  const htmlOutline = useMemo(() => canPreview ? extractReportOutline(normalizedReportHtml) : [], [canPreview, normalizedReportHtml]);
  const irOutline = useMemo(() => canStructuredEdit ? reportIrOutline(documentIr) : [], [canStructuredEdit, documentIr]);
  const reportOutline = canPreview ? htmlOutline : irOutline;
  const outlineTree = useMemo(() => buildOutlineTree(reportOutline), [reportOutline]);
  const citationSources = useMemo(() => buildCitationSources(output), [output]);
  const citationItems = useMemo(() => citationSources.map((source, index) => ({ source, sourceIndex: index })), [citationSources]);
  const filteredCitationItems = useMemo(() => {
    const query = citationQuery.trim().toLowerCase();
    if (!query) return citationItems;
    return citationItems.filter(({ source, sourceIndex }) => {
      const text = [sourceTitle(source, sourceIndex), source?.url, source?.platform, source?.source_type, source?.citation_span_id].join(' ').toLowerCase();
      return text.includes(query);
    });
  }, [citationItems, citationQuery]);
  const citationPageSize = 8;
  const citationPageCount = Math.max(1, Math.ceil(filteredCitationItems.length / citationPageSize));
  const safeCitationPage = Math.min(citationPage, citationPageCount);
  const visibleCitationItems = filteredCitationItems.slice((safeCitationPage - 1) * citationPageSize, safeCitationPage * citationPageSize);

  const initialEditorContent = canStructuredEdit ? reportIrToEditorJson(documentIr) : (hasGeneratedReport ? reportSeedHtml(output) : (reportHtml || reportSeedHtml(output)));

  const editor = useEditor({
    extensions: [
      StarterKit.configure({ link: false, underline: false }),
      IrBlockAttributes,
      Underline,
      TextStyle,
      Color.configure({ types: ['textStyle'] }),
      Highlight.configure({ multicolor: true }),
      CitationLink.configure({ openOnClick: true, autolink: true, linkOnPaste: true }),
      Placeholder.configure({ placeholder: 'Draft, revise, cite.' })
    ],
    content: initialEditorContent,
    editorProps: {
      attributes: {
        class: 'review-editor-content',
        spellcheck: 'false',
        autocorrect: 'off',
        autocapitalize: 'off',
        autocomplete: 'off',
        'data-gramm': 'false',
        'data-gramm_editor': 'false',
        'data-enable-grammarly': 'false',
        'data-lt-active': 'false'
      }
    },
    onUpdate: ({ editor: current }) => {
      if (structuredEditRef.current && documentIrRef.current) {
        const nextIr = editorJsonToReportIr(current.getJSON(), documentIrRef.current);
        documentIrRef.current = nextIr;
        onDocumentIrChange?.(nextIr);
        return;
      }
      onReportHtmlChange(current.getHTML());
    }
  });

  useEffect(() => {
    structuredEditRef.current = canStructuredEdit;
    documentIrRef.current = documentIr || documentIrRef.current;
  }, [canStructuredEdit, documentIr]);

  useEffect(() => {
    if (canPreview) setViewMode('preview');
    else setViewMode('edit');
    loadedEditorSourceRef.current = '';
  }, [canPreview, reportTaskId]);

  useEffect(() => {
    setCitationPage(1);
  }, [citationQuery, citationSources.length]);

  useEffect(() => {
    setExpandedOutlineKeys(new Set());
  }, [reportTaskId, reportOutline.length]);

  useEffect(() => {
    if (!editor) return;
    if (canStructuredEdit && documentIr) {
      const key = 'ir:' + (reportTaskId || 'latest');
      if (loadedEditorSourceRef.current !== key) {
        editor.commands.setContent(reportIrToEditorJson(documentIr), false);
        loadedEditorSourceRef.current = key;
      }
      return;
    }
    const key = hasGeneratedReport ? 'html-fallback:' + (reportTaskId || 'latest') : 'draft';
    const next = hasGeneratedReport ? reportSeedHtml(output) : (reportHtml || reportSeedHtml(output));
    if (loadedEditorSourceRef.current !== key || next !== editor.getHTML()) {
      editor.commands.setContent(next, false);
      loadedEditorSourceRef.current = key;
    }
  }, [editor, output, reportHtml, hasGeneratedReport, canStructuredEdit, documentIr, reportTaskId]);

  const updateReportPage = () => {
    const doc = reportFrameRef.current?.contentDocument;
    const root = doc?.scrollingElement || doc?.documentElement;
    if (!root || !root.clientHeight) return;
    const total = Math.max(1, Math.ceil(root.scrollHeight / root.clientHeight));
    const current = Math.min(total, Math.max(1, Math.floor(root.scrollTop / root.clientHeight) + 1));
    setReportPage({ current, total });
  };

  const handleReportLoad = () => {
    const doc = reportFrameRef.current?.contentDocument;
    const root = doc?.scrollingElement || doc?.documentElement;
    if (!root) return;
    root.onscroll = updateReportPage;
    window.setTimeout(updateReportPage, 80);
  };

  const goToReportPage = (delta) => {
    const doc = reportFrameRef.current?.contentDocument;
    const root = doc?.scrollingElement || doc?.documentElement;
    if (!root || !root.clientHeight) return;
    const next = Math.min(reportPage.total, Math.max(1, reportPage.current + delta));
    root.scrollTo({ top: (next - 1) * root.clientHeight, behavior: 'smooth' });
  };

  const jumpToOutline = (item) => {
    if (!item?.id) return;
    if (viewMode !== 'preview') setViewMode('preview');
    window.setTimeout(() => {
      const doc = reportFrameRef.current?.contentDocument;
      const target = doc?.getElementById(item.id);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 80);
  };

  const toggleOutlineNode = (key) => {
    setExpandedOutlineKeys((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const renderOutlineNodes = (nodes) => nodes.map((item) => {
    const hasChildren = item.children?.length > 0;
    const expanded = expandedOutlineKeys.has(item.key);
    return (
      <div className={'outline-node depth-' + Math.min(3, Math.max(1, item.rank - 1))} key={item.key}>
        <div className="outline-row">
          <button
            type="button"
            className={'outline-toggle' + (hasChildren ? '' : ' empty')}
            onClick={() => hasChildren && toggleOutlineNode(item.key)}
            aria-label={expanded ? 'Collapse section' : 'Expand section'}
            aria-expanded={hasChildren ? expanded : undefined}
            disabled={!hasChildren}
          >
            {hasChildren ? (expanded ? <DownOutlined /> : <RightOutlined />) : null}
          </button>
          <button type="button" className={'outline-link ' + item.level} onClick={() => jumpToOutline(item)}>{item.title}</button>
        </div>
        {hasChildren && expanded && <div className="outline-children">{renderOutlineNodes(item.children)}</div>}
      </div>
    );
  });

  const renderStructuredPreview = async () => {
    if (!canStructuredEdit || !documentIrRef.current) return reportHtml || '';
    setRenderingPreview(true);
    try {
      const data = await apiJson('/api/report/render-ir', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_ir: documentIrRef.current })
      });
      if (data.document_ir) {
        documentIrRef.current = data.document_ir;
        onDocumentIrChange?.(data.document_ir);
      }
      if (data.html_content) onReportHtmlChange(data.html_content);
      return data.html_content || '';
    } catch (error) {
      message.error(error.message || 'Could not render edited report');
      return reportHtml || '';
    } finally {
      setRenderingPreview(false);
    }
  };

  const handleModeChange = async (value) => {
    if (value === 'preview' && canStructuredEdit) await renderStructuredPreview();
    setViewMode(value);
  };

  const getSelection = () => {
    if (!editor) return null;
    const { from, to } = editor.state.selection;
    const quote = editor.state.doc.textBetween(from, to, ' ').trim();
    return { from, to, quote };
  };

  const captureSelection = () => {
    const selection = getSelection();
    if (!selection || selection.from === selection.to || !selection.quote) {
      message.warning('Select report text first');
      return null;
    }
    setPendingSelection(selection);
    return selection;
  };

  const highlightSelection = (color = '#d9f99d') => {
    if (!editor) return;
    const selection = captureSelection();
    if (!selection) return;
    editor.chain().focus().setTextSelection({ from: selection.from, to: selection.to }).setHighlight({ color }).run();
  };

  const addAnnotation = (noteText = commentDraft) => {
    if (!editor) return;
    const selection = pendingSelection || getSelection();
    const note = String(noteText || '').trim();
    if (!selection || selection.from === selection.to || !selection.quote) {
      message.warning('Select report text first');
      return;
    }
    editor.chain().focus().setTextSelection({ from: selection.from, to: selection.to }).setHighlight({ color: '#d9f99d' }).run();
    setAnnotations((items) => [{
      id: 'note_' + Date.now(),
      quote: selection.quote,
      note: note || 'Marked for review',
      createdAt: new Date().toISOString()
    }, ...items]);
    setCommentDraft('');
    setBubbleNoteOpen(false);
    setPendingSelection(null);
  };

  const openInlineNote = () => {
    const selection = captureSelection();
    if (selection) setBubbleNoteOpen(true);
  };

  const setLink = () => {
    if (!editor) return;
    const selection = captureSelection();
    if (!selection) return;
    const previousUrl = editor.getAttributes('link').href || '';
    const url = window.prompt('Source URL', previousUrl);
    if (url === null) return;
    if (!url) {
      editor.chain().focus().setTextSelection({ from: selection.from, to: selection.to }).extendMarkRange('link').unsetLink().run();
      return;
    }
    editor.chain().focus().setTextSelection({ from: selection.from, to: selection.to }).extendMarkRange('link').setLink({ href: url, target: '_blank', rel: 'noopener noreferrer nofollow' }).run();
  };

  const insertCitation = (source, sourceIndex) => {
    if (!editor) return;
    const label = '[' + (sourceIndex + 1) + ']';
    const title = sourceTitle(source, sourceIndex);
    const attrs = citationMarkAttrs(source, sourceIndex, title);
    const { from, to } = editor.state.selection;
    const marker = attrs.href ? { type: 'text', text: label, marks: [{ type: 'link', attrs }] } : { type: 'text', text: label };

    setViewMode('edit');
    if (from !== to && attrs.href) {
      editor.chain().focus().setTextSelection({ from, to }).setLink(attrs).setTextSelection(to).insertContent([{ type: 'text', text: ' ' }, marker]).run();
    } else {
      editor.chain().focus().insertContent([{ type: 'text', text: ' ' }, marker]).run();
    }
    message.success('Inserted citation ' + label);
  };

  const exportEditedHtml = async () => {
    const html = canStructuredEdit ? await renderStructuredPreview() : (editor?.getHTML() || '');
    downloadBlob(html || '', canStructuredEdit ? 'reviewed-report.html' : 'reviewed-draft.html');
  };

  return (
    <div className="review-shell">
      <div className="editor-surface">
        <div className="editor-toolbar">
          <Segmented
            size="small"
            value={canPreview && viewMode === 'preview' ? 'preview' : 'edit'}
            onChange={handleModeChange}
            options={[{ label: 'Preview', value: 'preview', disabled: !canPreview }, { label: 'Edit', value: 'edit' }]}
          />
          <Tooltip title="Heading"><Button onClick={() => editor?.chain().focus().toggleHeading({ level: 2 }).run()}>H2</Button></Tooltip>
          <Tooltip title="Subhead"><Button onClick={() => editor?.chain().focus().toggleHeading({ level: 3 }).run()}>H3</Button></Tooltip>
          <Tooltip title="Bold"><Button icon={<strong>B</strong>} onClick={() => editor?.chain().focus().toggleBold().run()} /></Tooltip>
          <Tooltip title="Italic"><Button icon={<em>I</em>} onClick={() => editor?.chain().focus().toggleItalic().run()} /></Tooltip>
          <Tooltip title="Underline"><Button icon={<u>U</u>} onClick={() => editor?.chain().focus().toggleUnderline().run()} /></Tooltip>
          <Tooltip title="Highlight"><Button icon={<HighlightOutlined />} onClick={() => highlightSelection('#d9f99d')} /></Tooltip>
          <Tooltip title="Source link"><Button icon={<LinkOutlined />} onClick={setLink} /></Tooltip>
          <Tooltip title="Accent text"><Button className="swatch-button green" onClick={() => editor?.chain().focus().setColor('var(--hku)').run()} /></Tooltip>
          {canStructuredEdit && <Tooltip title="Refresh rendered preview"><Button icon={<ReloadOutlined />} loading={renderingPreview} onClick={renderStructuredPreview}>Render</Button></Tooltip>}
          <Tooltip title="Export edited HTML"><Button icon={<CloudDownloadOutlined />} onClick={exportEditedHtml}>Export</Button></Tooltip>
          {canPreview && viewMode === 'preview' && (
            <div className="report-page-controls">
              <Button size="small" onClick={() => goToReportPage(-1)} disabled={reportPage.current <= 1}>Prev</Button>
              <span>{reportPage.current} / {reportPage.total}</span>
              <Button size="small" onClick={() => goToReportPage(1)} disabled={reportPage.current >= reportPage.total}>Next</Button>
            </div>
          )}
        </div>
        {canPreview && viewMode === 'preview' ? (
          <div className="report-preview-shell">
            <iframe ref={reportFrameRef} className="report-preview-frame" title="Rendered report preview" srcDoc={normalizedReportHtml} onLoad={handleReportLoad} />
          </div>
        ) : (
          <div className="editor-scroll">
            {editor && (
              <BubbleMenu
                editor={editor}
                options={{ placement: 'top', strategy: 'absolute' }}
                shouldShow={({ editor: activeEditor, state }) => activeEditor.isEditable && !state.selection.empty}
              >
                <div className="selection-menu">
                  <button type="button" onMouseDown={(event) => event.preventDefault()} onClick={() => highlightSelection('#d9f99d')}><HighlightOutlined /> Mark</button>
                  <button type="button" onMouseDown={(event) => event.preventDefault()} onClick={openInlineNote}><CommentOutlined /> Note</button>
                  <button type="button" onMouseDown={(event) => event.preventDefault()} onClick={setLink}><LinkOutlined /> Cite</button>
                  {bubbleNoteOpen && (
                    <div className="bubble-note">
                      <Input.TextArea value={commentDraft} onChange={(event) => setCommentDraft(event.target.value)} placeholder="Note" autoSize={{ minRows: 2, maxRows: 4 }} />
                      <Button size="small" type="primary" onClick={() => addAnnotation(commentDraft)}>Add</Button>
                    </div>
                  )}
                </div>
              </BubbleMenu>
            )}
            <EditorContent editor={editor} />
          </div>
        )}
      </div>
      <aside className="annotation-panel">
        {reportOutline.length > 0 && (
          <div className="outline-panel">
            <div className="citation-head">
              <h3>Outline</h3>
              <span>{reportOutline.length}</span>
            </div>
            <div className="outline-list">
              {renderOutlineNodes(outlineTree)}
            </div>
          </div>
        )}
        <div className="annotation-compose compact-note-head">
          <h3>Notes</h3>
          <div className="note-count"><CommentOutlined /> {annotations.length}</div>
        </div>
        <div className="annotation-list">
          {annotations.length === 0 && <Empty description="No notes yet" />}
          {annotations.map((item) => (
            <div className="annotation-card" key={item.id}>
              <span>{timeText(item.createdAt)}</span>
              <strong>{item.quote || 'General note'}</strong>
              <p>{item.note}</p>
            </div>
          ))}
        </div>
        <div className="citation-panel">
          <div className="citation-head">
            <h3>Citations</h3>
            <span>{filteredCitationItems.length}</span>
          </div>
          <Input size="small" value={citationQuery} onChange={(event) => setCitationQuery(event.target.value)} placeholder="Filter sources" />
          <div className="citation-list">
            {visibleCitationItems.map(({ source, sourceIndex }) => (
              <div className="citation-item" key={String(source?.url || source?.citation_span_id || sourceIndex)}>
                <button type="button" className="citation-insert" onClick={() => insertCitation(source, sourceIndex)}>
                  <span>{sourceIndex + 1}</span>
                  <strong>{sourceTitle(source, sourceIndex)}</strong>
                  <small>{source?.citation_span_id || source?.canonical_item_id || 'Evidence source'} · click to insert or bind</small>
                </button>
                {source.url && <a className="citation-open" href={source.url} target="_blank" rel="noreferrer" title="Open source"><LinkOutlined /></a>}
              </div>
            ))}
          </div>
          {!filteredCitationItems.length && <Empty description="No sources" />}
          {filteredCitationItems.length > citationPageSize && (
            <div className="citation-pager">
              <Button size="small" onClick={() => setCitationPage((page) => Math.max(1, page - 1))} disabled={safeCitationPage <= 1}>Prev</Button>
              <span>{safeCitationPage} / {citationPageCount}</span>
              <Button size="small" onClick={() => setCitationPage((page) => Math.min(citationPageCount, page + 1))} disabled={safeCitationPage >= citationPageCount}>Next</Button>
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}
