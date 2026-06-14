import { useState, useEffect } from 'react';
import {
  Button,
  Empty,
  Input,
  Tooltip,
  message
} from 'antd';
import {
  CommentOutlined,
  HighlightOutlined,
  LinkOutlined,
  CloudDownloadOutlined
} from '@ant-design/icons';
import { EditorContent, useEditor } from '@tiptap/react';
import { BubbleMenu } from '@tiptap/react/menus';
import StarterKit from '@tiptap/starter-kit';
import Highlight from '@tiptap/extension-highlight';
import Underline from '@tiptap/extension-underline';
import Link from '@tiptap/extension-link';
import Placeholder from '@tiptap/extension-placeholder';
import { TextStyle } from '@tiptap/extension-text-style';
import { Color } from '@tiptap/extension-color';
import { displayText, reportSeedHtml, timeText, downloadBlob, sourceTitle } from '../utils/helpers';

export default function ReviewEditor({ output, reportHtml, onReportHtmlChange, annotations, setAnnotations }) {
  const [commentDraft, setCommentDraft] = useState('');
  const [bubbleNoteOpen, setBubbleNoteOpen] = useState(false);
  const [pendingSelection, setPendingSelection] = useState(null);
  const citationSources = output?.source_data?.query_agent?.top_sources || [];
  const editor = useEditor({
    extensions: [
      StarterKit.configure({ link: false, underline: false }),
      Underline,
      TextStyle,
      Color.configure({ types: ['textStyle'] }),
      Highlight.configure({ multicolor: true }),
      Link.configure({ openOnClick: true, autolink: true, linkOnPaste: true }),
      Placeholder.configure({ placeholder: 'Draft, revise, cite.' })
    ],
    content: reportHtml || reportSeedHtml(output),
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
    onUpdate: ({ editor: current }) => onReportHtmlChange(current.getHTML())
  });

  useEffect(() => {
    if (!editor) return;
    const next = reportHtml || reportSeedHtml(output);
    if (next && next !== editor.getHTML()) {
      editor.commands.setContent(next, false);
    }
  }, [editor, output, reportHtml]);

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
      id: `note_${Date.now()}`,
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
    editor.chain().focus().setTextSelection({ from: selection.from, to: selection.to }).extendMarkRange('link').setLink({ href: url }).run();
  };

  return (
    <div className="review-shell">
      <div className="editor-surface">
        <div className="editor-toolbar">
          <Tooltip title="Heading"><Button onClick={() => editor?.chain().focus().toggleHeading({ level: 2 }).run()}>H2</Button></Tooltip>
          <Tooltip title="Subhead"><Button onClick={() => editor?.chain().focus().toggleHeading({ level: 3 }).run()}>H3</Button></Tooltip>
          <Tooltip title="Bold"><Button icon={<strong>B</strong>} onClick={() => editor?.chain().focus().toggleBold().run()} /></Tooltip>
          <Tooltip title="Italic"><Button icon={<em>I</em>} onClick={() => editor?.chain().focus().toggleItalic().run()} /></Tooltip>
          <Tooltip title="Underline"><Button icon={<u>U</u>} onClick={() => editor?.chain().focus().toggleUnderline().run()} /></Tooltip>
          <Tooltip title="Highlight"><Button icon={<HighlightOutlined />} onClick={() => highlightSelection('#d9f99d')} /></Tooltip>
          <Tooltip title="Source link"><Button icon={<LinkOutlined />} onClick={setLink} /></Tooltip>
          <Tooltip title="Accent text"><Button className="swatch-button green" onClick={() => editor?.chain().focus().setColor('var(--hku)').run()} /></Tooltip>
          <Tooltip title="Export HTML"><Button icon={<CloudDownloadOutlined />} onClick={() => downloadBlob(editor?.getHTML() || '', 'reviewed-report.html')}>Export</Button></Tooltip>
        </div>
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
      <aside className="annotation-panel">
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
          <h3>Citations</h3>
          {citationSources.slice(0, 6).map((source, index) => (
            <a key={`${source.url || index}`} href={source.url} target="_blank" rel="noreferrer">
              <span>{String(index + 1).padStart(2, '0')}</span>
              <strong>{sourceTitle(source, index)}</strong>
            </a>
          ))}
          {!citationSources.length && <Empty description="No sources" />}
        </div>
      </aside>
    </div>
  );
}
