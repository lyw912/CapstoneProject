function cloneJson(value) {
  if (!value || typeof value !== 'object') return value;
  return JSON.parse(JSON.stringify(value));
}

function cleanText(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

function textFromInlines(inlines) {
  if (!Array.isArray(inlines)) return '';
  return inlines.map((run) => String(run?.text || '')).join('');
}

function textFromEditorNode(node) {
  if (!node) return '';
  if (node.type === 'text') return String(node.text || '');
  return (node.content || []).map(textFromEditorNode).join('');
}

function markToEditorMark(mark) {
  if (!mark || !mark.type) return null;
  if (['bold', 'italic', 'underline', 'strike', 'code'].includes(mark.type)) return { type: mark.type };
  if (mark.type === 'highlight') {
    const color = mark.value || mark.color || mark.style?.backgroundColor || mark.style?.background || '#d9f99d';
    return { type: 'highlight', attrs: { color } };
  }
  if (mark.type === 'color') {
    const color = mark.value || mark.color || mark.style?.color;
    return color ? { type: 'textStyle', attrs: { color } } : null;
  }
  if (mark.type === 'link') {
    const attrs = {
      href: mark.href || mark.value || '',
      target: '_blank',
      rel: 'noopener noreferrer nofollow',
      title: mark.title || null,
      citationId: mark.citationId || mark.citation_id || mark.source_id || null,
      citationIndex: mark.citationIndex || mark.citation_index || null
    };
    return attrs.href ? { type: 'link', attrs } : null;
  }
  return null;
}

function inlinesToEditorContent(inlines) {
  if (!Array.isArray(inlines) || !inlines.length) return [];
  return inlines.map((run) => {
    const marks = (run?.marks || []).map(markToEditorMark).filter(Boolean);
    const node = { type: 'text', text: String(run?.text || '') };
    if (marks.length) node.marks = marks;
    return node;
  }).filter((node) => node.text);
}

function editorMarksToIrMarks(marks) {
  if (!Array.isArray(marks)) return [];
  return marks.map((mark) => {
    const attrs = mark.attrs || {};
    if (['bold', 'italic', 'underline', 'strike', 'code'].includes(mark.type)) return { type: mark.type };
    if (mark.type === 'highlight') return { type: 'highlight', value: attrs.color || '#d9f99d' };
    if (mark.type === 'textStyle' && attrs.color) return { type: 'color', value: attrs.color };
    if (mark.type === 'link' && attrs.href) {
      const link = { type: 'link', href: attrs.href };
      if (attrs.title) link.title = attrs.title;
      if (attrs.citationId) link.citationId = attrs.citationId;
      if (attrs.citationIndex) link.citationIndex = attrs.citationIndex;
      return link;
    }
    return null;
  }).filter(Boolean);
}

function editorContentToInlines(content) {
  const runs = [];
  const walk = (node) => {
    if (!node) return;
    if (node.type === 'text') {
      const text = String(node.text || '');
      if (!text) return;
      const run = { text };
      const marks = editorMarksToIrMarks(node.marks);
      if (marks.length) run.marks = marks;
      runs.push(run);
      return;
    }
    if (node.type === 'hardBreak') {
      runs.push({ text: '\n' });
      return;
    }
    (node.content || []).forEach(walk);
  };
  (content || []).forEach(walk);
  return runs.length ? runs : [{ text: '' }];
}

function lockedLabel(block) {
  const type = block?.type || 'block';
  const title = block?.title || block?.caption || block?.engine || block?.widgetType || block?.widgetId || '';
  return title ? 'Locked ' + type + ': ' + title : 'Locked ' + type + ' block';
}

function paragraphNodeFromText(text, attrs) {
  const node = { type: 'paragraph', attrs: attrs || {} };
  if (text) node.content = [{ type: 'text', text }];
  return node;
}

function blockToEditorNode(block, path) {
  const attrs = { irPath: path, irBlockType: block?.type || 'unknown' };
  if (!block || typeof block !== 'object') {
    return paragraphNodeFromText('', { ...attrs, irLocked: true });
  }
  if (block.type === 'heading') {
    return {
      type: 'heading',
      attrs: { ...attrs, level: Math.min(6, Math.max(1, Number(block.level) || 2)) },
      content: block.text ? [{ type: 'text', text: String(block.text) }] : []
    };
  }
  if (block.type === 'paragraph') {
    const content = inlinesToEditorContent(block.inlines);
    const node = { type: 'paragraph', attrs };
    if (content.length) node.content = content;
    return node;
  }
  if (block.type === 'list') {
    const listType = block.listType === 'ordered' ? 'orderedList' : 'bulletList';
    const items = Array.isArray(block.items) ? block.items : [];
    return {
      type: listType,
      attrs,
      content: items.map((itemBlocks) => {
        const first = Array.isArray(itemBlocks) ? itemBlocks.find((item) => item?.type === 'paragraph') : null;
        const content = inlinesToEditorContent(first?.inlines || [{ text: textFromInlines(first?.inlines) }]);
        return {
          type: 'listItem',
          content: [content.length ? { type: 'paragraph', content } : { type: 'paragraph' }]
        };
      })
    };
  }
  if (block.type === 'blockquote') {
    const child = (block.blocks || []).find((item) => item?.type === 'paragraph');
    const content = inlinesToEditorContent(child?.inlines);
    return {
      type: 'blockquote',
      attrs,
      content: [content.length ? { type: 'paragraph', content } : { type: 'paragraph' }]
    };
  }
  if (block.type === 'code') {
    return {
      type: 'codeBlock',
      attrs: { ...attrs, language: block.lang || null },
      content: block.content ? [{ type: 'text', text: String(block.content) }] : []
    };
  }
  if (block.type === 'hr') {
    return { type: 'horizontalRule', attrs };
  }
  return paragraphNodeFromText(lockedLabel(block), { ...attrs, irLocked: true });
}

export function reportIrToEditorJson(documentIr) {
  const nodes = [];
  const chapters = Array.isArray(documentIr?.chapters) ? documentIr.chapters : [];
  chapters.forEach((chapter, chapterIndex) => {
    (chapter.blocks || []).forEach((block, blockIndex) => {
      nodes.push(blockToEditorNode(block, 'chapters.' + chapterIndex + '.blocks.' + blockIndex));
    });
  });
  return { type: 'doc', content: nodes.length ? nodes : [paragraphNodeFromText('')] };
}

function getAtPath(root, path) {
  if (!path) return undefined;
  return String(path).split('.').reduce((node, key) => node?.[key], root);
}

function setAtPath(root, path, value) {
  const parts = String(path || '').split('.');
  if (!parts.length) return;
  let node = root;
  for (let index = 0; index < parts.length - 1; index += 1) {
    const key = parts[index];
    if (node?.[key] == null) return;
    node = node[key];
  }
  node[parts[parts.length - 1]] = value;
}

function listNodeToIrBlock(node, original) {
  const block = { ...(original || {}), type: 'list', listType: node.type === 'orderedList' ? 'ordered' : 'bullet' };
  block.items = (node.content || []).map((item) => {
    const paragraph = (item.content || []).find((child) => child.type === 'paragraph');
    return [{ type: 'paragraph', inlines: editorContentToInlines(paragraph?.content || []) }];
  });
  return block;
}

function editorNodeToIrBlock(node, original) {
  if (!node || node.attrs?.irLocked) return cloneJson(original);
  if (node.type === 'heading') {
    return {
      ...(original || {}),
      type: 'heading',
      level: Number(node.attrs?.level) || Number(original?.level) || 2,
      text: cleanText(textFromEditorNode(node)),
      anchor: original?.anchor || ''
    };
  }
  if (node.type === 'paragraph') {
    return { ...(original || {}), type: 'paragraph', inlines: editorContentToInlines(node.content || []) };
  }
  if (node.type === 'bulletList' || node.type === 'orderedList') return listNodeToIrBlock(node, original);
  if (node.type === 'blockquote') {
    const paragraph = (node.content || []).find((child) => child.type === 'paragraph');
    return { ...(original || {}), type: 'blockquote', blocks: [{ type: 'paragraph', inlines: editorContentToInlines(paragraph?.content || []) }] };
  }
  if (node.type === 'codeBlock') {
    return { ...(original || {}), type: 'code', lang: node.attrs?.language || original?.lang || '', content: textFromEditorNode(node) };
  }
  if (node.type === 'horizontalRule') return { ...(original || {}), type: 'hr' };
  return cloneJson(original);
}

export function editorJsonToReportIr(editorJson, originalIr) {
  const nextIr = cloneJson(originalIr || {});
  const nodes = editorJson?.content || [];
  nodes.forEach((node) => {
    const path = node?.attrs?.irPath;
    if (!path) return;
    const original = getAtPath(originalIr, path);
    const updated = editorNodeToIrBlock(node, original);
    if (updated) setAtPath(nextIr, path, updated);
  });
  return nextIr;
}

export function reportIrOutline(documentIr) {
  const items = [];
  const chapters = Array.isArray(documentIr?.chapters) ? documentIr.chapters : [];
  chapters.forEach((chapter, chapterIndex) => {
    (chapter.blocks || []).forEach((block, blockIndex) => {
      if (block?.type !== 'heading') return;
      const title = cleanText(block.text);
      if (!title) return;
      items.push({
        id: block.anchor || 'ir-heading-' + chapterIndex + '-' + blockIndex,
        level: 'h' + (block.level || 2),
        title,
        irPath: 'chapters.' + chapterIndex + '.blocks.' + blockIndex
      });
    });
  });
  return items.slice(0, 80);
}

export function citationMarkAttrs(source, index, title) {
  return {
    href: String(source?.url || '').trim(),
    target: '_blank',
    rel: 'noopener noreferrer nofollow',
    title: title || '',
    citationId: source?.citation_span_id || source?.canonical_item_id || '',
    citationIndex: String(index + 1)
  };
}
