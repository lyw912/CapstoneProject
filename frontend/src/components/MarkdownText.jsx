import DOMPurify from 'dompurify';
import { displayText, escapeHtml } from '../utils/helpers';

export default function MarkdownText({ value }) {
  const escaped = escapeHtml(displayText(value, 'No reading available'));
  const html = escaped
    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/__([^_]+)__/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/_([^_]+)_/g, '<em>$1</em>')
    .replace(/\n/g, '<br>');
  return <span className="markdown-text" dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(html) }} />;
}
