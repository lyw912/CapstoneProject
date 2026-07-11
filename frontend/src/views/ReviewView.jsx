import { useState } from 'react';
import { Alert, Button, Progress, Space, Tag, Timeline, message } from 'antd';
import { motion } from 'framer-motion';
import {
  FileTextOutlined,
  CloudDownloadOutlined,
  BookOutlined,
  FilePdfOutlined
} from '@ant-design/icons';
import SectionTitle from '../components/SectionTitle';
import ReviewEditor from '../components/ReviewEditor';
import { apiJson, downloadBlob } from '../utils/helpers';

function filenameFromDisposition(disposition, fallback) {
  const match = String(disposition || '').match(/filename="?([^";]+)"?/i);
  return match?.[1] || fallback;
}

async function downloadFetchResponse(response, fallbackName) {
  if (!response.ok) {
    const text = await response.text();
    let messageText = text || 'Export failed';
    try {
      const payload = JSON.parse(text);
      messageText = payload.error || payload.message || messageText;
    } catch {
      // Keep raw text.
    }
    throw new Error(messageText);
  }
  const blob = await response.blob();
  const filename = filenameFromDisposition(response.headers.get('Content-Disposition'), fallbackName);
  const href = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = href;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(href);
}

export default function ReviewView({ output, reportTask, reportHtml, setReportHtml, documentIr, setReportIr, annotations, setAnnotations, theme, reportEvents, generateReport }) {
  const [exporting, setExporting] = useState('');
  const hasGeneratedReport = Boolean(reportTask?.has_result || reportTask?.report_file_ready || (reportTask?.status === 'completed' && reportTask?.task_id));
  const canExport = Boolean(documentIr || reportTask?.task_id);

  const fallbackDownload = (path) => {
    if (!reportTask?.task_id) return;
    window.location.assign(path.replace(':taskId', reportTask.task_id));
  };

  const exportHtml = async () => {
    if (!documentIr) {
      fallbackDownload('/api/report/download/:taskId');
      return;
    }
    setExporting('html');
    try {
      const data = await apiJson('/api/report/render-ir', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_ir: documentIr })
      });
      if (data.document_ir) setReportIr(data.document_ir);
      if (data.html_content) setReportHtml(data.html_content);
      downloadBlob(data.html_content || '', 'reviewed-report.html', 'text/html');
    } catch (error) {
      message.error(error.message || 'HTML export failed');
    } finally {
      setExporting('');
    }
  };

  const exportMarkdown = async () => {
    if (!documentIr) {
      fallbackDownload('/api/report/export/md/:taskId');
      return;
    }
    setExporting('md');
    try {
      const response = await fetch('/api/report/export/md-from-ir', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_ir: documentIr })
      });
      await downloadFetchResponse(response, 'reviewed-report.md');
    } catch (error) {
      message.error(error.message || 'Markdown export failed');
    } finally {
      setExporting('');
    }
  };

  const exportPdf = async () => {
    if (!documentIr) {
      fallbackDownload('/api/report/export/pdf/:taskId');
      return;
    }
    setExporting('pdf');
    try {
      const response = await fetch('/api/report/export/pdf-from-ir', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_ir: documentIr, optimize: true })
      });
      await downloadFetchResponse(response, 'reviewed-report.pdf');
    } catch (error) {
      message.error(error.message || 'PDF export failed');
    } finally {
      setExporting('');
    }
  };

  return (
    <motion.section key="review" className="review-page" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
      <div className="review-header studio-card">
        <SectionTitle eyebrow="Report board" title="Edit, highlight, and annotate the final narrative" action={<Tag color={hasGeneratedReport ? 'green' : 'gold'}>{hasGeneratedReport ? 'Full report' : 'Draft seed'}</Tag>} />
        <Space wrap>
          <Button type={hasGeneratedReport ? 'default' : 'primary'} icon={<FileTextOutlined />} onClick={generateReport}>Generate Report</Button>
          <Button icon={<CloudDownloadOutlined />} disabled={!canExport} loading={exporting === 'html'} onClick={exportHtml}>HTML</Button>
          <Button icon={<BookOutlined />} disabled={!canExport} loading={exporting === 'md'} onClick={exportMarkdown}>Markdown</Button>
          <Button icon={<FilePdfOutlined />} disabled={!canExport} loading={exporting === 'pdf'} onClick={exportPdf}>PDF</Button>
        </Space>
      </div>
      {!hasGeneratedReport && <Alert type="warning" showIcon message="Analysis draft only" description="This editor is showing a Coordinator-based draft until a full ReportEngine report is generated." />}
      {reportTask?.status === 'running' && <Alert type="info" showIcon message="Report generation is running" description={<Progress percent={reportTask.progress || 0} strokeColor={theme.primary} />} />}
      <ReviewEditor output={output} reportHtml={reportHtml} onReportHtmlChange={setReportHtml} documentIr={documentIr} onDocumentIrChange={setReportIr} annotations={annotations} setAnnotations={setAnnotations} hasGeneratedReport={hasGeneratedReport} reportTaskId={reportTask?.task_id} />
      <div className="studio-card event-card">
        <SectionTitle eyebrow="Generation stream" title="Recent writing events" />
        <Timeline items={(reportEvents.length ? reportEvents : [{ message: 'No report events yet', type: 'idle' }]).slice(0, 8).map((item) => ({ color: item.type === 'error' ? 'red' : item.type === 'warning' ? 'gold' : 'green', children: <span>{item.message}</span> }))} />
      </div>
    </motion.section>
  );
}
